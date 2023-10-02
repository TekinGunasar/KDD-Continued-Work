from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

from src.preprocessing import *
from src.dataset_wrappers import *
from random import randrange

from tqdm.notebook import tqdm
from math import ceil
import numpy as np
import os


class DeepEmbeddedClustering():


    #Experiment directory passed in assumes that there is already a pre-trained auto-encoder there. 
    def __init__(self,dataset_path,TRAINING_SETTINGS_JSON,experiment_dir):

        self.save_results_to = os.path.join(experiment_dir,'DEC Training Results')
        if not os.path.isdir(self.save_results_to):
            print(f'Creating directory where DEC results will be saved in {self.save_results_to}')
            os.makedirs(self.save_results_to,exist_ok=True)
        
        #clearing training history to make sure loss from other experiments is not included in subsequent experimens
        self.history = {
            'loss': {
                'training':[],
                'validation':[]
            },
            'accuracy': {
                'training':[],
                'validation':[]
            }
        }

        self.dataset_path = dataset_path
        self.experiment_dir = experiment_dir
        
        #lrs = learning rate scheduler
        self.has_lrs = False

        training_settings = load_training_settings(TRAINING_SETTINGS_JSON)

        if len(training_settings) == 3:
            self.has_lrs = True
            self.TRAINING_SETTINGS,self.DATA_FORMAT,self.LR_SCHEDULER = training_settings

        self.TRAINING_SETTINGS,self.DATA_FORMAT = training_settings

        self.parse_dataset()


    
    def parse_dataset(self,normalize=True,scaler=StandardScaler()):
        data_dict = load_dataset(self.dataset_path)

        self.training_trials = normalize_data(data_dict['training']['trials'],scaler)
        self.training_labels = LabelEncoder().fit_transform(data_dict['training']['labels'])

        self.validation_trials = normalize_data(data_dict['validation']['trials'],scaler)
        self.validation_labels = LabelEncoder().fit_transform(data_dict['validation']['labels'])

        self.testing_trials = normalize_data(data_dict['testing']['trials'],scaler)
        self.testing_labels = LabelEncoder().fit_transform(data_dict['testing']['labels'])
        

    
    def load_auto_encoder(self,auto_encoder_name = 'auto_encoder.keras'):
        auto_encoder_path = os.path.join(self.experiment_dir,auto_encoder_name)
        self.auto_encoder = load_model(auto_encoder_path)

    
    def extract_encoder(self):
        encoder_idx = len(self.auto_encoder.layers) // 2 + 1
        encoder_layers = self.auto_encoder.layers[:encoder_idx]
        
        self.encoder = Sequential(encoder_layers)
        input_shape = input_shape = [None,self.DATA_FORMAT['NUM_CHANNELS'],self.DATA_FORMAT['TRIAL_LENGTH'],1]

        self.encoder.build(input_shape = input_shape)


    def get_embeddings(self,x_batch,verbose=False):
        if x_batch.ndim == 2:
            if verbose:
                print('You have inputted an individual trial. Most TensorFlow functionality assumes you are passing in a batch of data. Reformatting your input...')

            x_batch = np.expand_dims(np.reshape(x_batch, [
                self.DATA_FORMAT['NUM_CHANNELS'],self.DATA_FORMAT['TRIAL_LENGTH'],1
            ]),axis=0)

            return self.encoder(x_batch)

        elif x_batch.ndim == 3:
            if verbose:
                print('You have inputted a numpy array representing a batch of examples... expanding dimension to ndim = 4 for TensorFlow compatability...')

            x_batch = np.expand_dims(x_batch,axis=x_batch.ndim)
            return self.encoder(x_batch)

        return self.encoder(x_batch)

    def initialize_cluster_centers(self,embedded_data):
        self.cluster_centers =  tf.cast(tf.Variable(KMeans(n_clusters=2).fit(embedded_data).cluster_centers_),tf.float32)
        
    def compile_encoder(self,loss,optimizer):
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer.lr = self.TRAINING_SETTINGS['learning_rate']
        
        self.encoder.compile(loss = self.loss,optimizer = self.optimizer)

    
    
    def soft_assignments(self,embedded_data, cluster_centers, alpha=1):
        # Calculate pairwise distances between embedded data and cluster centers
        pairwise_distances = tf.reduce_sum(tf.square(embedded_data[:, tf.newaxis] - cluster_centers), axis=-1)
    
        # Compute the t-distribution kernel
        kernel = tf.pow(1 + pairwise_distances / alpha, -(alpha + 1) / 2)
        kernel = kernel / (tf.reduce_sum(kernel, axis=-1, keepdims=True))
    
        return kernel

    
    def target_distribution(self,kernel):
        # Normalize the kernel to obtain the target distribution
        q = tf.square(kernel) / tf.reduce_sum(kernel, axis=0, keepdims=True)
        q = q / tf.reduce_sum(q, axis=1, keepdims=True)
    
        return q


    #cm = cluster membership
    def get_acc_by_cm(self,embeddings,cluster_centers,labels):
        soft_assignments = self.soft_assignments(embeddings,cluster_centers)
        predicted_labels = tf.math.argmax(soft_assignments,axis=-1).numpy()
        
        acc = accuracy_score(labels,predicted_labels)
        
        return acc

    
    def evaluate_training_loss(self):
        training_embeddings = self.get_embeddings(self.training_trials)
        
        soft_assignments = self.soft_assignments(training_embeddings,self.cluster_centers)
        target_distribution = self.target_distribution(soft_assignments)
        
        training_kld = self.loss(soft_assignments,target_distribution)
        
        return training_kld

    
    def evaluate_validation_loss(self):
        validation_embeddings = self.get_embeddings(self.validation_trials)
        
        soft_assignments = self.soft_assignments(validation_embeddings,self.cluster_centers)
        target_distribution = self.target_distribution(soft_assignments)
        
        val_kld = self.loss(soft_assignments,target_distribution)
        
        return val_kld

    def evaluate_training_accuracy(self):
        training_embeddings = self.get_embeddings(self.testing_trials)
        
        soft_assignments = self.soft_assignments(training_embeddings,self.cluster_centers)
        target_distribution = self.target_distribution(soft_assignments)
        
        train_acc = self.get_acc_by_cm(training_embeddings,self.cluster_centers,self.testing_labels)
        
        return train_acc

    def evaluate_validation_accuracy(self):
        validation_embeddings = self.get_embeddings(self.validation_trials)
        
        soft_assignments = self.soft_assignments(validation_embeddings,self.cluster_centers)
        target_distribution = self.target_distribution(soft_assignments)
        
        val_acc = self.get_acc_by_cm(validation_embeddings,self.cluster_centers,self.validation_labels)
        
        return val_acc

    
    def train_step(self,x_batch_train,y_batch_train):

        with tf.GradientTape(persistent = True) as tape:

            tape.watch(self.cluster_centers)

            cur_embeddings = self.encoder(x_batch_train)

            soft_assignments = self.soft_assignments(cur_embeddings,self.cluster_centers)
            target_distribution = self.target_distribution(soft_assignments)
            
            loss = self.loss(soft_assignments,target_distribution)

        grad_encoder_params = tape.gradient(loss,self.encoder.trainable_weights)
        grad_cluster_centers = tape.gradient(loss,self.cluster_centers)

        self.optimizer.apply_gradients(zip(grad_encoder_params,self.encoder.trainable_weights))
        self.cluster_centers += self.optimizer.lr * self.cluster_centers
        
            
    def train(self,save_experiment = False,model_name = None):

        initial_training_acc = self.evaluate_training_accuracy()
        initial_validation_acc = self.evaluate_validation_accuracy()

        print(f'Initial training accuracy (by soft assignments): {initial_training_acc}')
        print(f'Initial validation accuracy (by soft assignments): {initial_validation_acc}\n')

        
        train_trials_as_tf_dataset = create_tf_dataset(
            trials = self.training_trials,
            labels = self.training_labels,
            batch = True,
            batch_size = self.TRAINING_SETTINGS['BATCH_SIZE'],
            include_labels = True
        )

        num_epochs = self.TRAINING_SETTINGS['NUM_EPOCHS']
        
        steps_per_epoch = ceil(len(self.training_trials) / self.TRAINING_SETTINGS['BATCH_SIZE'])

        print(f'Training steps per epoch: ~{steps_per_epoch}\n')
        
        for epoch in range(num_epochs):
            
            epoch_progress = tqdm(total=len(range(steps_per_epoch)), unit=' batch', position=0, leave=False)
            epoch_progress.set_description(desc=f'Epoch {epoch + 1} / {num_epochs}') 
            
            for step,(x_batch_train,y_batch_train) in enumerate(train_trials_as_tf_dataset):    
                self.train_step(x_batch_train,y_batch_train)
                epoch_progress.update(1)

           
            validation_loss = self.evaluate_validation_loss()
            self.history['loss']['validation'].append(validation_loss)
            
            train_loss = self.evaluate_training_loss()
            self.history['loss']['training'].append(train_loss)

            train_acc = self.evaluate_training_accuracy()
            self.history['accuracy']['training'].append(train_acc)
        
            val_acc = self.evaluate_validation_accuracy()
            self.history['accuracy']['validation']
            
            print(f'Training loss: {train_loss}')
            print(f'Validation loss: {validation_loss}\n')

            print(f'Training accuracy (by soft assignments): {train_acc}')
            print(f'Validation accuracy (by soft assignments): {val_acc}')

        self.plot_metrics()

    
    def plot_metrics(self):
        
        fig,axs = plt.subplots(1,2,figsize=(15,4))

        axs[0].plot(self.history['loss']['training'])
        axs[0].plot(self.history['loss']['validation'])

        axs[0].set_title('Loss Curve (BCE)')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('BCE(S,T)')

        axs[0].legend(['Training','Validation'])


        
        axs[1].plot(self.history['accuracy']['training'])
        axs[1].plot(self.history['accuracy']['validation'])
        
        axs[1].set_title('Accuracy Curve')
        
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')

        axs[1].legend(['Training','Accuracy'])

        

















