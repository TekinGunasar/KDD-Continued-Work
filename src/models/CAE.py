import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D,Conv2D,Dense,Reshape,BatchNormalization,InputLayer,Flatten,Dropout,LeakyReLU,ReLU,MaxPooling2D,MaxPooling1D,Conv1DTranspose,Conv2DTranspose
from tensorflow.keras import losses,optimizers
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score

from math import ceil,floor
import numpy as np
import pickle
import os

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from math import ceil, floor

import json

from src.preprocessing import *
from src.dataset_wrappers import *

from sklearn.cluster import KMeans

class ConvolutionalAutoEncoder:

    models_per_epoch = {}
    
    
    def __init__(self,dataset_path,TRAINING_SETTINGS_JSON):
        
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
        
        self.auto_encoder = Sequential()

        self.has_lrs = False
        
        training_settings = load_training_settings(TRAINING_SETTINGS_JSON)
        if len(training_settings) == 3:
            self.TRAINING_SETTINGS,self.DATA_FORMAT, self.LR_SCHEDULER = training_settings
            self.has_lrs = True

        else:
            self.TRAINING_SETTINGS,self.DATA_FORMAT = training_settings
        
        self.parse_dataset()

    
    def parse_dataset(self,scaler = StandardScaler()):
        data_dict = load_dataset(self.dataset_path)

        self.training_trials = normalize_data(data_dict['training']['trials'],scaler)
        self.training_labels = data_dict['training']['labels']

        self.validation_trials = normalize_data(data_dict['validation']['trials'],scaler)
        self.validation_labels = data_dict['validation']['labels']

        self.testing_trials = normalize_data(data_dict['testing']['trials'],scaler)
        self.testing_labels = data_dict['testing']['labels']
        

    def set_lr_scheduler(self,decay_steps_dilution=None):

        #Since we have a small dataset, doing LR decay every training step
        steps_per_epoch = ceil(len(self.training_trials) / self.TRAINING_SETTINGS['BATCH_SIZE'])
        num_epochs = self.TRAINING_SETTINGS['NUM_EPOCHS']

        n_decay_steps = steps_per_epoch * num_epochs

        #setting final learning rate to be one order of magnitude smaller than initial learning rate
        final_lr = self.TRAINING_SETTINGS['learning_rate'] * 0.1

        if self.LR_SCHEDULER['DECAY_TYPE'] == 'Polynomial':
            self.lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate = self.TRAINING_SETTINGS['learning_rate'],
                decay_steps = n_decay_steps,
                end_learning_rate = final_lr,
                power = self.LR_SCHEDULER['POWER']
            )

        elif self.LR_SCHEDULER['DECAY_TYPE'] == 'Exponential':
            self.lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = self.TRAINING_SETTINGS['learning_rate'],
                decay_steps = n_decay_steps,
                decay_rate = self.LR_SCHEDULER['POWER']
            )

    '''
        -TODO:

            Also fix dimension calculation after convolution outputs -> usually off by one? maybe need to use ceil/floor.
            
            Rewrite build_encoder and build_decoder so that there can be more than just one
            architecture -> i.e iteratively add layers to the encoder, and then the decoder 
            structure will be symmetric.

            Main challenge: Calculating shape of data after one spatial convolution and some
            n temporal convolutions.
    '''
    def build_encoder(self,show_summary = False):

        stride_length = self.TRAINING_SETTINGS['STRIDE_LENGTH']
        self.encoder = Sequential()

        #Input Layer
        input_shape = input_shape = [self.DATA_FORMAT['NUM_CHANNELS'],self.DATA_FORMAT['TRIAL_LENGTH'],1]
        self.encoder.add(InputLayer(input_shape = input_shape))


        ### Spatial Conv layers - start
        
        #Initial Spatial Conv Layer
        self.encoder.add(Conv2D(
            filters = self.TRAINING_SETTINGS['NUM_INITIAL_KERNELS'],
            kernel_size = (self.DATA_FORMAT['NUM_CHANNELS'],self.TRAINING_SETTINGS['KERNEL_WIDTH']),
            strides = (stride_length,stride_length)
        ))

        self.spatial_conv_output_shape = (
                (floor((self.DATA_FORMAT['TRIAL_LENGTH'] - self.TRAINING_SETTINGS['KERNEL_WIDTH']) / stride_length)+1),
                self.TRAINING_SETTINGS['NUM_INITIAL_KERNELS']
        )
        
        self.encoder.add(Reshape(target_shape =  self.spatial_conv_output_shape))
        self.encoder.add(LeakyReLU())

       
        #spatial conv layers - end

        #temporal conv -layers - start
 
        self.encoder.add(Conv1D(
            filters = self.TRAINING_SETTINGS['NUM_INITIAL_KERNELS'] * 2,
            kernel_size = self.TRAINING_SETTINGS['KERNEL_WIDTH'],
            strides = stride_length,
            groups = self.TRAINING_SETTINGS['NUM_INITIAL_KERNELS']
        ))

        self.temporal_conv_output_shape = (
            (floor((self.spatial_conv_output_shape[0] - self.TRAINING_SETTINGS['KERNEL_WIDTH']) / stride_length) + 1),
            self.TRAINING_SETTINGS['NUM_INITIAL_KERNELS'] * 2
        )

        self.encoder.add(Reshape(target_shape = self.temporal_conv_output_shape))
        self.encoder.add(LeakyReLU())
        
        #FC layers - start

        self.encoder.add(Flatten())
        self.encoder.add(Dense(self.TRAINING_SETTINGS['LATENT_DIMS'],activation='tanh'))
        
        #Fully connected layer - end

        
        if show_summary:
            print(self.encoder.summary())

    def build_decoder(self,show_summary = False):

        stride_length = self.TRAINING_SETTINGS['STRIDE_LENGTH']
        self.decoder = Sequential()

        #Going from latent space to FC layer 
        input_shape = (1,self.TRAINING_SETTINGS['LATENT_DIMS'])
        output_shape = self.DATA_FORMAT['TRIAL_LENGTH'] * self.DATA_FORMAT['NUM_CHANNELS']

        self.decoder.add(InputLayer(input_shape = input_shape))

        self.decoder.add(Dense(
            self.temporal_conv_output_shape[0] * self.temporal_conv_output_shape[1],activation='tanh'
        ))

        self.decoder.add(Reshape(
            target_shape = self.temporal_conv_output_shape
        ))
        
        self.decoder.add(Conv1DTranspose(
            filters = self.TRAINING_SETTINGS['NUM_INITIAL_KERNELS'],
            kernel_size = self.TRAINING_SETTINGS['KERNEL_WIDTH'],
            strides = stride_length,
        ))

        self.decoder.add(Reshape(
            target_shape = [1] + list(self.spatial_conv_output_shape) 
        ))

        self.decoder.add(Conv2DTranspose(
            filters = 1,
            kernel_size = (self.DATA_FORMAT['NUM_CHANNELS'],self.TRAINING_SETTINGS['KERNEL_WIDTH']),
            strides = (stride_length,stride_length),
        ))

        self.decoder.add(Reshape(
            target_shape = (self.DATA_FORMAT['NUM_CHANNELS'],self.DATA_FORMAT['TRIAL_LENGTH'])
        ))

        
    def build_auto_encoder(self,show_summary=False):

        self.build_encoder()
        self.build_decoder()
        
        self.auto_encoder = Sequential()

        input_shape = (self.DATA_FORMAT['NUM_CHANNELS'],self.DATA_FORMAT['TRIAL_LENGTH'],1)
        
        self.auto_encoder.add(InputLayer(
                    input_shape = input_shape         
                    ))               

        [self.auto_encoder.add(layer) for layer in self.encoder.layers]

        #Some issues adding decoder layers sometime, so this is done as a check incase issues come up
        for i,layer in enumerate(self.decoder.layers):

            try:
                self.auto_encoder.add(layer)
            except:
                print(f'Error when trying to add {layer} at index {i}')

        if show_summary:
            print(self.auto_encoder.summary())

    def compile_auto_encoder(self,loss,optimizer):

        self.loss = loss
        self.optimizer = optimizer

        self.optimizer.lr = self.TRAINING_SETTINGS['learning_rate']
        
        self.auto_encoder.compile(
            optimizer = optimizer,
            loss = loss
        )

        if self.has_lrs:
            self.set_lr_scheduler()


    def initialize_cluster_centers(self,embedded_data):
        cluster_centers =  tf.cast(tf.Variable(KMeans(n_clusters=2).fit(embedded_data).cluster_centers_),tf.float32)
        return cluster_centers

    def soft_assignments(self,embedded_data, cluster_centers, alpha=1):
        pairwise_distances = tf.reduce_sum(tf.square(embedded_data[:, tf.newaxis] - cluster_centers), axis=-1)
    
        kernel = tf.pow(1 + pairwise_distances / alpha, -(alpha + 1) / 2)
        kernel = kernel / (tf.reduce_sum(kernel, axis=-1, keepdims=True))
    
        return kernel

    def evaluate_accuracy(self,data,true_labels):

        embedded_data = self.encoder(data)
        cluster_centers = self.initialize_cluster_centers(embedded_data)
        soft_assignments = self.soft_assignments(embedded_data,cluster_centers)

        predicted_labels = tf.math.argmax(soft_assignments,axis=-1).numpy()
        
        acc = accuracy_score(true_labels,predicted_labels)

        return acc
    
    def evaluate_entire_dataset_loss(self,dataset):
        reconstructed_data = self.auto_encoder(dataset)
        loss = self.loss(dataset,reconstructed_data).numpy()

        return loss

    def update_encoder(self):
        encoder_idx = len(self.auto_encoder.layers) // 2 + 1
        encoder_layers = self.auto_encoder.layers[:encoder_idx]
        
        self.encoder = Sequential(encoder_layers)
        input_shape = input_shape = [None,self.DATA_FORMAT['NUM_CHANNELS'],self.DATA_FORMAT['TRIAL_LENGTH'],1]

        self.encoder.build(input_shape = input_shape)


    def train_step(self,x_batch_train,current_step):

        with tf.GradientTape() as tape:
            reconstructed_data = self.auto_encoder(x_batch_train)
    
            #Default loss is MSE
            cur_loss = self.loss(x_batch_train,reconstructed_data)

                   
        gradients = tape.gradient(cur_loss,self.auto_encoder.trainable_weights)
        
        #Gradient Clipping
        if not bool(self.TRAINING_SETTINGS['CLIP_GRADIENTS']):
            clip_threshold = self.TRAINING_SETTINGS['CLIP_VALUE']
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=clip_threshold)
            gradients = clipped_gradients

        assert type(gradients) is not None

        #lrs = learning rate scheduler
        if self.has_lrs:
            cur_lr = self.lr_scheduler(self.optimizer.iterations)
            self.optimizer.learning_rate.assign(cur_lr)
        
        self.optimizer.apply_gradients(zip(gradients,self.auto_encoder.trainable_weights))

    
    def train(self,save_experiment = False,experiment_dir = None,model_name = None):

        train_trials_as_tf_dataset = create_tf_dataset(
            trials = self.training_trials,
            batch = True,
            batch_size = self.TRAINING_SETTINGS['BATCH_SIZE']
        )

        if save_experiment and not experiment_dir:
            experiment_dir = input('Please provide an experiment directory')
        
        if save_experiment and not model_name:
            model_name = input('Please provide a model name with the .keras extension')

        if save_experiment and not os.path.isdir(experiment_dir):
            print(f'Creating directory {experiment_dir}\n')
            os.makedirs(experiment_dir,exist_ok = True)

        num_epochs = self.TRAINING_SETTINGS['NUM_EPOCHS']

        steps_per_epoch = ceil(len(self.training_trials) / self.TRAINING_SETTINGS['BATCH_SIZE'])
        
        print(f'Training steps per epoch: ~{steps_per_epoch}\n')
        
        for epoch in range(num_epochs):

            epoch_progress = tqdm(total=len(range(steps_per_epoch)), unit=' batch', position=0, leave=False)
            epoch_progress.set_description(desc=f'Epoch {epoch + 1}/{num_epochs}')
            
            for step,x_batch_train in enumerate(train_trials_as_tf_dataset):

                current_step = (epoch + 1) * step
                self.train_step(x_batch_train,current_step)

                epoch_progress.update(1)

            self.update_encoder()
            
            val_loss = self.evaluate_entire_dataset_loss(self.validation_trials)
            train_loss = self.evaluate_entire_dataset_loss(self.training_trials)

            val_acc = self.evaluate_accuracy(self.validation_trials,self.validation_labels)
            train_acc = self.evaluate_accuracy(self.training_trials,self.training_labels)
    
            print(f'Validation loss: {val_loss}')
            print(f'Training loss: {train_loss}\n')

            print(f'Validation accuracy: {val_acc}')
            print(f'Training accuracy: {train_acc}\n')

            self.history['loss']['training'].append(train_loss)
            self.history['loss']['validation'].append(val_loss)

            self.history['accuracy']['training'].append(train_acc)
            self.history['accuracy']['validation'].append(val_acc)
            
        

        if save_experiment:
            self.save_model(experiment_dir,model_name)
            self.evaluate_test_metrics()
            self.save_training_history_json(experiment_dir)
            self.save_training_settings(experiment_dir)
            self.plot_loss_curve(save_figure = True,experiment_dir = experiment_dir,figure_name = 'loss_curve.png')


    def save_model(self,experiment_dir,model_name):
        model_path = os.path.join(experiment_dir,model_name)
        self.auto_encoder.save(model_path)      

    
    def save_training_history_json(self,experiment_dir=None,json_name='training_history.json'):

        serializable_history = {
            'training_loss': convert_to_floats(self.history['loss']['training']),
            'validation_loss': convert_to_floats(self.history['loss']['validation']),
            'training_accuracy': self.history['accuracy']['training'],
            'validation_accuracy': self.history['accuracy']['validation'],
            'testing_loss': float(self.history['testing_loss']),
            'testing_accuracy': float(self.history['testing_accuracy'])
        }

        serializable_history_json = json.dumps(serializable_history)
        
        history_path = os.path.join(experiment_dir,json_name)

        with open(history_path,'w') as json_outfile:
            json_outfile.write(serializable_history_json)

        json_outfile.close()


    
    def save_training_settings(self,experiment_dir,json_name = 'TRAINING_SETTINGS.json'):

        if self.has_lrs:
            training_settings_json = json.dumps({
                'TRAINING_SETTINGS':self.TRAINING_SETTINGS,
                'LEARNING_RATE_SCHEDULER':self.LR_SCHEDULER
            })

        else:
            training_settings_json = json.dumps({
                'TRAINING_SETTINGS':self.TRAINING_SETTINGS
            })

        
        settings_path = os.path.join(experiment_dir,json_name)

        with open(settings_path,'w') as json_outfile:
            json_outfile.write(training_settings_json)

        json_outfile.close()

    

    def plot_loss_curve(self,experiment_dir=None,save_figure = False,figure_name=None):

        fig,axs = plt.subplots(1,2,figsize=(15,4))

        axs[0].plot(self.history['loss']['training'],alpha=0.4,lw = 5)
        axs[0].plot(self.history['loss']['validation'],alpha=0.6,lw = 5)

        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('MSE Loss')

        axs[0].set_title('Loss Curve')

        
        axs[1].plot(self.history['accuracy']['training'],alpha=0.4,lw = 5)
        axs[1].plot(self.history['accuracy']['validation'],alpha=0.6,lw = 5)

        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')

        axs[1].set_title('Accuracy Curve')

        if save_figure and not experiment_dir:
            print('Please provide an experiment directory to save the figure to')
            return

        elif save_figure and not figure_name:
            print('Please provide a figure name')
            return
        
        figure_path = experiment_dir + '/' + figure_name

        plt.savefig(figure_path,format='png')

    def evaluate_test_metrics(self):
        reconstructed_test_data = self.auto_encoder(self.testing_trials)
        test_mse = self.loss(self.testing_trials,reconstructed_test_data).numpy()
        test_acc = self.evaluate_accuracy(self.testing_trials,self.testing_labels)

        print('Adding test loss and test accuracy to training history...')
        print(f'Test loss is {test_mse}')
        print(f'Test accuracy is {test_acc}')

        self.history['testing_loss'] = test_mse
        self.history['testing_accuracy'] = test_acc
        

    
    
    #Only doing this for test data
    def show_random_reconstruction(self,show_noise=False):
        rand_idx = np.random.randint(low = 0,high = len(self.testing_trials) - 1)
        
        rand_test_trial = self.testing_trials[rand_idx]
        reconstructed_signal = self.auto_encoder(np.expand_dims(rand_test_trial,axis=0)).numpy()[0]

        rand_test_C3,rand_test_C4 = rand_test_trial[0],rand_test_trial[1]
        reconstructed_C3,reconstructed_C4 = reconstructed_signal[0],reconstructed_signal[1]
        
        removed_noise_C3 = reconstructed_C3 - rand_test_C3
        removed_noise_C4 = reconstructed_C4 - rand_test_C4

        
        fig,axs = plt.subplots(1,2,figsize=(15,4))

        axs[0].plot(rand_test_C3)
        axs[0].plot(rand_test_C4)
        axs[0].set_title('Original Signal')
        axs[0].legend(['C3','C4'])

        axs[1].plot(reconstructed_C3)
        axs[1].plot(reconstructed_C4)
        axs[1].set_title('Reconstructed Signal')
        axs[1].legend(['C3','C4'])

        fig_noise_C3,axs_noise_C3 = plt.subplots(1,1,figsize=(10,2))
        axs_noise_C3.plot(removed_noise_C3)
        axs_noise_C3.set_title('Removed noise C3')

        fig_noise_C4,axs_noise_C4 = plt.subplots(1,1,figsize=(10,2))
        axs_noise_C4.plot(removed_noise_C4)
        axs_noise_C4.set_title('Removed noise C4')

        fig_noise_hist,axs_noise_hist = plt.subplots(1,2,figsize=(15,4))
        axs_noise_hist[0].hist(removed_noise_C3,bins=15)

        axs_noise_hist[1].hist(removed_noise_C4,bins=15)


        

        
    
        







