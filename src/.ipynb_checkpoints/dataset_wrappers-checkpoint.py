### A collection of various miscellaneous helper functions we use

import numpy as np

import tensorflow as tf
import sys
import pickle
import json


def load_dataset(path):

    with open(path,'rb') as pickle_file:
        return pickle.load(pickle_file)
    
    pickle_file.close()

def load_training_settings(TRAINING_SETTINGS_JSON):

    json_data = json.load(open(TRAINING_SETTINGS_JSON,'r'))
    TRAINING_SETTINGS = json_data['TRAINING_SETTINGS']
    DATA_FORMAT = json_data['DATA_FORMAT']

    try:
        LEARNING_RATE_SCHEDULER = json_data['LEARNING_RATE_SCHEDULER']
        return TRAINING_SETTINGS,DATA_FORMAT,LEARNING_RATE_SCHEDULER
    except:
        return TRAINING_SETTINGS,DATA_FORMAT
    

#Extracts trials from TensorFlow TakeDataset type
#TakeDataset type much more efficient during training to batch and do efficient operations, but e.g to do something like 
#get loss for entire validation data, and not just a batch... not as easy
def convert_take_dataset(take_dataset,n_chans=2,trial_length=170):

    as_list = list(take_dataset.as_numpy_iterator())
    batch_size = len(as_list[0][0])
    
    n_trials = len(as_list) * batch_size

    concatenated_trials = np.ones(shape=(n_trials,n_chans,trial_length))
    concatenated_labels = np.ones(n_trials)

    for i in range(len(as_list)):

        try:
            current_batch,current_labels = as_list[i][0],as_list[i][1]
            
            start_slice = batch_size*i
            end_slice = start_slice + batch_size
            
            concatenated_trials[start_slice:end_slice] = current_batch
            concatenated_labels[start_slice:end_slice] = current_labels
        except ValueError:
            print('Ran into issues with ragged entries. Re-calling function')
            convert_take_dataset(take_dataset,n_chans,trial_length)
    
    return concatenated_trials,concatenated_labels







        


























    

    