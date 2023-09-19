from scipy.signal import periodogram
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline


#These assume C3,C4 channel set up



def interp_psd(original_freqs, target_freqs, original_power_spectrum):
    cs = CubicSpline(original_freqs, original_power_spectrum)
    interpolated_power_spectrum = cs(target_freqs)
    return interpolated_power_spectrum



def linear_to_decibel(linear_power):
    reference_power = 1.0
    decibel_power = 10 * np.log10(linear_power / reference_power)
    return decibel_power


#Default is mu rhythm
def zoom_into_psd(psd,zoom_range=(8,13),plot=False,return_data=False):
    psd_data_indices,psd_data = psd
    
    lb,ub = zoom_range
    
    lb_idx = np.argmax(psd_data_indices >= lb)
    ub_idx = np.argmax(psd_data_indices >= ub)

    relevant_indices = psd_data_indices[lb_idx:ub_idx]

    relevant_psd_data = np.array([
        psd_data[chan][lb_idx:ub_idx] for chan in range(len(psd_data))
    ])

    plt.plot(relevant_indices,relevant_psd_data[0])
    plt.plot(relevant_indices,relevant_psd_data[1])

    plt.legend(['C3','C4'])
    
    
    
def normalize_data(trials):

    num_channels,trial_length = trials.shape[1],trials.shape[2]
    flattened_trials = np.reshape(trials,[len(trials),num_channels*trial_length])

    normalized_trials = MinMaxScaler().fit_transform(flattened_trials)
    reformatted_trials = np.reshape(normalized_trials,[len(trials),num_channels,trial_length])

    return reformatted_trials

def create_tf_dataset(trials,labels=None,batch = False,batch_size = None,include_labels=False):

    if batch and not batch_size:
        print('Please provide a batch size')
        sys.exit(1)

    if include_labels:
        batch_tf_dataset = tf.data.Dataset.from_tensor_slices((trials,labels)).batch(batch_size).shuffle(len(trials))    
        return batch_tf_dataset
    
    #Since dataset is relatively small, setting shuffle buffer size to num. examples
    batch_tf_dataset = tf.data.Dataset.from_tensor_slices((trials)).batch(batch_size).shuffle(len(trials))    
    return batch_tf_dataset

        
def convert_to_floats(numpy_floats):
    to_float = lambda x : float(x)
    as_floats = list(map(to_float,numpy_floats))

    return as_floats