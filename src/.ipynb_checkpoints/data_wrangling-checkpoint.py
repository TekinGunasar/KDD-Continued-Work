from scipy.io import loadmat
import numpy as np
import os
import pickle
from tqdm import tqdm

from sklearn.model_selection import train_test_split

def load_mat(path):
    return loadmat(path)

def get_markers(path):
    #Reshaping markers to be in a more convenient form
    markers = load_mat(path)['o']['marker'][0][0]
    return np.reshape(markers,[markers.size,])

def get_sampling_rate(path):
    return load_mat(path)['o']['sampFreq'][0][0][0][0]

def get_raw_eeg_data(path):
    return load_mat(path)['o']['data'][0][0]

def get_input_lead_organization(path):
    return load_mat(path)['o'][0][0][-2]
    #C3 located at index 4
    #C4 located at index 5

def get_signal_onsets(path):
    markers = get_markers(path)
    sampling_rate = get_sampling_rate(path)

    signal_onset_indices = []
    signal_onset_type = []

    lastFound = 0
    for i in range(len(markers)):

        if markers[i] == 0 or markers[i] == 3:
            lastFound += 1
            continue

        elif markers[i] != 0 and lastFound > sampling_rate:
            lastFound = 0
            signal_onset_indices.append(i)
            signal_onset_type.append(markers[i])
    
    return signal_onset_indices,signal_onset_type

def get_C3_C4(path):
    raw_eeg = get_raw_eeg_data(path)
    C3 = raw_eeg[:,4]
    C4 = raw_eeg[:,5]
    C3_C4_eeg = np.array([C3,C4])

    return C3_C4_eeg

def get_mi_trials(path,trial_length=0.85):
    mi_trials = []

    C3_C4_eeg = get_C3_C4(path)

    signal_onset_indices,labels = get_signal_onsets(path)
    sampling_rate = get_sampling_rate(path)

    for index in signal_onset_indices:
        trial_start = index
        trial_end = trial_start + int(sampling_rate * trial_length)

        cur_trial_eeg = C3_C4_eeg[:,trial_start:trial_end]

        mi_trials.append(cur_trial_eeg)

    return mi_trials,labels
    

def organize_into_groups(marker_dict,trials,labels,average_signals=False):

    groups = set(marker_dict.values())
    num_groups = len(marker_dict)
    
    allocated_lists = [[] for _ in range(num_groups)]
    group_dict = {k:v for (k,v) in zip(groups,allocated_lists)}

    for i,trial in enumerate(trials):
        cur_marker = labels[i]
        cur_mi = marker_dict[cur_marker]

        group_dict[cur_mi].append(trial)

    if average_signals:
        averaged_group_dict = group_dict
        
        for group in groups:
            averaged_group_dict[group] = np.mean(group_dict[group],axis=0)

        return group_dict,averaged_group_dict



#maybe condense this... but only create data set once so maybe not worth the effort either.
def create_dataset(data_dir,save_to,train_split=0.7,val_split=0.2,test_split=0.1,trial_length=0.85,return_unread=False):

    ### First organizing data across all the files into X,y pairs
    dataset_dir = os.listdir(data_dir)
    X = []
    y = []

    unread_files = []

    for file in dataset_dir:

        try:

            cur_data_path = os.path.join(data_dir, file)

            print(f'Reading {cur_data_path}\n')
            cur_mi_trials,cur_labels = get_mi_trials(cur_data_path, trial_length)

            X += cur_mi_trials
            y += cur_labels

        except:
            print(f'Error Reading {cur_data_path}\n')
            unread_files.append(cur_data_path)

    X = np.array(X)
    y = np.array(y)

    ###


    ### splitting data into train, validation, and test groups

    X_train, X_val_and_test,y_train,y_val_and_test = train_test_split(
        X,y,test_size=val_split+test_split,random_state=42
    )

    val_split_size = val_split / (val_split + test_split) 

    print(f'Train trials length: {len(X_train)},Train labels length: {len(y_train)}')

    X_val,X_test,y_val,y_test = train_test_split(
        X_val_and_test,y_val_and_test,
        test_size = 1-val_split_size,
        random_state = 42
    )

    print(f'Validation trials length: {len(X_val)},Validation labels length: {len(y_val)}')
    print(f'Testing trials length: {len(X_test)},Testing labels length: {len(y_test)}')

    data_dict = {
        'training': {
            'trials':X_train,
            'labels':y_train
        },
        'validation': {
            'trials':X_val,
            'labels':y_val
        },
        'testing': {
            'trials':X_test,
            'labels':y_test
        }
    }

    ####


    ### Saving data -> for prospective users, make sure any experiments you do use the same data
    
    with open(save_to,'wb') as pickle_file:
        pickle.dump(data_dict,pickle_file)
    pickle_file.close()
    
    if unread_files and return_unread:
        print([f'Unable to read {file}' for file in unread_files],' this list of files is being returned');  
        return unread_files



    
















