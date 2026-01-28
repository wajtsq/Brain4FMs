import os
import sys
sys.path.append('/bench-mark')
import json
import numpy as np
import warnings
import mne

from data_preprocess.EEGMMIDB_preprocess.config import PreprocessArgs
from data_preprocess.EEGMMIDB_preprocess.utils import _generate_subject_groups, _merge_data, _segment_data

warnings.filterwarnings("ignore", category=RuntimeWarning)

task_dict = {'T0': [1, 2],      # rest
             'RS': [3, 7, 11],  # real singel left(T1)/ right(T2) fist
             'RB': [5, 9, 13],  # real both fists(T1)/ feet(T2)
             'IS': [4, 8, 12],  # imagine singel left(T1)/ right(T2) fist
             'IB': [6, 10, 14], # imagine both fists(T1)/ feet(T2)
            }


def generate_sub_data(args, task):
    channel_file_I = os.path.join(args.data_save_dir, f'group_data_I', 'channels_lst.json')
    channel_file_R = os.path.join(args.data_save_dir, f'group_data_R', 'channels_lst.json')
    
    for root, dirs, _ in os.walk(args.data_root):
        for dir in dirs:
            if 'S' not in dir:
                continue
            run_list = task_dict[task]
            data_list = {'T1': [], 'T2': []}
            for run in run_list:
                file_path = os.path.join(root, dir, f'{dir}R{run:02}.edf')
                raw = mne.io.read_raw_edf(file_path, preload=True)
                
                if not os.path.exists(channel_file_I):
                    channels = raw.info['ch_names']
                    channels = [name.split('.')[0].upper() for name in channels]
                    with open(channel_file_I, 'w') as f:
                        json.dump(channels, f)
                    with open(channel_file_R, 'w') as f:
                        json.dump(channels, f)
                
                data = raw.get_data()
                ch_num, ch_len = data.shape
                data = _segment_data(args, data)
                time_list = raw.annotations.duration
                task_list = raw.annotations.description
                time_counter = 0
                for i, task_id in enumerate(task_list):
                    if task_id == 'T0':
                        time_counter += time_list[i]
                        continue
                    start = time_counter
                    end = start + time_list[i]
                    task_data = data[:, int(args.sfreq*start):int(args.sfreq*end)]
                    if task_data.shape[1] < int(args.sfreq*args.seq_len):
                        time_counter = end
                        break
                    task_data = task_data[:, :int(args.sfreq*args.seq_len)].reshape(1, ch_num, -1)
                    data_list[task_id].append(task_data)
                    time_counter = end
                
            T1_data = np.concatenate(data_list['T1'], axis=0)
            T2_data = np.concatenate(data_list['T2'], axis=0)
            np.save(os.path.join(args.data_save_dir, f'{dir}_{task}_T1.npy'), T1_data)
            np.save(os.path.join(args.data_save_dir, f'{dir}_{task}_T2.npy'), T2_data)
            print(f'subject {dir} {task} data saved')
    


    
def generate_subject_data(args, task):
    for subject in range(args.subject_num):
        label = []
        data_S_T1 = np.load(os.path.join(args.data_save_dir, f'S{(subject+1) :03d}_{task}S_T1.npy'))
        label += [0]*data_S_T1.shape[0]
        data_S_T2 = np.load(os.path.join(args.data_save_dir, f'S{(subject+1) :03d}_{task}S_T2.npy'))
        label += [1]*data_S_T2.shape[0]
        data_B_T1 = np.load(os.path.join(args.data_save_dir, f'S{(subject+1) :03d}_{task}B_T1.npy'))
        label += [2]*data_B_T1.shape[0]
        data_B_T2 = np.load(os.path.join(args.data_save_dir, f'S{(subject+1) :03d}_{task}B_T2.npy'))
        label += [3]*data_B_T2.shape[0]
        
        data = np.concatenate((data_S_T1, data_S_T2, data_B_T1, data_B_T2), axis=0)
        label = np.array(label)
        np.save(os.path.join(args.data_save_dir, f'S{(subject+1) :03d}_{task}_data.npy'), data)
        np.save(os.path.join(args.data_save_dir, f'S{(subject+1) :03d}_{task}_label.npy'), label)
        
        print(f'data of subject {subject} is saved')
    
    
def generate_group_data(args, task):
    subject_groups = _generate_subject_groups(args)
            
    for i, g in enumerate(subject_groups):
        data, label = _merge_data(args, g, task)
        save_dir = os.path.join(args.data_save_dir, f'group_data_{task}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, f'group_{i}_data.npy'), data)
        np.save(os.path.join(save_dir, f'group_{i}_label.npy'), label)

        print(f'data of group {i} is saved')


def merge_group_data(args, task):
    data, label = [], []
    for g_id in range(args.group_num):
        data.append(np.load(os.path.join(args.data_save_dir,  f'group_data_{task}/group_{g_id}_data.npy')))
        label.append(np.load(os.path.join(args.data_save_dir, f'group_data_{task}/group_{g_id}_label.npy')))
    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)
    np.save(os.path.join(args.data_save_dir, f'group_data_{task}/all_data.npy'), data)
    np.save(os.path.join(args.data_save_dir, f'group_data_{task}/all_label.npy'), label)



args = PreprocessArgs()
generate_sub_data(args, 'RS')
generate_sub_data(args, 'RB')
generate_sub_data(args, 'IS')
generate_sub_data(args, 'IB')

generate_subject_data(args, 'R')    # real
generate_subject_data(args, 'I')    # imagine
generate_group_data(args, 'R')
generate_group_data(args, 'I')
# merge_group_data(args)