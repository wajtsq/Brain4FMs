import os
import pickle
import sys
sys.path.append('/bench-mark')
import scipy.io as sio
import numpy as np
import warnings

from data_preprocess.seed_IV_preprocess.config import PreprocessArgs
from data_preprocess.seed_IV_preprocess.utils import _generate_subject_groups, _merge_data, _segment_data

warnings.filterwarnings("ignore", category=RuntimeWarning)

session_label = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                  [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                  [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]

useless_ch = ['M1', 'M2', 'VEO', 'HEO']

def generate_sub_subject_data(args):
    for root, dirs, _ in os.walk(args.data_root):
        for dir in dirs:
            folder = os.path.join(root, dir)
            files = os.listdir(folder)
            session_labels = session_label[int(dir)-1]
            for file in files:
                if '.mat' in file:
                    file_name = file.split('.')[0]
                    subject, name = file_name.split('_')
                    raw = sio.loadmat(os.path.join(folder, file))
                    
                    data, labels = [], []
                    for key in raw.keys():
                        if '_eeg' not in key:
                            continue
                        count = int(key[-1])-1
                        
                        sub_data = raw[key]
                        sub_data = _segment_data(args, sub_data)
                        data.append(sub_data)
                        label = session_labels[count]
                        labels += [label for _ in range(sub_data.shape[0])]
                        
                    data = np.concatenate(data, axis=0)  
                    labels = np.array(labels)
                    save_dir = os.path.join(args.data_save_dir, dir)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                            
                    np.save(os.path.join(save_dir, f'{subject}_{int(dir)}_data.npy'), data)
                    np.save(os.path.join(save_dir, f'{subject}_{int(dir)}_label.npy'), labels)
                    
                    print(f'data of subject {subject} {dir} saved')


def generate_subject_data(args):
    generate_sub_subject_data(args)
    dirs = os.listdir(args.data_save_dir)            
    for sub in range(1, args.subject_num+1):
        datas, labels = [], []
        for dir in dirs:
            if '.npy' in dir or 'group' in dir or '.pkl' in dir:
                continue
            data = np.load(os.path.join(args.data_save_dir, dir, f'{sub}_{dir}_data.npy'))
            label = np.load(os.path.join(args.data_save_dir, dir, f'{sub}_{dir}_label.npy'))
            datas.append(data)
            labels.append(label)
            
        datas = np.concatenate(datas, axis=0)  
        labels = np.concatenate(labels, axis=0)  
        np.save(os.path.join(args.data_save_dir, f'{sub}_data.npy'), datas)
        np.save(os.path.join(args.data_save_dir, f'{sub}_label.npy'), labels)
    

def generate_group_data(args):
    subject_groups = _generate_subject_groups(args)
            
    for i, g in enumerate(subject_groups):
        data, label = _merge_data(args, g)
        save_dir = os.path.join(args.data_save_dir, 'group_data')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, f'group_{i}_data.npy'), data)
        np.save(os.path.join(save_dir, f'group_{i}_label.npy'), label)

        print(f'data of group {i} is saved')


def merge_group_data(args):
    data, label = [], []
    for g_id in range(args.group_num):
        data.append(np.load(os.path.join(args.data_save_dir,  f'group_data/group_{g_id}_data.npy')))
        label.append(np.load(os.path.join(args.data_save_dir, f'group_data/group_{g_id}_label.npy')))
    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)
    np.save(os.path.join(args.data_save_dir, f'group_data/all_data.npy'), data)
    np.save(os.path.join(args.data_save_dir, f'group_data/all_label.npy'), label)


def store_channels(args):
    import json
    channel_file = os.path.join(args.data_save_dir, 'group_data', 'channels_lst.json')
    channels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', \
                'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', \
                'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', \
                'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', \
                'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
    if not os.path.exists(channel_file):
        with open(channel_file, 'w') as f:
            json.dump(channels, f)
                    

args = PreprocessArgs()
store_channels(args)
generate_subject_data(args)
generate_group_data(args)
# merge_group_data(args)