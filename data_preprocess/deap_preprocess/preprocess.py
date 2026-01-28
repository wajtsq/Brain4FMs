import os
import pickle
import sys
sys.path.append('/bench-mark')
import pandas as pd
import numpy as np
import warnings
import mne

from data_preprocess.deap_preprocess.config import PreprocessArgs
from data_preprocess.deap_preprocess.utils import _generate_subject_groups, _merge_data, _get_labels, store_channels

warnings.filterwarnings("ignore", category=RuntimeWarning)


def generate_subject_data(args):
    args.data_root = os.path.join(args.data_root, 'data_preprocessed_python')
    
    for root, dirs, files in os.walk(args.data_root):
        for file in files:
            if '.dat' in file:
                subject = file.split('.')[0]
                with open(os.path.join(args.data_root, file), 'rb') as f:
                    data_dict = pickle.load(f, encoding='latin1')
                data = data_dict['data'][:, :args.ch_num, :args.sfreq*60]
                bsz, ch_num, ch_len = data.shape
                n = ch_len // (args.sfreq*args.seq_len)
                data = data.reshape(bsz, ch_num, args.sfreq*args.seq_len, -1)
                data = data.transpose(0, 3, 1, 2).reshape(bsz * n, ch_num, -1)
                
                # Valence Arousal Dominance Liking (1-9)
                labels = data_dict['labels']
                labels = _get_labels(args, labels, n)
                
                np.save(os.path.join(args.data_save_dir, f'{subject}_data.npy'), data)
                np.save(os.path.join(args.data_save_dir, f'{subject}_label.npy'), labels)
                
                print(f'data of subject {subject} saved')
        
    
    
def generate_group_data(args):
    subject_groups = _generate_subject_groups(args)
            
    for i, g in enumerate(subject_groups):
        data, label = _merge_data(args, g)
        save_dir = os.path.join(args.data_save_dir, 'group_data')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, f'group_{i}_data.npy'), data.astype(np.float32))
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



args = PreprocessArgs()
store_channels(args)
generate_subject_data(args)
generate_group_data(args)
# merge_group_data(args)