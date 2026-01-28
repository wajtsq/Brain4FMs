import os
import pickle
import sys
sys.path.append('/bench-mark')
import pandas as pd
import numpy as np
import warnings
import mne

from data_preprocess.EEGMat_preprocess.config import PreprocessArgs
from data_preprocess.EEGMat_preprocess.utils import _generate_subject_groups, _merge_data, _select_channels_from_edf, store_channel
from data_preprocess.utils import _segment_data

warnings.filterwarnings("ignore", category=RuntimeWarning)

quality_dict = {0: 'B', 1: 'G'}


def generate_sub_subject_data(args):
    meta_data = pd.read_csv(os.path.join(args.data_root, 'subject-info.csv'))
    
    for root, dirs, files in os.walk(args.data_root):
        for file in files:
            if '.edf' in file:
                raw = mne.io.read_raw_edf(os.path.join(root, file))
                raw = _select_channels_from_edf(raw)
                data = raw.get_data()[:, ::2]
                subject, label = (file.split('.')[0]).split('_')
                data = _segment_data(args, args.sfreq, data)
                
                # 0: bad quality count, 1: good quality count 
                quality_num = meta_data[meta_data['Subject'] == subject]['Count quality'].values[0]
                quality = quality_dict[quality_num.T]
                np.save(os.path.join(args.data_save_dir, f'{subject}_{quality}_{str(int(label)-1)}.npy'), data)
                
                print(f'data of subject {subject} saved')

    
def generate_subject_data(args):
    generate_sub_subject_data(args)
    meta_data = pd.read_csv(os.path.join(args.data_root, 'subject-info.csv'))
    for subject in range(args.subject_num):
        label = []
        quality_num = meta_data[meta_data['Subject'] == f'Subject{subject :02d}']['Count quality'].values[0]
        quality = quality_dict[quality_num.T]

        data_true = np.load(os.path.join(args.data_save_dir, f'Subject{subject :02d}_{quality}_1.npy'))
        label += [1]*data_true.shape[0]
        data_false = np.load(os.path.join(args.data_save_dir, f'Subject{subject :02d}_{quality}_0.npy'))
        label += [0]*data_false.shape[0]
        data = np.concatenate((data_true, data_false), axis=0)
        label = np.array(label)
        np.save(os.path.join(args.data_save_dir, f'Subject{subject}_data.npy'), data)
        np.save(os.path.join(args.data_save_dir, f'Subject{subject}_label.npy'), label)
        
    
    
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



args = PreprocessArgs()
generate_subject_data(args)
generate_group_data(args)
store_channel(args)
# merge_group_data(args)