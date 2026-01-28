import os
import pickle
import sys
sys.path.append('/bench-mark')
from mne.io import read_raw_edf
import numpy as np
import json
import warnings

from data_preprocess.mphce_preprocess.config import PreprocessArgs
from data_preprocess.mphce_preprocess.utils import _generate_subject_groups, _merge_data
from data_preprocess.utils import _segment_data

warnings.filterwarnings("ignore", category=RuntimeWarning)


def generate_subject_data(args):
    channel_file_mdd = os.path.join(args.data_save_dir, 'group_data_mdd', 'channels_lst.json')
    channel_file_state = os.path.join(args.data_save_dir, 'group_data_state', 'channels_lst.json')
    
    for root, dir, files in os.walk(args.data_root):
        for file in files:
            if '.edf' in file:
                file_name = file.split('.')[0]
                label, subject, state = file_name.split(' ')
                raw = read_raw_edf(os.path.join(root, file), verbose=False, preload=False)
                
                if not os.path.exists(channel_file_mdd):
                    channels = raw.info['ch_names'][: -3]
                    channels = [name.split(' ')[-1].split('-')[0] for name in channels]
                    with open(channel_file_mdd, 'w') as f:
                        json.dump(channels, f)
                    with open(channel_file_state, 'w') as f:
                        json.dump(channels, f)
                        
                data = raw.get_data()[:19]
                sfreq = int(raw.info['sfreq'])
                data = _segment_data(args, sfreq, data)
                save_dir = os.path.join(args.data_save_dir, f'{subject}_{label}')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.save(os.path.join(save_dir, f'{state}_data.npy'), data)
                print(f'data of subject {subject} saved')

        break


def generate_group_data(args, task):
    subject_groups = _generate_subject_groups(args, task)
    for i, g in enumerate(subject_groups):
        data, label = _merge_data(args, g, task)
        save_dir = os.path.join(args.data_save_dir, f'group_data_{task}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, f'group_{i}_data.npy'), data)
        np.save(os.path.join(save_dir, f'group_{i}_label.npy'), label)

        print(f'data of group {i}, task {task} saved')


def merge_group_data(args, task):
    data, label = [], []
    for g_id in range(args.group_num):
        data.append(np.load(os.path.join(args.data_save_dir,  f'group_data_{task}/group_{g_id}_data.npy')))
        label.append(np.load(os.path.join(args.data_save_dir, f'group_data_{task}/group_{g_id}_label.npy')))
    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)
    np.save(os.path.join(args.data_save_dir, f'group_data_{task}/all_data.npy'), data)
    np.save(os.path.join(args.data_save_dir, f'group_data_{task}/all_label.npy'), label)


def count_group_data(args, task):
    path = os.path.join(args.data_save_dir, f'subject_groups_{task}.pkl')
    groups = pickle.load(open(path, 'rb'))
    category = ['control', task]
    for g_id in range(args.group_num):
        subject_num = len(groups[g_id])
        print(f'group {g_id}: {subject_num} subjects')
        label = np.load(os.path.join(args.data_save_dir, f'group_data_{task}/group_{g_id}_label.npy'))
        for i, c in enumerate(category):
            print(f'class {c}: {np.sum(label==i)} samples')



args = PreprocessArgs()
generate_subject_data(args)
generate_group_data(args, 'state')
generate_group_data(args, 'mdd')
merge_group_data(args, 'mdd')
# count_group_data(args, 'mdd')