import os
import numpy as np
import random
import pickle

from data_preprocess.utils import _split_subjects


def _generate_subject_groups(args):
    path = os.path.join(args.data_save_dir, 'subject_list.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            subject_list = pickle.load(f)
    
    else:
        group_sub_num = args.subject_num // args.group_num
        subject_list = []
        numbers = list(range(0, args.subject_num))
        random.shuffle(numbers)
        for i in range(0, args.subject_num, group_sub_num):
            group = numbers[i:i + group_sub_num]
            subject_list.append(group)
        with open(os.path.join(path), 'wb') as f:
            pickle.dump(subject_list, f)
        
    return subject_list


selected_channel_names = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8', 'EEG T3',
                            'EEG T4', 'EEG C3', 'EEG C4', 'EEG T5', 'EEG T6', 'EEG P3', 'EEG P4',
                            'EEG O1', 'EEG O2', 'EEG Fz', 'EEG Cz', 'EEG Pz']

def _select_channels_from_edf(raw):
    all_channel_names = raw.info['ch_names']
    raw.pick_channels(selected_channel_names)
    return raw

def store_channel(args):
    import json
    channel_file = os.path.join(args.data_save_dir, 'group_data', 'channels_lst.json')
    if not os.path.exists(channel_file):
        channels = [name.split(' ')[-1] for name in selected_channel_names]
        with open(channel_file, 'w') as f:
            json.dump(channels, f)


def _merge_data(args, group):
    datas, labels = [], []
    for sub in group:
        data = np.load(os.path.join(args.data_save_dir, f'Subject{sub}_data.npy'))
        label = np.load(os.path.join(args.data_save_dir, f'Subject{sub}_label.npy'))
        datas.append(data)
        labels.append(label)
        
    datas = np.concatenate(datas, axis=0)
    labels = np.concatenate(labels)
    return datas, labels
