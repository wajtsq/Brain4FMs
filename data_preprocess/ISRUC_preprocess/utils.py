import os
import numpy as np
import random
import pickle


def count_subjects(path):
    files = os.listdir(path)
    subjects = []
    for file in files:
        if 'data.npy' in file:
            subjects.append(int(file.split('_')[0]))
    return subjects
        

def _generate_subject_groups(args):
    path = os.path.join(args.data_save_dir, 'subject_list.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            subject_list = pickle.load(f)
    
    else:
        subject_list = []
        numbers = count_subjects(args.data_save_dir)
        args.subject_num = len(numbers)
        group_sub_num = args.subject_num // args.group_num
        remainder = args.subject_num % args.group_num
        random.shuffle(numbers)
        start = 0
        for i in range(args.group_num):
            group_size = group_sub_num
            if i < remainder:
                group_size += 1
            group = numbers[start: start + group_size]
            subject_list.append(group)
            start += group_size
        with open(os.path.join(path), 'wb') as f:
            pickle.dump(subject_list, f)
        
    return subject_list

selected_channel_names = ['F3-A2', 'C3-A2', 'O1-A2', 'F4-A1', 'C4-A1', 'O2-A1']

def _select_channels_from_edf(raw):
    all_channel_names = raw.info['ch_names']
    if len(set(selected_channel_names)-set(all_channel_names)) > 0:
        return None
    raw.pick_channels(selected_channel_names)
    return raw


def store_channels(args):
    import json
    channel_file = os.path.join(args.data_save_dir, 'group_data', 'channels_lst.json')
    if not os.path.exists(channel_file):
        with open(channel_file, 'w') as f:
            json.dump(selected_channel_names, f)


def _merge_data(args, group):
    datas, labels = [], []
    for sub in group:
        data = np.load(os.path.join(args.data_save_dir, f'{sub}_data.npy'))
        label = np.load(os.path.join(args.data_save_dir, f'{sub}_label.npy'))
        datas.append(data)
        labels.append(label)
        
    datas = np.concatenate(datas, axis=0)
    labels = np.concatenate(labels)
    return datas, labels
