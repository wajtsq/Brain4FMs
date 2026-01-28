import os
import numpy as np
import random
import pickle


def _generate_subject_groups(args):
    path = os.path.join(args.data_save_dir, 'subject_list.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            subject_list = pickle.load(f)
    
    else:
        group_sub_num = args.subject_num // args.group_num
        remainder = args.subject_num % args.group_num
        subject_list = []
        numbers = list(range(1, args.subject_num + 1))
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


def _merge_data(args, group):
    datas, labels = [], []
    for sub in group:
        data = np.load(os.path.join(args.data_save_dir, f's{sub:02d}_data.npy'))
        label = np.load(os.path.join(args.data_save_dir, f's{sub:02d}_label.npy'))
        datas.append(data)
        labels.append(label)
        
    datas = np.concatenate(datas, axis=0)
    labels = np.concatenate(labels)
    return datas, labels


def _get_labels(args, labels, num):
    real_labels = []
    for label in labels:
        is_like = (label[0] + label[3])/2
        is_calm = (label[1] + label[2])/2
        if is_like < args.threshold and is_calm < args.threshold:
            real_labels += [0]*num
        elif is_like < args.threshold and is_calm > args.threshold:
            real_labels += [1]*num
        elif is_like > args.threshold and is_calm < args.threshold:
            real_labels += [2]*num 
        else:
            real_labels += [3]*num 
    return np.array(real_labels)


def store_channels(args):
    import json
    channel_file = os.path.join(args.data_save_dir, 'group_data', 'channels_lst.json')
    channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', \
                'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', \
                'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
    if not os.path.exists(channel_file):
        with open(channel_file, 'w') as f:
            json.dump(channels, f)