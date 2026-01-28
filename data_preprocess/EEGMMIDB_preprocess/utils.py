import os
import numpy as np
import random
import pickle

from data_preprocess.utils import _band_pass_filter, _notch_filter, _std_multi_dim


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


def _segment_data(args, data):
    # filtering
    data = _band_pass_filter(data, args.sfreq, args.high_pass_filter, args.low_pass_filter)
    data = _notch_filter(data, args.sfreq, args.notch_filter, args.quality_factor)
    # normalization
    # data = _std_multi_dim(data)
    
    return data.astype(np.float32)



def _merge_data(args, group, task):
    datas, labels = [], []
    for sub in group:
        data = np.load(os.path.join(args.data_save_dir, f'S{sub :03d}_{task}_data.npy'))
        label = np.load(os.path.join(args.data_save_dir, f'S{sub :03d}_{task}_label.npy'))
        datas.append(data)
        labels.append(label)
        
    datas = np.concatenate(datas, axis=0)
    labels = np.concatenate(labels)
    return datas, labels
