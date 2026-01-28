import os
import numpy as np
import pickle
import random
from scipy.signal import butter, lfilter

from data_preprocess.utils import _split_subjects


def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


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
        data = np.load(os.path.join(args.data_save_dir, f'{sub}_data.npy'))
        label = np.load(os.path.join(args.data_save_dir, f'{sub}_label.npy'))
        datas.append(data)
        labels.append(label)
        
    datas = np.concatenate(datas, axis=0)
    labels = np.concatenate(labels)
    return datas, labels


def _segment_data(args, data):
    ch_num, ch_len = data.shape
    data = _butter_bandpass_filter(data, args.l_freq, args.h_freq, args.sfreq, order=5)
    patch_len = int(args.patch_secs * args.sfreq)
    seq_pts = patch_len * args.seq_len
    seq_num = ch_len // seq_pts
    data = data[:, :seq_num * seq_pts].reshape(ch_num, seq_num, -1)
    data = data.transpose(1, 0, 2)

    return data.astype(np.float32)