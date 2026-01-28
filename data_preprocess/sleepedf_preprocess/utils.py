import os
import pickle
import random

import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
from scipy import signal
from scipy.signal.windows import gaussian, hann, hamming
from scipy.signal import ShortTimeFFT
from fractions import Fraction

selected_channel_names = ['EEG Fpz-Cz']

def _select_channels_from_edf(raw):
    all_channel_names = raw.info['ch_names']
    raw.pick_channels(selected_channel_names)
    return raw


def store_channel(args):
    import json
    channel_file = os.path.join(args.data_save_dir, 'group_data', 'channels_lst.json')
    ch_names = [name.split(' ')[1] for name in selected_channel_names]
    if not os.path.exists(channel_file):
        with open(channel_file, 'w') as f:
            json.dump(ch_names, f)
    

def _count_subject(args):
    files = os.listdir(args.data_root)
    subjects_lst = []
    inform_disc = {}
    for file in files:
        if '-PSG.edf' not in file:
            if '-Hypnogram' in file:
                inform_disc[file[:6]] = file[7]
            continue
        subjects_lst.append(file[:6])
    return subjects_lst, inform_disc



def _band_pass_filter(data, fs, low_cut, high_cut):
    b, a = signal.butter(N=4, Wn=[low_cut, high_cut], btype='bandpass', fs=fs)
    data = signal.filtfilt(b, a, data)

    return data


def _notch_filter(data, fs, freq, q):
    b, a = signal.iirnotch(freq, q, fs)
    data = signal.lfilter(b, a, data)

    return data


def _segment_data(args, data):
    sfreq = int(args.sfreq)
    data = _band_pass_filter(data, sfreq, args.high_pass_filter, args.low_pass_filter)
    # data = _notch_filter(data, sfreq, args.notch_filter, args.quality_factor)
    ch_num, _ = data.shape

    patch_len = int(sfreq * args.patch_secs)
    seq_len = patch_len*args.seq_len
    seq_num = data.shape[-1] // seq_len
    data = data[:, : seq_num*seq_len].reshape(-1, ch_num, patch_len)

    return data.astype(np.float32)


def _generate_subject_groups(args):
    path = os.path.join(args.data_save_dir, 'subject_list.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            subject_list = pickle.load(f)
    else:
        subject_list = []
        group_sub_num = args.subject_num // args.group_num
        remainder = args.subject_num % args.group_num
        numbers, _ = _count_subject(args)
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


def _merge_data(args, subjects):
    group_data, group_label = [], []
    for sub in subjects:
        data = np.load(os.path.join(args.data_save_dir, f'{sub}_data.npy'))
        label = np.load(os.path.join(args.data_save_dir, f'{sub}_label.npy'))
        group_data.append(data)
        group_label.append(label)

    group_data = np.concatenate(group_data, axis=0)
    group_label = np.concatenate(group_label)

    return group_data, group_label


