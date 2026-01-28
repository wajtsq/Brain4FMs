import os
import random

import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
import pandas as pd
from scipy import signal
from scipy.signal.windows import gaussian, hann, hamming
from scipy.signal import ShortTimeFFT
from fractions import Fraction
import pickle
from data_preprocess.utils import _split_subjects, _std, _std_spec, _std_data_segment, _std_data


def _generate_groups(args):
    path = os.path.join(args.data_save_dir, f'subject_groups.pkl')
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    else:
        adhd_list, hc_list = [], []
        for root, dir, files in os.walk(args.data_save_dir):
            for file in files:
                if '_data.npy' in file:
                    sub_name = file[:-9]
                    if 'ADHD' in file:
                        adhd_list.append(sub_name)
                    elif 'C' in file:
                        hc_list.append(sub_name)
            break

        adhd_group = _split_subjects(adhd_list, args.group_num)
        hc_group   = _split_subjects(hc_list, args.group_num)
        groups = []
        for i in range(args.group_num):
            groups.append(adhd_group[i] + hc_group[i])

        pickle.dump(groups, open(path, 'wb'))

    return groups


def _merge_data(args, group):
    data, pos, label = [], [], []
    for sub in group:
        sub_data = np.load(os.path.join(args.data_save_dir, f'{sub}_data.npy'))
        sub_pos = np.load(os.path.join(args.data_save_dir, f'{sub}_pos.npy'))
        if 'C' in sub:
            sub_label = np.zeros(sub_data.shape[0], dtype=int)
        elif 'ADHD' in sub:
            sub_label = np.ones(sub_data.shape[0], dtype=int)
        else:
            raise ValueError

        # sub_data = _std_data_segment(sub_data)

        data.append(sub_data)
        pos.append(sub_pos)
        label.append(sub_label)

    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)
    pos = np.concatenate(pos, axis=0)

    return data, pos, label





