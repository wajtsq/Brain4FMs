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
from data_preprocess.utils import _split_subjects, _std_spec, _std_data_segment


def _generate_groups(args, sz_list, hc_list):
    path = os.path.join(args.data_save_dir, 'subject_groups.pkl')
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    else:
        sz_group = _split_subjects(sz_list, args.group_num)
        hc_group = _split_subjects(hc_list, args.group_num)
        groups = []
        for i in range(args.group_num):
            groups.append(sz_group[i]+hc_group[i])

        pickle.dump(groups, open(path, 'wb'))

    return groups


def _merge_data(args, group):
    data, label = [], []
    for sub in group:
        sub_data = np.load(os.path.join(args.data_save_dir, f'{sub}_data.npy'))

        if 's' in sub:
            sub_label = np.ones(sub_data.shape[0], dtype=int)
        elif 'h' in sub:
            sub_label = np.zeros(sub_data.shape[0], dtype=int)
        else:
            raise ValueError

        # sub_data = _std_data_segment(sub_data)

        data.append(sub_data)
        label.append(sub_label)

    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)

    return data, label