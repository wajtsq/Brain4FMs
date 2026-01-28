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
from data_preprocess.utils import _split_subjects, _std, _std_spec, _std_data_segment


def _generate_groups(args, task):
    path = os.path.join(args.data_save_dir, f'subject_groups_{task}.pkl')
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    else:
        c_sub, a_sub, f_sub = [], [], []
        subject_info = pd.read_csv(os.path.join(args.data_root, 'participants.tsv'), delimiter='\t')
        for row_id in range(len(subject_info)):
            row = subject_info.iloc[row_id]
            sub = row['participant_id']
            group = row['Group']
            if group == 'C':
                c_sub.append(f'{sub}_{group}')
            elif group == 'A':
                a_sub.append(f'{sub}_{group}')
            elif group == 'F':
                f_sub.append(f'{sub}_{group}')

        c_group = _split_subjects(c_sub, args.group_num)
        a_group = _split_subjects(a_sub, args.group_num)
        f_group = _split_subjects(f_sub, args.group_num)

        groups = []
        for i in range(args.group_num):
            if task == 'AD':
                groups.append(c_group[i] + a_group[i])
            elif task == 'FTD':
                groups.append(c_group[i] + f_group[i])
        pickle.dump(groups, open(path, 'wb'))

    return groups


def _merge_data(args, group):
    data, label = [], []
    for sub in group:
        sub_data = np.load(os.path.join(args.data_save_dir, f'{sub}_data.npy'))
        l = sub.split('_')[-1]
        if l == 'C':
            sub_label = np.zeros(sub_data.shape[0], dtype=int)
        elif l == 'A' or l == 'F':
            sub_label = np.ones(sub_data.shape[0], dtype=int)
        else:
            raise ValueError
        
        # sub_data = _std_data_segment(sub_data)   # normalization

        data.append(sub_data)
        label.append(sub_label)

    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)

    return data, label

