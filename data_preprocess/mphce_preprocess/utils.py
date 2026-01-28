import os
import random

import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
from scipy import signal
from scipy.signal.windows import gaussian, hann, hamming
from scipy.signal import ShortTimeFFT
from fractions import Fraction
import pickle
from data_preprocess.utils import _split_subjects, _std, _std_spec, _std_data_segment


def _get_subject_list(args):
    subject_list = []
    for root, dir, files in os.walk(args.data_save_dir):
        for d in dir:
            if '_H' in d or '_MDD' in d:
                subject_list.append(d)
        break

    return subject_list


def _generate_subject_groups(args, task):
    path = os.path.join(args.data_save_dir, f'subject_groups_{task}.pkl')
    if os.path.exists(path):
        group = pickle.load(open(path, 'rb'))
    else:
        subject_list = _get_subject_list(args)
        mdd_list, hc_list = [], []
        for sub in subject_list:
            if '_H' in sub:
                hc_list.append(sub)
            elif '_MDD' in sub:
                mdd_list.append(sub)

        hc_group = _split_subjects(hc_list, args.group_num)
        mdd_group = _split_subjects(mdd_list, args.group_num)

        group = []
        for i in range(args.group_num):
            group.append(hc_group[i]+mdd_group[i])
        pickle.dump(group, open(path, 'wb'))

    return group


def _load_sub_data(args, sub):
    data_len = [0, 0, 0]
    sub_data = []
    for i, state in enumerate(['EC', 'EO', 'TASK']):
        data_path = os.path.join(args.data_save_dir, sub, f'{state}_data.npy')

        if os.path.exists(data_path):
            data = np.load(data_path)
            data_len[i] = data.shape[0]
            sub_data.append(data)

    sub_data = np.concatenate(sub_data, axis=0)
    # sub_data = _std_data_segment(sub_data)

    return sub_data, data_len


def _merge_data(args, group, task):
    data, label = [], []
    for sub in group:
        sub_data, data_len = _load_sub_data(args, sub)
        data.append(sub_data)

        if task == 'state':
            if data_len[0] > 0:
                label.append(np.zeros(data_len[0], dtype=int))
            if data_len[1] > 0:
                label.append(np.ones(data_len[1], dtype=int))
            if data_len[2] > 0:
                label.append(2*np.ones(data_len[2], dtype=int))

        elif task == 'mdd':
            if '_H' in sub:
                l = 0
            elif '_MDD' in sub:
                l = 1
            else:
                raise ValueError
            label.append(l * np.ones(sum(data_len), dtype=int))
        else:
            raise ValueError

    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)

    return data, label





