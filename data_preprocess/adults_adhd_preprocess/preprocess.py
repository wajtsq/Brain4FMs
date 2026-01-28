import os
import pickle
import sys
sys.path.append('/bench-mark')
from scipy.io import loadmat
import numpy as np
from scipy import signal
import pandas as pd
import json
import matplotlib.pyplot as plt

from data_preprocess.adults_adhd_preprocess.config import PreprocessArgs
from data_preprocess.adults_adhd_preprocess.utils import _merge_data, _generate_groups, _std_data
from data_preprocess.utils import _segment_data, _std_spec
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

CELL_CHANNELS = [
    ("Cz", "F4"),  # cell 1
    ("Cz", "F4"),  # cell 2
    ("Cz", "F4"),  # cell 3
    ("Cz", "F4"),  # cell 4
    ("Cz", "F4"),  # cell 5
    ("Cz", "F4"),  # cell 6
    ("O1", "F4"),  # cell 7
    ("O1", "F4"),  # cell 8
    ("O1", "F4"),  # cell 9
    ("F3", "F4"),  # cell 10
    ("Fz", "F4"),  # cell 11
]
channel_name = ["Cz", "F4", "O1", "F3", "Fz"]
CH2IDX = {ch: i for i, ch in enumerate(channel_name)}

# def generate_subject_data(args):
#     for group in ['FADHD', 'FC', 'MADHD', 'MC']:
#         mat_data = loadmat(os.path.join(args.data_root, f'{group}.mat'))[group][0]
#         group_data = np.swapaxes(np.concatenate(list(mat_data), axis=1), axis1=1, axis2=2)
#         sub_num = group_data.shape[0]
#         for i in range(sub_num):
#             data = group_data[i]
#             if group == 'FADHD' and i == 6:
#                 continue
#             # data = _std_data(data)    # std
#             data = _segment_data(args, args.sfreq, data)
#             np.save(os.path.join(args.data_save_dir, f'{group}_{i}_data.npy'), data)
#             print(f'data of subject {group}_{i} saved')

def generate_subject_data(args):
    for group in ['FADHD', 'FC', 'MADHD', 'MC']:
        mat_data = loadmat(os.path.join(args.data_root, f'{group}.mat'))[group][0]
        # mat_data[k].shape = (sub_num, 2, T_k)
        sub_num = mat_data[0].shape[0]
        cell_num = mat_data.shape[0]
        for i in range(sub_num):
            if group == 'FADHD' and i == 6:
                continue

            all_seg_data = []
            all_seg_pos  = []

            for k in range(cell_num):
                data_cell = mat_data[k][i]          # (2, T_k)
                data_cell = np.swapaxes(data_cell, axis1=0, axis2=1)
                ch1, ch2  = CELL_CHANNELS[k]

                pos_pair = np.array([
                    CH2IDX[ch1],
                    CH2IDX[ch2]
                ], dtype=np.int64)

                seg_data = _segment_data(args, args.sfreq, data_cell)   # (N,2,L)
                seg_pos = np.tile(pos_pair[None,:], (seg_data.shape[0],1))

                all_seg_data.append(seg_data)
                all_seg_pos.append(seg_pos)

            subject_data = np.concatenate(all_seg_data, axis=0)  # (total_N,2,L)
            subject_pos  = np.concatenate(all_seg_pos,  axis=0)  # (total_N,2)

            np.save(os.path.join(args.data_save_dir,f'{group}_{i}_data.npy'), subject_data)
            np.save(os.path.join(args.data_save_dir,f'{group}_{i}_pos.npy'),  subject_pos)

            print(f'data/pos of subject {group}_{i} saved')


def generate_group_data(args):
    groups = _generate_groups(args)
    for i, g in enumerate(groups):
        data, pos, label = _merge_data(args, g)

        np.save(os.path.join(args.data_save_dir, f'group_data/group_{i}_data.npy'), data)
        np.save(os.path.join(args.data_save_dir, f'group_data/group_{i}_pos.npy'), pos)
        np.save(os.path.join(args.data_save_dir, f'group_data/group_{i}_label.npy'), label)
        print(f'data of group {i} saved')
    channel_file = os.path.join(args.data_save_dir, 'group_data', 'channels_lst.json')
    if not os.path.exists(channel_file):
        with open(channel_file, 'w') as f:
            json.dump(channel_name, f)


def merge_group_data(args):
    data, label = [], []
    for g_id in range(args.group_num):
        data.append(np.load(os.path.join(args.data_save_dir, f'group_data/group_{g_id}_data.npy')))
        label.append(np.load(os.path.join(args.data_save_dir, f'group_data/group_{g_id}_label.npy')))
    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)
    np.save(os.path.join(args.data_save_dir, f'group_data/all_data.npy'), data)
    np.save(os.path.join(args.data_save_dir, f'group_data/all_label.npy'), label)


def count_group_data(args):
    path = os.path.join(args.data_save_dir, f'subject_groups.pkl')
    groups = pickle.load(open(path, 'rb'))
    category = ['control', 'adhd']
    for g_id in range(args.group_num):
        subject_num = len(groups[g_id])
        print(f'group {g_id}: {subject_num} subjects')
        label = np.load(os.path.join(args.data_save_dir, f'group_data/group_{g_id}_label.npy'))
        for i, c in enumerate(category):
            print(f'class {c}: {np.sum(label==i)} samples')



args = PreprocessArgs()
generate_subject_data(args)
generate_group_data(args)
# merge_group_data(args)
# count_group_data(args)