import os
import pickle
import sys
sys.path.append('/bench-mark')
from scipy.io import loadmat
import numpy as np
import json

from data_preprocess.sleepdep_preprocess.config import PreprocessArgs
from data_preprocess.sleepdep_preprocess.utils import _merge_data
from data_preprocess.utils import _segment_data, _std_spec, _split_subjects

from mne.io import read_raw_eeglab
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


channels = ['Fp1', 'AF3', 'AF7', 'Fz', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3',
            'FC5', 'FT7', 'Cz', 'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3', 'CP5',
            'TP7', 'TP9', 'Pz', 'P1', 'P3', 'P5', 'P7', 'PO3', 'PO7', 'Oz',
            'O1', 'Fpz', 'Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2',
            'FC4', 'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CPz', 'CP2', 'CP4',
            'CP6', 'TP8', 'TP10', 'P2', 'P4', 'P6', 'P8', 'POz', 'PO4', 'PO8', 'O2']


def generate_subject_data(args):
    subject_list = [f'sub-{str(i).zfill(2)}' for i in range(1, args.subject_num+1)]
    channel_file = os.path.join(args.data_save_dir, 'group_data', 'channels_lst.json')
    for sub in subject_list[:]:
        for ses in [1 ,2]:
            data_dir = os.path.join(args.data_root, sub, f'ses-{ses}/eeg')
            ses_data = []
            for root, dir, files in os.walk(data_dir):
                for file in files:
                    if '.set' in file:
                        try:
                            raw = read_raw_eeglab(os.path.join(root, file), verbose=False, preload=True)
                        except RuntimeError:
                            print(f'error on {file}')
                        else:
                            ch_names = raw.ch_names
                            if not os.path.exists(channel_file):
                                with open(channel_file, 'w') as f:
                                    json.dump(ch_names, f)
                                    
                            valid_idx = []
                            for ch in channels:
                                for idx, ch_name in enumerate(ch_names):
                                    if ch_name == ch:
                                        valid_idx.append(idx)
                                        break

                            if len(valid_idx) == len(channels):
                                data = raw.get_data()[valid_idx][:, ::2]
                                data = _segment_data(args, args.sfreq, data)
                                ses_data.append(data)
                            else:
                                print(f'{file}: invalid channels')

            if len(ses_data) > 0:
                ses_data = np.concatenate(ses_data, axis=0)
                save_dir = os.path.join(args.data_save_dir, sub)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.save(os.path.join(save_dir, f'ses_{ses}_data.npy'), ses_data)
                print(f'data and spec of {sub}, ses {ses} saved')
            else:
                print(f'empty for {sub}, ses {ses}')


def generate_group_data(args):
    path = os.path.join(args.data_save_dir, f'subject_groups.pkl')
    subject_list = [f'sub-{str(i).zfill(2)}' for i in range(1, args.subject_num + 1)]
    if os.path.exists(path):
        groups = pickle.load(open(path, 'rb'))
    else:
        groups = _split_subjects(subject_list, args.group_num)
        pickle.dump(groups, open(path, 'wb'))
    for i, g in enumerate(groups):
        data, label = _merge_data(args, g)

        np.save(os.path.join(args.data_save_dir, f'group_data/group_{i}_data.npy'), data)
        np.save(os.path.join(args.data_save_dir, f'group_data/group_{i}_label.npy'), label)
        print(f'data of group {i} saved')


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
    category = ['normal', 'deprivation']
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

