import os
import pickle
import sys
sys.path.append('/bench-mark')
import numpy as np
import pandas as pd
import json
from data_preprocess.adfd_preprocess.config import PreprocessArgs
from data_preprocess.adfd_preprocess.utils import _merge_data, _generate_groups
from data_preprocess.utils import _segment_data

from mne.io import read_raw_eeglab
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


subject_list = [f'sub-{str(i).zfill(3)}' for i in range(1, 89)]


def generate_subject_data(args):
    subject_info = pd.read_csv(os.path.join(args.data_root, 'participants.tsv'), delimiter='\t')
    channel_file = os.path.join(args.data_save_dir, 'group_data_AD', 'channels_lst.json')
    for row_id in range(len(subject_info)):
        row = subject_info.iloc[row_id]
        sub = row['participant_id']
        group = row['Group']

        data_path = os.path.join(args.data_root, f'{sub}/eeg/{sub}_task-eyesclosed_eeg.set')
        raw = read_raw_eeglab(data_path, verbose=False, preload=True)
        if not os.path.exists(channel_file):
            channels = raw.info['ch_names']
            with open(channel_file, 'w') as f:
                json.dump(channels, f)
        data = raw.get_data()[:, ::2]      # downsampled
        data = _segment_data(args, args.sfreq, data)
        if not os.path.exists(args.data_save_dir):
            os.mkdir(args.data_save_dir)
        np.save(os.path.join(args.data_save_dir, f'{sub}_{group}_data.npy'), data)

        print(f'data of subject {sub} saved')


def generate_group_data(args, task):
    groups = _generate_groups(args, task)
    for i, g in enumerate(groups):
        data, label = _merge_data(args, g)  # bsz, ch_num, N
        path = os.path.join(args.data_save_dir, f'group_data_{task}')
        if not os.path.exists(path):
            os.mkdir(path)
        np.save(os.path.join(path, f'group_{i}_data.npy'), data)
        np.save(os.path.join(path, f'group_{i}_label.npy'), label)
        print(f'data of group {i} saved')


def merge_group_data(args, task):
    data, label = [], []
    for g_id in range(args.group_num):
        data.append(np.load(os.path.join(args.data_save_dir,  f'group_data_{task}/group_{g_id}_data.npy')))
        label.append(np.load(os.path.join(args.data_save_dir, f'group_data_{task}/group_{g_id}_label.npy')))
    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)
    np.save(os.path.join(args.data_save_dir, f'group_data_{task}/all_data.npy'), data)
    np.save(os.path.join(args.data_save_dir, f'group_data_{task}/all_label.npy'), label)


def count_group_data(args, task):
    path = os.path.join(args.data_save_dir, f'subject_groups.pkl')
    groups = pickle.load(open(path, 'rb'))
    category = ['control', task]
    for g_id in range(args.group_num):
        subject_num = len(groups[g_id])
        print(f'group {g_id}: {subject_num} subjects')
        label = np.load(os.path.join(args.data_save_dir, f'group_data_{task}/group_{g_id}_label.npy'))
        for i, c in enumerate(category):
            print(f'class {c}: {np.sum(label==i)} samples')


args = PreprocessArgs()
# Preprocess the downloaded raw data to obtain data and labels of one subject.
generate_subject_data(args)
# Group the data according to patients.
generate_group_data(args, 'AD')
# Merge all data
# merge_group_data(args, 'AD')
# count_group_data(args, 'AD')

