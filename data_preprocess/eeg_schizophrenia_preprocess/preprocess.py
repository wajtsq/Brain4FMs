import os
import pickle
import sys
sys.path.append('/bench-mark')
from mne.io import read_raw_edf
import numpy as np
import json

from data_preprocess.eeg_schizophrenia_preprocess.config import PreprocessArgs
from data_preprocess.eeg_schizophrenia_preprocess.utils import _generate_groups, _merge_data
from data_preprocess.utils import _segment_data

from mne.io import read_raw_edf
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

sz_list = [f's{str(i).zfill(2)}' for i in range(1, 15)]
hc_list = [f'h{str(i).zfill(2)}' for i in range(1, 15)]

channels = ['Fp2', 'F8', 'T4', 'T6', 'O2', 'Fp1', 'F7', 'T3', 'T5', 'O1', 'F4', 'C4', 'P4', 'F3', 'C3', 'P3', 'Fz', 'Cz', 'Pz']


def generate_subject_data(args):
    channel_file = os.path.join(args.data_save_dir, 'group_data', 'channels_lst.json')
    for root, dir, files in os.walk(args.data_root):
        for file in files:
            if '.edf' in file:
                subject_name = file.split('.')[0]
                edf = read_raw_edf(os.path.join(root, file), verbose=False, preload=True)
                
                channels = edf.info['ch_names']
                if not os.path.exists(channel_file):
                    with open(channel_file, 'w') as f:
                        json.dump(channels, f)
                        
                data = edf.get_data()
                ch_idx = []
                for name in channels:
                    for idx, ch_name in enumerate(edf.ch_names):
                        if name == ch_name:
                            ch_idx.append(idx)
                            break
                data = data[ch_idx][:, :-1000]
                # plot_data(data[:, :100], 'a')

                data = _segment_data(args, args.sfreq, data)

                np.save(os.path.join(args.data_save_dir, f'{subject_name}_data.npy'), data)
                print(f'file {file} processed')
        break


def generate_group_data(args):
    subject_groups = _generate_groups(args, sz_list, hc_list)
    for i, g in enumerate(subject_groups):
        data, label = _merge_data(args, g)

        np.save(os.path.join(args.data_save_dir, f'group_data/group_{i}_data.npy'), data)
        np.save(os.path.join(args.data_save_dir, f'group_data/group_{i}_label.npy'), label)

        print(f'group {i} processed')


def merge_group_data(args):
    data, label = [], []
    for g_id in range(args.group_num):
        data.append(np.load(os.path.join(args.data_save_dir,  f'group_data/group_{g_id}_data.npy')))
        label.append(np.load(os.path.join(args.data_save_dir, f'group_data/group_{g_id}_label.npy')))
    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)
    np.save(os.path.join(args.data_save_dir, f'group_data/all_data.npy'), data)
    np.save(os.path.join(args.data_save_dir, f'group_data/all_label.npy'), label)


def count_group_data(args):
    path = os.path.join(args.data_save_dir, f'subject_groups.pkl')
    groups = pickle.load(open(path, 'rb'))
    category = ['control', 'schizophrenia']
    for g_id in range(args.group_num):
        subject_num = len(groups[g_id])
        print(f'group {g_id}: {subject_num} subjects')
        label = np.load(os.path.join(args.data_save_dir, f'group_data/group_{g_id}_label.npy'))
        for i, c in enumerate(category):
            print(f'class {c}: {np.sum(label==i)} samples')


def raw_data(args):
    subject_list = sz_list + hc_list
    data = []
    for sub in subject_list:
        sub_data = np.load(os.path.join(args.data_save_dir, f'{sub}_data.npy'))
        data.append(sub_data)

    data = np.concatenate(data, axis=1)
    np.save(os.path.join(args.data_save_dir, 'data.npy'), data)


args = PreprocessArgs()
generate_subject_data(args)
generate_group_data(args)
# merge_group_data(args)
# count_group_data(args)

