import os
import pickle
import sys
sys.path.append('/bench-mark')
from mne.io import read_raw_edf
import numpy as np
from scipy import signal

from data_preprocess.chbmit_preprocess.config import PreprocessArgs
from data_preprocess.chbmit_preprocess.utils import analysis_summary_txt, _segment_data
from data_preprocess.utils import _split_subjects, _merge_data

from mne.io import read_raw_edf
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4',
            'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9',
            'FT9-FT10', 'FT10-T8', 'T8-P8-1']


def generate_data(args):
    subject_list = [str(i).zfill(2) for i in range(1, args.subject_num+1)]
    for subject in subject_list:
        sub_data, sub_label = [], []
        data_root = os.path.join(args.data_root, f'chb{subject}')
        summary_file = os.path.join(data_root, f'chb{subject}-summary.txt')
        annotation = analysis_summary_txt(summary_file)
        sfreq = None
        for root, dir, files in os.walk(data_root):
            for file in files:
                if file in annotation.keys():
                    raw = read_raw_edf(os.path.join(root, file), preload=True, verbose=False)
                    eeg_data = raw.get_data()
                    sfreq = int(raw.info['sfreq'])
                    file_rec = annotation[file]
                    seizure_num = file_rec['seizure_num']
                    start = file_rec['start']
                    end = file_rec['end']
                    # channel selection
                    ch_idx = []
                    for ch_name in channels:
                        for idx, name in enumerate(raw.ch_names):
                            if name == ch_name:
                                ch_idx.append(idx)
                                break

                    if len(ch_idx) == len(channels):
                        eeg_data = eeg_data[ch_idx]
                        data, label = _segment_data(args, sfreq, eeg_data, start, end, seizure_num)
                        sub_data.append(data)
                        sub_label.append(label)

            break

        sub_data = np.concatenate(sub_data, axis=0)
        sub_label = np.concatenate(sub_label)
        save_dir = os.path.join(args.data_save_dir, subject)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, 'data.npy'), sub_data)
        np.save(os.path.join(save_dir, 'label.npy'), sub_label)

        print(f'data of subject {subject} saved')


def group_data(args):
    path = os.path.join(args.data_save_dir, 'subject_groups.pkl')
    if os.path.exists(path):
        groups = pickle.load(open(path, 'rb'))
    else:
        subject_list = [str(i).zfill(2) for i in range(1, args.subject_num + 1)]
        groups = _split_subjects(subject_list, args.group_num)
        pickle.dump(groups, open(path, 'wb'))

    for i, g in enumerate(groups):
        data, label = _merge_data(args, g)
        np.save(os.path.join(args.data_save_dir, f'group_data/group_{i}_data.npy'), data)
        np.save(os.path.join(args.data_save_dir, f'group_data/group_{i}_label.npy'), label)
        print(f'data and label of group {i} saved')


def count_group_data(args):
    path = os.path.join(args.data_save_dir, f'subject_groups.pkl')
    groups = pickle.load(open(path, 'rb'))
    category = ['normal', 'seizure']
    for g_id in range(args.group_num):
        subject_num = len(groups[g_id])
        print(f'group {g_id}: {subject_num} subjects')
        label = np.load(os.path.join(args.data_save_dir, f'group_data/group_{g_id}_label.npy'))
        for i, c in enumerate(category):
            print(f'class {c}: {np.sum(label==i)} samples')


def merge_group_data(args):
    data, label = [], []
    for g_id in range(args.group_num):
        data.append(np.load(os.path.join(args.data_save_dir, f'group_data/group_{g_id}_data.npy')))
        label.append(np.load(os.path.join(args.data_save_dir, f'group_data/group_{g_id}_label.npy')))
    data = np.concatenate(data, axis=1)
    label = np.concatenate(label)
    np.save(os.path.join(args.data_save_dir, f'group_data/all_data.npy'), data)
    np.save(os.path.join(args.data_save_dir, f'group_data/all_label.npy'), label)


def store_channel(args):
    import json
    channel_file = os.path.join(args.data_save_dir, 'group_data', 'channels_lst.json')
    ch_names = [name.split('-')[0]+'-'+name.split('-')[1] for name in channels]
    if not os.path.exists(channel_file):
        with open(channel_file, 'w') as f:
            json.dump(ch_names, f)
    


args = PreprocessArgs()
generate_data(args)
group_data(args)
store_channel(args)
# merge_group_data(args)
# count_group_data(args)

