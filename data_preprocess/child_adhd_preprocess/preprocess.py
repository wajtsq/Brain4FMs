import os
import pickle
import sys
sys.path.append('/bench-mark')
from scipy.io import loadmat
import numpy as np
from scipy import signal

from data_preprocess.child_adhd_preprocess.config import PreprocessArgs
from data_preprocess.child_adhd_preprocess.utils import _merge_data, _generate_groups
from data_preprocess.utils import _std_spec, _segment_data, _std_multi_dim

from mne.io import read_raw_eeglab
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def generate_subject_data(args):
    for group in ['ADHD_part1', 'ADHD_part2', 'Control_part1', 'Control_part2']:
        sub_group = group.split('_')[0]
        for root, dir, files in os.walk(os.path.join(args.data_root, group)):
            for file in files:
                if '.mat' in file:
                    sub_name = file.split('.')[0]
                    data = loadmat(os.path.join(root, file))[sub_name]
                    data = np.swapaxes(data, axis1=0, axis2=1)
                    # data = _std_multi_dim(data)
                    data = _segment_data(args, args.sfreq, data)
                    np.save(os.path.join(args.data_save_dir, f'{sub_group}_{sub_name}_data.npy'), data)
                    print(f'data of subject {sub_name} saved')


def generate_group_data(args):
    groups = _generate_groups(args)
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
    data = np.concatenate(data, axis=1)
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

