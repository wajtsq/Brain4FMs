import os
import sys
import pandas as pd
from scipy.io import loadmat
import warnings

sys.path.append('/bench-mark')
from mne.io import read_raw_edf
import numpy as np
from scipy import signal

from data_preprocess.mf_preprocess.config import PreprocessArgs, mayo_groups, fnusa_groups, mayo_pat_ids, fnusa_pat_ids
from data_preprocess.mf_preprocess.utils import _segment_data, _merge_data


warnings.filterwarnings("ignore", category=RuntimeWarning)


def generate_data(args, dataset_name, pat_ids):
    data_path = os.path.join(args.data_root, dataset_name)
    meta_data = pd.read_csv(os.path.join(data_path, 'segments.csv'))
    for sub in pat_ids:
        sub_meta_data = meta_data[meta_data.patient_id == sub]
        sub_label = np.array(sub_meta_data.category_id.array)
        sub_seg_ids = sub_meta_data.segment_id.array
        sub_data = []
        for i, seg_id in enumerate(sub_seg_ids):
            data = loadmat(os.path.join(data_path, f'DATASET_{dataset_name}', f'{seg_id}.mat'))['data']
            # down sample to 1000Hz
            data = data[:, ::5]
            data = _segment_data(args, args.sfreq, data)
            
            std = np.std(data, axis=-1, keepdims=True)
            if np.min(std) <= 0:
                data = None
            if data is None:
                sub_label = np.delete(sub_label, i)
            else:
                sub_data.append(data)
        sub_data = np.concatenate(sub_data, axis=0)

        save_dir = os.path.join(args.data_save_dir, dataset_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data = sub_data.reshape(sub_data.shape[0], 1, -1)
        np.save(os.path.join(save_dir, f'sub{sub}_data.npy'), data)
        np.save(os.path.join(save_dir, f'sub{sub}_label.npy'), sub_label)
        print(f'data and label of dataset {dataset_name} sub{sub} saved')


def group_data(args, dataset_name, groups):
    for i, g in enumerate(groups):
        data, label = _merge_data(args, dataset_name, g)
        valid_pos = np.where(label > 0)[0]

        data = data[valid_pos]
        label = label[valid_pos] - 1
        print(f'{dataset_name} group {i}: {np.sum(label==0), np.sum(label==1), np.sum(label==2)}')
        label[label==2] = 0
        np.save(os.path.join(args.data_save_dir, dataset_name, f'group_data/group_{i}_data.npy'), data)
        np.save(os.path.join(args.data_save_dir, dataset_name, f'group_data/group_{i}_label.npy'), label)
        print(f'data and label of dataset {dataset_name} group {i} saved')



def merge_group_data(args, dataset_name, group_num):
    data, label = [], []
    for g_id in range(group_num):
        data.append(np.load(os.path.join( args.data_save_dir, dataset_name, f'group_data/group_{g_id}_data.npy')))
        label.append(np.load(os.path.join(args.data_save_dir, dataset_name, f'group_data/group_{g_id}_label.npy')))
    data = np.concatenate(data, axis=1)
    label = np.concatenate(label)
    np.save(os.path.join(args.data_save_dir, dataset_name, f'group_data/all_data.npy'), data)
    np.save(os.path.join(args.data_save_dir, dataset_name, f'group_data/all_label.npy'), label)


def count_group_data(args, dataset_name, groups):
    category = ['normal', 'seizure']
    for g_id in range(len(groups)):
        subject_num = len(groups[g_id])
        print(f'group {g_id}: {subject_num} subjects')
        label = np.load(os.path.join(args.data_save_dir, dataset_name, f'group_data/group_{g_id}_label.npy'))
        for i, c in enumerate(category):
            print(f'class {c}: {np.sum(label==i)} samples')


def raw_data(args):
    spec = []
    for dataset_name in ['MAYO', 'FNUSA']:
        for i in range(args.group_num):
            spec.append(np.load(os.path.join(args.data_save_dir, dataset_name, f'group_data/group_{i}_spec.npy')))

    spec = np.concatenate(spec, axis=1)
    np.save('/home/nas/share/TUEG/preprocessed_data/mixture/mf.npy', spec)


args = PreprocessArgs()
generate_data(args, 'MAYO', mayo_pat_ids)
generate_data(args, 'FNUSA', fnusa_pat_ids)
group_data(args, 'MAYO', mayo_groups)
group_data(args, 'FNUSA', fnusa_groups)
# merge_group_data(args, 'MAYO', 6)
# merge_group_data(args, 'FNUSA', 6)
# count_group_data(args, 'MAYO', mayo_groups)
# count_group_data(args, 'FNUSA', fnusa_groups)

# raw_data(args)