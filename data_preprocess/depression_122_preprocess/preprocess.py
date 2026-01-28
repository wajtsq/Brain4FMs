import os
import pickle
import sys
sys.path.append('/bench-mark')
from mne.io import read_raw_edf
import numpy as np
import pandas as pd
from mne.io import read_raw_eeglab
import warnings
import json

from data_preprocess.depression_122_preprocess.config import PreprocessArgs
from data_preprocess.depression_122_preprocess.utils import _generate_bdi_groups, _generate_stai_groups, _merge_data
from data_preprocess.utils import _segment_data

warnings.filterwarnings("ignore", category=RuntimeWarning)

subject_list = [f'sub-{str(i).zfill(3)}' for i in range(1, 123)]


def generate_subject_score(args):
    bdi_dict, stai_dict = {}, {}
    df = pd.read_csv(os.path.join(args.data_root, 'participants.tsv'), delimiter='\t')
    for i in range(args.subject_num):
        row = df.iloc[i]
        pid = row.participant_id
        bdi = row.BDI
        stai = row.STAI

        bdi_dict[pid] = bdi
        stai_dict[pid] = stai

    pickle.dump(bdi_dict, open(os.path.join(args.data_save_dir, 'BDI.pkl'), 'wb'))
    pickle.dump(stai_dict, open(os.path.join(args.data_save_dir, 'STAI.pkl'), 'wb'))
    print('BDI and STAI scores saved')


def generate_subject_data(args):
    channel_file_BDI = os.path.join(args.data_save_dir, 'group_data_BDI', 'channels_lst.json')
    channel_file_STAI = os.path.join(args.data_save_dir, 'group_data_STAI', 'channels_lst.json')
    for run in [1, 2]:
        for sub in subject_list:
            data_path = os.path.join(args.data_root, f'{sub}/eeg')
            file_name = f'{sub}_task-Rest_run-0{run}_eeg.set'
            if not os.path.exists(os.path.join(data_path, file_name)): 
                continue
            raw = read_raw_eeglab(os.path.join(data_path, file_name), verbose=False, preload=True)
            
            if os.path.exists(os.path.join(data_path, file_name)):
                if not os.path.exists(channel_file_BDI):
                    channels = raw.info['ch_names'][: -2]
                    with open(channel_file_BDI, 'w') as f:
                        json.dump(channels, f)
                    with open(channel_file_STAI, 'w') as f:
                        json.dump(channels, f)
                        
                data = raw.get_data()[:, ::2]
                data = data[:64]    # remove ECOG signal
                sfreq = int(raw.info['sfreq']) / 2
                data = _segment_data(args, sfreq, data)

                np.save(os.path.join(args.data_save_dir, f'{sub}_data_run{run}.npy'), data)
                print(f'{file_name} processed')


def generate_group_data(args, group_std):
    if group_std == 'BDI':
        score_dict  = pickle.load(open(os.path.join(args.data_root, 'BDI.pkl'), 'rb'))
        groups = _generate_bdi_groups(args, score_dict)
    elif group_std == 'STAI':
        score_dict = pickle.load(open(os.path.join(args.data_root, 'STAI.pkl'),'rb'))
        groups = _generate_stai_groups(args, score_dict)
    else:
        raise ValueError
    for i, g in enumerate(groups):
        data, label = _merge_data(args, g, score_dict, group_std)
        assert data.shape[0] == label.shape[0]
        np.save(os.path.join(args.data_save_dir, f'group_data_{group_std}/group_{i}_data.npy'), data)
        np.save(os.path.join(args.data_save_dir, f'group_data_{group_std}/group_{i}_label.npy'), label)
        print(f'standard {group_std}: group {i} saved')


def merge_group_data(args, group_std):
    data, label = [], []
    for g_id in range(args.group_num):
        data.append(np.load(os.path.join(args.data_save_dir,  f'group_data_{group_std}/group_{g_id}_data.npy')))
        label.append(np.load(os.path.join(args.data_save_dir, f'group_data_{group_std}/group_{g_id}_label.npy')))
    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)
    np.save(os.path.join(args.data_save_dir, f'group_data_{group_std}/all_data.npy'), data)
    np.save(os.path.join(args.data_save_dir, f'group_data_{group_std}/all_label.npy'), label)


def count_group_data(args, group_std):
    path = os.path.join(args.data_save_dir, 'bdi_groups.pkl')
    groups = pickle.load(open(path, 'rb'))
    category = ['control', 'depression']
    for g_id in range(args.group_num):
        subject_num = len(groups[g_id])
        print(f'group {g_id}: {subject_num} subjects')
        label = np.load(os.path.join(args.data_save_dir, f'group_data_{group_std}/group_{g_id}_label.npy'))
        for i, c in enumerate(category):
            print(f'class {c}: {np.sum(label==i)} samples')


args = PreprocessArgs()
# get BDI and STAI score
# generate_subject_score(args)
generate_subject_data(args)
generate_group_data(args, 'BDI')
generate_group_data(args, 'STAI')
# merge_group_data(args, 'BDI')
# count_group_data(args, 'BDI')

