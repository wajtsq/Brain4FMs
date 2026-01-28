import os
import pickle
import sys
sys.path.append('/bench-mark')
from mne.io import read_raw_edf
import numpy as np
import warnings
import re

from data_preprocess.sleepedf_preprocess.config import PreprocessArgs
from data_preprocess.sleepedf_preprocess.utils import _segment_data, _generate_subject_groups, _merge_data,\
                                                      _select_channels_from_edf, _count_subject, store_channel

warnings.filterwarnings("ignore", category=RuntimeWarning)

phase_dict = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 
              'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4,}


def extract_info(s):
    sleep_pattern = re.compile(r'Sleep stage (W|1|2|3|4|R)')
    sleep_matches = list(re.finditer(sleep_pattern, s))
    time_lst = []
    sleep_list = []

    last_index = 0
    for match in sleep_matches:
        sub_str = s[last_index:match.start()]
        time_match = re.search(r'\+(\d+)', sub_str)
        if time_match:
            time_lst.append(int(time_match.group(1)))
            sleep_list.append(phase_dict["Sleep stage " + match.group(1)])
        last_index = match.end()
    
    assert len(time_lst) == len(sleep_list)
    return time_lst, sleep_list


def generate_data(args):
    subjects, inform_disc = _count_subject(args)
    for sub in subjects:
        raw = read_raw_edf(os.path.join(args.data_root, f'{sub}J0-PSG.edf'), preload=True)
        with open(os.path.join(args.data_root, f'{sub}J{inform_disc[sub]}-Hypnogram.edf')) as f:
            inform_txt = f.readlines()[0]
            time_lst, sleep_lst = extract_info(inform_txt)
        
        raw = _select_channels_from_edf(raw)
        data = raw.get_data()
        data = _segment_data(args, data)
        time_lst.append(data.shape[0])
        
        labels, datas = [], []
        start = 0
        for i, sleep_id in enumerate(sleep_lst):
            duration = (time_lst[i+1] - time_lst[i]) // args.seq_len * args.seq_len
            if duration <= 0:
                start = time_lst[i+1]
                continue
            
            sub_data = data[start :start+duration].reshape(-1, args.ch_num, args.seq_len*args.sfreq)
            labels.append([sleep_id]*sub_data.shape[0])
            datas.append(sub_data)
            start = time_lst[i+1]
            
        datas = np.concatenate(datas, axis=0)
        labels = np.concatenate(labels)
        np.save(os.path.join(args.data_save_dir, f'{sub}_data.npy'), datas)
        np.save(os.path.join(args.data_save_dir, f'{sub}_label.npy'), labels)

        print(f'data and label of subject {sub} saved')


def group_data(args):
    groups = _generate_subject_groups(args)

    for id, group in enumerate(groups):
        data, label = _merge_data(args, group)
        np.save(os.path.join(args.data_save_dir, f'group_data/group_{id}_data.npy'), data)
        np.save(os.path.join(args.data_save_dir, f'group_data/group_{id}_label.npy'), label)

        print(f'data and label of group {id} saved')




args = PreprocessArgs()
generate_data(args)
group_data(args)
store_channel(args)
