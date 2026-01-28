import os
import re
import sys
sys.path.append('/bench-mark')
import numpy as np
import mne
import warnings

from data_preprocess.ISRUC_preprocess.config import PreprocessArgs
from data_preprocess.ISRUC_preprocess.utils import _generate_subject_groups, _merge_data, _select_channels_from_edf, store_channels
from data_preprocess.utils import _segment_data

warnings.filterwarnings("ignore", category=RuntimeWarning)


def generate_data(args):
    for sub in range(args.subject_num):
        edf_path = os.path.join(args.data_root, str(sub+1), f'{sub+1}.edf')
        raw = mne.io.read_raw_edf(edf_path)
        raw = _select_channels_from_edf(raw)    # special channels
        if raw == None:
            continue
        data = raw.get_data()
        data =  _segment_data(args, args.sfreq, data)
        
        with open(os.path.join(args.data_root, str(sub+1), f'{sub+1}_1.txt')) as f:
            label = [''.join(re.findall(r'\d+', i)) for i in f.readlines()]
        label = np.array([int(i) - 1 if int(i) == 5 else int(i) for i in label[: data.shape[0]]])
        
        np.save(os.path.join(args.data_save_dir, f'{sub+1}_data.npy'), data)
        np.save(os.path.join(args.data_save_dir, f'{sub+1}_label.npy'), label)

        print(f'data and label of subject {sub} saved')


def generate_group_data(args):
    subject_groups = _generate_subject_groups(args)
            
    for i, g in enumerate(subject_groups):
        data, label = _merge_data(args, g)
        save_dir = os.path.join(args.data_save_dir, f'group_data')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, f'group_{i}_data.npy'), data)
        np.save(os.path.join(save_dir, f'group_{i}_label.npy'), label)

        print(f'data of group {i} is saved')




args = PreprocessArgs()
store_channels(args)
generate_data(args)
generate_group_data(args)

