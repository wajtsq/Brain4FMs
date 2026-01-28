import os
import re
import sys
sys.path.append('/bench-mark')
import numpy as np
import mne
import warnings

from data_preprocess.UCSD_preprocess.config import PreprocessArgs
from data_preprocess.UCSD_preprocess.utils import _generate_subject_groups, _merge_data, _select_channels_from_edf, store_channels
from data_preprocess.utils import _segment_data

warnings.filterwarnings("ignore", category=RuntimeWarning)


def generate_data(args, is_pd, task):
    for root, dirs, _ in os.walk(args.data_root):
        for dir in dirs:
            if ('sub-pd' not in dir and is_pd) or (not is_pd and 'sub-pd' in dir) or 'sub' not in dir:
                continue
            bdf_path = os.path.join(args.data_root, dir, f'ses-{task}', 'eeg', f'{dir}_ses-{task}_task-rest_eeg.bdf')
            raw = mne.io.read_raw_bdf(bdf_path)
            raw = _select_channels_from_edf(raw)    # special channels
            data = raw.get_data()[:, ::2]   # downsample
            data =  _segment_data(args, args.sfreq, data)

            
            np.save(os.path.join(args.data_save_dir, f'{dir}_{task}_data.npy'), data)

            print(f'data and label of subject {dir} saved')
            

def generate_group_data(args, task):
    subject_groups = _generate_subject_groups(args, task)
            
    for i, g in enumerate(subject_groups):
        data, label = _merge_data(args, g, task)
        save_dir = os.path.join(args.data_save_dir, f'group_data_{task}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, f'group_{i}_data.npy'), data)
        np.save(os.path.join(save_dir, f'group_{i}_label.npy'), label)

        print(f'data of group {i} is saved')




args = PreprocessArgs()
store_channels(args)
generate_data(args, False, 'hc')
generate_data(args, True, 'on')
generate_data(args, True, 'off')
generate_group_data(args, 'on')
generate_group_data(args, 'off')
