import os
import random

import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
import pandas as pd
from scipy import signal
from scipy.signal.windows import gaussian, hann, hamming
from scipy.signal import ShortTimeFFT
from fractions import Fraction
import pickle
from data_preprocess.utils import _split_subjects, _std, _std_spec, _std_data_segment, _std_data


def _merge_data(args, group):
    data, label = [], []
    for sub in group:
        sub_data, sub_label = [], []
        for ses in [1, 2]:
            if os.path.exists(os.path.join(args.data_save_dir, sub, f'ses_{ses}_data.npy')):
                ses_data = np.load(os.path.join(args.data_save_dir, sub, f'ses_{ses}_data.npy'))
                ses_label = np.ones(ses_data.shape[0], dtype=int) * (ses - 1)
                sub_data.append(ses_data)
                sub_label.append(ses_label)
        if len(sub_data) > 0:
            sub_data = np.concatenate(sub_data, axis=0)
            sub_label = np.concatenate(sub_label)

            # sub_data = _std_data_segment(sub_data)

            data.append(sub_data)
            label.append(sub_label)
        else:
            print(f'{sub} invalid')

    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)

    return data, label





