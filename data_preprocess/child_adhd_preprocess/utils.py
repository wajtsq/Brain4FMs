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


def _vis_data_spec(data, secs, sfreq, t, f, spec):
    plt.figure(figsize=(10, 6))
    x_axis = np.linspace(0, secs, int(sfreq * secs))
    plt.plot(x_axis, data)
    plt.title('Time Domain Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, spec, shading='auto')
    plt.colorbar(label='Power Spectral Density (dB/Hz)')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()


def _vis_spec(t, f, spec):
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, spec, shading='auto')
    plt.colorbar(label='Power Spectral Density (dB/Hz)')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()


def _high_pass_filter(data, fs, cutoff):
    b, a = signal.butter(N=4, Wn=cutoff, btype='highpass', fs=fs)
    data = signal.lfilter(b, a, data)

    return data


def _band_pass_filter(data, fs, low_cut, high_cut):
    b, a = signal.butter(N=4, Wn=[low_cut, high_cut], btype='bandpass', fs=fs)
    data = signal.filtfilt(b, a, data)

    return data


def _notch_filter(data, fs, freq, q):
    b, a = signal.iirnotch(freq, q, fs)
    data = signal.filtfilt(b, a, data)

    return data


def _cal_spec(args, sfreq, data):
    window_size = int(sfreq * args.spec_window_secs)
    win = gaussian(M=window_size, std=window_size // 6, sym=False)
    hop = int(sfreq * args.hop_secs)
    SFT = ShortTimeFFT(win=win, hop=hop, fs=sfreq, mfft=window_size, scale_to=args.scale_to, fft_mode=args.fft_mode)

    _, patch_len = data.shape
    sx = SFT.spectrogram(data)
    f = SFT.f
    t = SFT.t(patch_len)

    return f, t, sx


def _generate_groups(args):
    path = os.path.join(args.data_save_dir, f'subject_groups.pkl')
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    else:
        adhd_list, hc_list = [], []
        for root, dir, files in os.walk(args.data_save_dir):
            for file in files:
                if '_data.npy' in file:
                    sub_group, sub_name = file.split('_')[0], file.split('_')[1]
                    if sub_group == 'ADHD':
                        adhd_list.append(f'{sub_group}_{sub_name}')
                    elif sub_group == 'Control':
                        hc_list.append(f'{sub_group}_{sub_name}')
            break

        adhd_group = _split_subjects(adhd_list, args.group_num)
        hc_group   = _split_subjects(hc_list, args.group_num)
        groups = []
        for i in range(args.group_num):
            groups.append(adhd_group[i] + hc_group[i])

        pickle.dump(groups, open(path, 'wb'))

    return groups


def _merge_data(args, group):
    data, label = [], []
    for sub in group:
        sub_data = np.load(os.path.join(args.data_save_dir, f'{sub}_data.npy'))
        if 'Control' in sub:
            sub_label = np.zeros(sub_data.shape[0], dtype=int)
        elif 'ADHD' in sub:
            sub_label = np.ones(sub_data.shape[0], dtype=int)
        else:
            raise ValueError

        # sub_data = _std_data_segment(sub_data)

        data.append(sub_data)
        label.append(sub_label)

    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)

    return data, label





