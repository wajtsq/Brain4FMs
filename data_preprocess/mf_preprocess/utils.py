import os
import random

import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
from scipy import signal
from scipy.signal.windows import gaussian, hann, hamming
from scipy.signal import ShortTimeFFT
from fractions import Fraction


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


def _high_pass_filter(data, fs, cutoff):
    b, a = signal.butter(N=4, Wn=cutoff, btype='highpass', fs=fs)
    data = signal.lfilter(b, a, data)

    return data


def _low_pass_filter(data, fs, cutoff):
    b, a = signal.butter(N=4, Wn=cutoff, btype='lowpass', fs=fs)
    data = signal.filtfilt(b, a, data)

    return data

def _band_pass_filter(data, fs, low_cut, high_cut):
    b, a = signal.butter(N=4, Wn=[low_cut, high_cut], btype='bandpass', fs=fs)
    data = signal.filtfilt(b, a, data)

    return data


def _notch_filter(data, fs, freq, q):
    b, a = signal.iirnotch(freq, q, fs)
    data = signal.lfilter(b, a, data)

    return data


def _cal_spec(args, sfreq, data):
    window_size = int(sfreq * args.spec_window_secs)
    win = gaussian(M=window_size, std=window_size // 6, sym=False)
    hop = int(sfreq * args.hop_secs)
    SFT = ShortTimeFFT(win=win, hop=hop, fs=sfreq, mfft=window_size, scale_to=args.scale_to, fft_mode=args.fft_mode)
    seq_num, context_len, patch_len = data.shape
    data = data.reshape(-1, patch_len)

    sx = SFT.spectrogram(data)
    f = SFT.f
    t = SFT.t(patch_len)
    # t = SFT.t(3000)

    _, f_size, t_size = sx.shape
    sx = sx.reshape(seq_num, context_len, f_size, t_size)
    return f, t, sx


def _segment_data(args, sfreq, data):
    # filtering
    data = _low_pass_filter(data, sfreq, sfreq / 3)
    # data = _notch_filter(data, sfreq, args.notch_filter, args.quality_factor)
    # segment
    seq_num, _ = data.shape
    patch_len = int(sfreq * args.patch_secs)
    data = data.reshape(seq_num, args.seq_len, patch_len)

    return data.astype(np.float32)


def _std(data):
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std

    return data


def _merge_data(args, dataset_name, subjects):
    data = []
    label = []
    for s in subjects:
        sub_data = np.load(os.path.join(args.data_save_dir, dataset_name, f'sub{s}_data.npy'))
        sub_label = np.load(os.path.join(args.data_save_dir, dataset_name, f'sub{s}_label.npy'))

        # ori_shape = sub_spec.shape
        # sub_spec = sub_spec.reshape(-1)
        # sub_spec = _std(sub_spec)
        # sub_spec = sub_spec.reshape(ori_shape)

        data.append(sub_data)
        label.append(sub_label)

    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)
    # data = np.expand_dims(data, axis=1)

    return data, label
