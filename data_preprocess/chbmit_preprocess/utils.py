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
import seaborn as sns
from data_preprocess.utils import _std_spec, _std_data, _std_data_segment


def find_edf_files(file_dir):
    edf_files = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.endswith(".edf"):
                file_path = os.path.join(root, file)
                edf_files.append(file_path)

    return edf_files


def _vis_data_spec(data, secs, sfreq, t, f, spec, name):
    plt.figure(figsize=(10, 6))
    x_axis = np.linspace(0, secs, data.shape[-1])
    plt.plot(x_axis, data)
    plt.title('Time Domain Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(False)
    # plt.show()
    plt.savefig(f'./img/{name}_signal.pdf')

    # plt.figure(figsize=(10, 6))
    # # sns.heatmap(spec, cmap='jet', aspect='auto', xticklabels=10, yticklabels=10)
    # plt.pcolormesh(t, f, spec, shading='auto')
    # plt.colorbar(label='Power Spectral Density (dB/Hz)')
    # plt.title('Spectrogram')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # # plt.show()
    # plt.savefig(f'./img/{name}_spec.pdf')
    # ...


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
    data = signal.lfilter(b, a, data)

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


def _segment_data(args, sfreq, data, start, end, seizure_num):
    # filtering
    data = _band_pass_filter(data, sfreq, args.high_pass_filter, sfreq / 3)
    data = _notch_filter(data, sfreq, args.notch_filter, args.quality_factor)
    # segment
    ch_num, ch_len = data.shape
    label = np.zeros(ch_len, dtype=int)
    for i in range(seizure_num):
        label[sfreq * start[i]:sfreq * end[i]] = 1
    # truncate
    patch_len = int(args.patch_secs * sfreq)
    seq_pts = args.seq_len * patch_len
    seq_num = ch_len // seq_pts
    assert seq_num > 0
    data = data[:, :seq_num * seq_pts].reshape(-1, patch_len)
    label = label[:seq_num * seq_pts].reshape(seq_num, seq_pts)
    label = np.mean(label, axis=-1)
    label = np.where(label > args.label_thres, 1, 0)
    data = data.reshape(ch_num, seq_num, -1)
    data = data.transpose(1, 0, 2)

    return data.astype(np.float32), label


def _cal_raw_spec(args, sfreq, data):
    window_size = int(sfreq * args.spec_window_secs)
    win = gaussian(M=window_size, std=window_size // 6, sym=False)
    hop = int(sfreq * args.hop_secs)
    SFT = ShortTimeFFT(win=win, hop=hop, fs=sfreq, mfft=window_size, scale_to=args.scale_to, fft_mode=args.fft_mode)
    ch_num, patch_num, patch_len = data.shape
    data = data.reshape(-1, patch_len)

    sx = SFT.spectrogram(data)
    f = SFT.f
    t = SFT.t(patch_len)

    _, f_size, t_size = sx.shape
    sx = sx.reshape(ch_num, patch_num, f_size, t_size)
    # sx = np.real(sx)
    return f, t, sx



def analysis_summary_txt(summary_file):
    records = {}

    with open(summary_file, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        line = lines[i].strip()  # Remove trailing newlines and whitespace

        if line.startswith('Number of Seizures in File: '):
            x = int(line.split(': ')[-1])  # Extract the numeric value x

            if x > 0 and i - 3 >= 0 and i + 2 < len(lines):
                start, end = [], []
                for j in range(1, x+1):
                    s_line = lines[i + 2*j-1].strip().split(':')[-1].strip()
                    e_line = lines[i + 2*j].strip().split(':')[-1].strip()

                    s = int(s_line.split(' ')[0])
                    e = int(e_line.split(' ')[0])
                    start.append(s)
                    end.append(e)

                file_name = lines[i - 3].strip()[11:]
                if '.edf' not in file_name:
                    file_name = lines[i - 1].strip()[11:]
                records[file_name] = {
                    'seizure_num': x,
                    'start': start,
                    'end': end
                }

    return records


def _spilt_data(eeg_data, sfreq, start, end, seizure_num):
    assert len(start) == len(end) == seizure_num
    seizure_list, normal_list = [], []
    start = [s*sfreq for s in start]
    end = [e*sfreq for e in end]
    normal_list.append(eeg_data[:, :start[0]])
    normal_list.append(eeg_data[:, end[-1]:])
    for i in range(seizure_num):
        seizure_list.append(eeg_data[:, start[i]:end[i]])
        if i < seizure_num-1:
            normal_list.append(eeg_data[:, end[i]:start[i+1]])

    return seizure_list, normal_list
