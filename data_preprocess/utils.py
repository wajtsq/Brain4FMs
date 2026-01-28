import os
import numpy as np
import random
from scipy.signal.windows import gaussian
from scipy.signal import ShortTimeFFT
from scipy import signal


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

def _segment_data(args, sfreq, data):
    # filtering
    data = _band_pass_filter(data, sfreq, args.high_pass_filter, sfreq / 3)
    data = _notch_filter(data, sfreq, args.notch_filter, args.quality_factor)
    # segment
    ch_num, ch_len = data.shape
    # truncate
    patch_len = int(args.patch_secs * sfreq)
    seq_pts = patch_len * args.seq_len
    seq_num = ch_len // seq_pts
    data = data[:, :seq_num * seq_pts].reshape(ch_num, seq_num, -1)
    data = data.transpose(1, 0, 2)
    
    return data.astype(np.float32)


def _sample_data(args, subject, normal_ratio):
    data = np.load(os.path.join(args.data_save_dir, subject, 'data.npy'))
    label = np.load(os.path.join(args.data_save_dir, subject, 'label.npy'))

    # data = _std_data_segment(data)     # normalization
    seizure_pos = np.where(label == 1)[0]
    normal_pos = np.where(label == 0)[0]

    normal_sample_num = int(len(seizure_pos)*normal_ratio)
    np.random.shuffle(normal_pos)
    sample_normal_pos = normal_pos[:normal_sample_num]

    sample_data = np.concatenate([data[seizure_pos], data[sample_normal_pos]], axis=0)
    sample_label = np.concatenate([np.ones(len(seizure_pos), dtype=int), np.zeros(len(sample_normal_pos), dtype=int)])

    return sample_data, sample_label


def _merge_data(args, subjects):
    data, label = [], []
    for s in subjects:
        sample_data, sample_label = _sample_data(args, s, args.normal_ratio)
        data.append(sample_data)
        label.append(sample_label)

    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)

    return data, label


def _cal_spec(sfreq, data, spec_window_secs):
    window_size = int(sfreq * spec_window_secs)
    win = gaussian(M=window_size, std=window_size // 6, sym=False)
    hop_secs = spec_window_secs / 2
    hop = int(sfreq * hop_secs)
    SFT = ShortTimeFFT(win=win, hop=hop, fs=sfreq, mfft=window_size, 
                       scale_to='psd', fft_mode='onesided')

    _, patch_len = data.shape
    sx = SFT.spectrogram(data)
    f = SFT.f
    t = SFT.t(patch_len)

    return f, t, sx



def _split_subjects(subjects, group_num):
    random.shuffle(subjects)
    groups = [[] for _ in range(group_num)]
    for i, s in enumerate(subjects):
        groups[i%group_num].append(s)

    return groups


def _std(data):
    mean = np.mean(data)
    std = np.std(data)
    assert std > 0
    data = (data - mean) / std

    return data


def _std_spec(spec):
    ori_shape = spec.shape
    spec = spec.reshape(-1)
    spec_mean = np.mean(spec)
    spec_std = np.std(spec)
    spec = _std(spec)
    spec = spec.reshape(ori_shape)

    return spec, spec_mean, spec_std


def _std_multi_dim(data):       # normalization
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    assert np.min(std) > 0
    data = (data - mean) / std

    return data


def _std_data(data):
    ori_shape = data.shape
    data = data.reshape(ori_shape[0], -1)
    data = _std_multi_dim(data)
    data = data.reshape(ori_shape)

    return data


def _std_data_segment(data):
    ori_shape = data.shape
    data = data.reshape(ori_shape[0]*ori_shape[1], -1)    # (bsz*ch_num, N)
    data = _std_multi_dim(data)
    data = data.reshape(ori_shape)

    return data
