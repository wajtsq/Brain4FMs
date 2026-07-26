import copy
import os
import sys
import warnings
import json

sys.path.append('/bench-mark')

import numpy as np

from data_preprocess.HUP_preprocess.config import PreprocessArgs
from data_preprocess.HUP_preprocess.utils import (
    filter_data,
    list_ieeg_edfs,
    read_participants,
    read_seizure_intervals,
    resample_data,
    segment_data,
    stable_subject_channels,
)


warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    import mne

    mne.set_log_level("ERROR")
except ModuleNotFoundError:
    mne = None


SUBDATASETS = (
    ("HUP-SEEG", "SEEG"),
    ("HUP-ECoG", "ECOG"),
)


def _modality_args(args, implant):
    modality_args = copy.copy(args)
    modality_args.channel_types = (implant.upper(),)
    return modality_args


def _load_run(edf_path, channels, args):
    if mne is None:
        raise ImportError("mne is required to read EDF files in this preprocessing environment")
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    raw.pick_channels(list(channels), ordered=True)
    raw.load_data(verbose=False)
    data = raw.get_data()
    if data.shape[0] != len(channels):
        raise ValueError(f"Expected {len(channels)} channels, but got {data.shape[0]} after picking")
    sfreq = float(raw.info["sfreq"])
    if args.common_average_reference:
        data = data - data.mean(axis=0, keepdims=True)

    data = filter_data(data, sfreq, args)
    if args.sfreq is not None and int(round(sfreq)) != args.sfreq:
        data = resample_data(data, sfreq, args.sfreq)
        sfreq = args.sfreq

    return data, int(round(sfreq))


def _dataset_save_dir(args, dataset_name):
    return os.path.join(os.path.dirname(args.data_save_dir), dataset_name)


def _implant_subjects(args, implant):
    participants = read_participants(args.data_root)
    return participants[participants["implant"] == implant]["participant_id"].tolist()


def _subject_channel_map(args, subjects):
    subject_channels = {}
    skipped = {}
    for subject in subjects:
        edf_paths = list_ieeg_edfs(args.data_root, subject)
        if not edf_paths:
            skipped[subject] = "no_edf"
            continue
        channels = stable_subject_channels(edf_paths, args)
        if not channels:
            skipped[subject] = "no_stable_good_ieeg_channels"
            continue
        subject_channels[subject] = channels
    return subject_channels, skipped


def _make_patient_groups(subjects):
    return [[subject] for subject in sorted(subjects)]


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _patient_subjects_and_channels(args, implant):
    args = _modality_args(args, implant)
    subjects = _implant_subjects(args, implant)
    subject_channels, _ = _subject_channel_map(args, subjects)
    subjects = sorted(subject_channels)
    return subjects, subject_channels


def group_data(args, dataset_name, groups):
    save_dir = _dataset_save_dir(args, dataset_name)
    group_dir = os.path.join(save_dir, "group_data")
    os.makedirs(group_dir, exist_ok=True)
    for group_id, subjects in enumerate(groups):
        data, label = [], []
        for subject in subjects:
            sub_dir = os.path.join(save_dir, subject)
            if not os.path.exists(os.path.join(sub_dir, "data.npy")):
                continue
            sub_data = np.load(os.path.join(sub_dir, "data.npy"))
            sub_label = np.load(os.path.join(sub_dir, "label.npy"))

            seizure_pos = np.where(sub_label == 1)[0]
            normal_pos = np.where(sub_label == 0)[0]
            if len(seizure_pos) == 0:
                print(f"{subject}: no seizure windows in saved data, skipped")
                continue
            rng = np.random.default_rng(0)
            normal_num = min(len(normal_pos), int(len(seizure_pos) * args.normal_ratio))
            normal_pos = rng.permutation(normal_pos)[:normal_num]
            pos = np.concatenate([seizure_pos, normal_pos])
            if len(pos) == 0:
                continue
            data.append(sub_data[pos])
            label.append(sub_label[pos])

        if not data:
            print(f"group {group_id}: no subject data, skipped")
            continue
        data = np.concatenate(data, axis=0)
        label = np.concatenate(label, axis=0)
        np.save(os.path.join(group_dir, f"group_{group_id}_data.npy"), data)
        np.save(os.path.join(group_dir, f"group_{group_id}_label.npy"), label)
        if len(subjects) == 1:
            channel_json = os.path.join(save_dir, subjects[0], "channel_lst.json")
            if os.path.exists(channel_json):
                with open(channel_json) as f:
                    channels = json.load(f)
                _write_json(os.path.join(group_dir, f"group_{group_id}_channel_lst.json"), channels)
        print(f"group {group_id}: saved {data.shape}, seizure={int(label.sum())}")


def generate_data(args, dataset_name, implant):
    args = _modality_args(args, implant)
    save_root = _dataset_save_dir(args, dataset_name)
    os.makedirs(save_root, exist_ok=True)
    subjects, subject_channels_map = _patient_subjects_and_channels(args, implant)
    processed_subjects = []

    for subject in subjects:
        edf_paths = list_ieeg_edfs(args.data_root, subject)
        if not edf_paths:
            print(f"{subject}: no EDF files, skipped")
            continue

        subject_channels = subject_channels_map[subject]
        if not subject_channels:
            print(f"{subject}: no stable good iEEG channels, skipped")
            continue

        if not any(read_seizure_intervals(path) for path in edf_paths):
            print(f"{subject}: no seizure intervals, skipped before EDF loading", flush=True)
            continue

        sub_data, sub_label = [], []
        for edf_path in edf_paths:
            try:
                size_gb = os.path.getsize(edf_path) / 1024 ** 3
                print(
                    f"{dataset_name} {subject}: loading {os.path.basename(edf_path)} "
                    f"({size_gb:.2f} GB, {len(subject_channels)} channels)",
                    flush=True,
                )
                data, sfreq = _load_run(edf_path, subject_channels, args)
                print(
                    f"{dataset_name} {subject}: loaded {os.path.basename(edf_path)} "
                    f"shape={data.shape}, sfreq={sfreq}",
                    flush=True,
                )
                intervals = read_seizure_intervals(edf_path)
                data, label = segment_data(args, sfreq, data, intervals)
                if data is None:
                    continue
                sub_data.append(data)
                sub_label.append(label)
                print(
                    f"{dataset_name} {subject}: segmented {os.path.basename(edf_path)} "
                    f"windows={label.shape[0]}, seizure={int(label.sum())}",
                    flush=True,
                )
            except Exception as exc:
                print(f"{subject}: failed {os.path.basename(edf_path)} ({exc})", flush=True)

        if not sub_data:
            print(f"{subject}: no valid windows, skipped")
            continue

        sub_data = np.concatenate(sub_data, axis=0)
        sub_label = np.concatenate(sub_label, axis=0)
        if int(sub_label.sum()) == 0:
            print(f"{subject}: no seizure windows, skipped")
            continue

        save_dir = os.path.join(save_root, subject)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "data.npy"), sub_data)
        np.save(os.path.join(save_dir, "label.npy"), sub_label)
        _write_json(os.path.join(save_dir, "channel_lst.json"), list(subject_channels))
        processed_subjects.append(subject)
        print(f"{subject}: saved {sub_data.shape}, seizure windows={int(sub_label.sum())}")

    return _make_patient_groups(processed_subjects)


if __name__ == "__main__":
    args = PreprocessArgs()
    for dataset_name, implant in SUBDATASETS:
        subject_groups = generate_data(args, dataset_name, implant)
        group_data(args, dataset_name, subject_groups)
