import json
import os
from collections import defaultdict
from fractions import Fraction
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal


def bids_read_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", encoding="utf-8-sig")


def read_participants(data_root: str) -> pd.DataFrame:
    participants = bids_read_table(os.path.join(data_root, "participants.tsv"))
    for column in participants.select_dtypes(include="object").columns:
        participants[column] = participants[column].str.strip()
    return participants.fillna("n/a")


def discover_subjects(data_root: str) -> List[str]:
    participants_path = os.path.join(data_root, "participants.tsv")
    if os.path.exists(participants_path):
        return read_participants(data_root)["participant_id"].tolist()
    return sorted(d for d in os.listdir(data_root) if d.startswith("sub-"))


def subject_ieeg_dir(data_root: str, subject: str) -> str:
    return os.path.join(data_root, subject, "ses-presurgery", "ieeg")


def list_ieeg_edfs(data_root: str, subject: str) -> List[str]:
    ieeg_dir = subject_ieeg_dir(data_root, subject)
    if not os.path.isdir(ieeg_dir):
        return []
    return sorted(
        os.path.join(ieeg_dir, f)
        for f in os.listdir(ieeg_dir)
        if f.endswith("_ieeg.edf")
    )


def sidecar_path(edf_path: str, suffix: str) -> str:
    return edf_path.replace("_ieeg.edf", suffix)


def load_channel_table(edf_path: str) -> pd.DataFrame:
    return bids_read_table(sidecar_path(edf_path, "_channels.tsv")).fillna("n/a")


def is_usable_channel(name: str, row: pd.Series, args) -> bool:
    if str(row["type"]).upper() not in set(args.channel_types):
        return False
    if str(row["status"]).lower() != args.pick_channel_status.lower():
        return False
    upper_name = name.upper()
    return not any(keyword in upper_name for keyword in args.exclude_name_keywords)


def good_channels_for_file(edf_path: str, args) -> List[str]:
    channels = load_channel_table(edf_path)
    return [
        row["name"]
        for _, row in channels.iterrows()
        if is_usable_channel(row["name"], row, args)
    ]


def stable_subject_channels(edf_paths: Sequence[str], args) -> List[str]:
    per_file = [good_channels_for_file(path, args) for path in edf_paths]
    if not per_file:
        return []
    keep = set(per_file[0])
    for channels in per_file[1:]:
        keep &= set(channels)
    return [name for name in per_file[0] if name in keep]


def channel_metadata(edf_paths: Sequence[str], channels: Sequence[str]) -> Dict[str, dict]:
    descriptions = defaultdict(set)
    types = {}
    for edf_path in edf_paths:
        table = load_channel_table(edf_path)
        for _, row in table.iterrows():
            name = row["name"]
            if name not in channels:
                continue
            types[name] = row["type"]
            for value in str(row["status_description"]).split(","):
                value = value.strip().lower()
                if value and value != "n/a":
                    descriptions[name].add(value)

    return {
        name: {
            "type": types.get(name, "n/a"),
            "status_description": sorted(descriptions[name]),
            "is_soz": "soz" in descriptions[name],
            "is_resect": "resect" in descriptions[name],
        }
        for name in channels
    }


def read_seizure_intervals(edf_path: str) -> List[Tuple[float, float]]:
    events_path = sidecar_path(edf_path, "_events.tsv")
    if not os.path.exists(events_path):
        return []
    events = bids_read_table(events_path)
    starts = events.loc[events["trial_type"] == "sz onset", "onset"].astype(float).tolist()
    ends = events.loc[events["trial_type"] == "sz offset", "onset"].astype(float).tolist()
    return [(start, end) for start, end in zip(starts, ends) if end > start]


def filter_data(data: np.ndarray, sfreq: float, args) -> np.ndarray:
    high = args.high_pass_filter
    low = args.low_pass_filter
    if low is not None:
        low = min(low, sfreq / 2 - 1.0)

    if high is not None and low is not None and low > high:
        b, a = signal.butter(N=4, Wn=[high, low], btype="bandpass", fs=sfreq)
        data = signal.filtfilt(b, a, data, axis=-1)
    elif high is not None:
        b, a = signal.butter(N=4, Wn=high, btype="highpass", fs=sfreq)
        data = signal.filtfilt(b, a, data, axis=-1)
    elif low is not None:
        b, a = signal.butter(N=4, Wn=low, btype="lowpass", fs=sfreq)
        data = signal.filtfilt(b, a, data, axis=-1)

    notch_freq = args.notch_filter
    if notch_freq is not None:
        while notch_freq < sfreq / 2:
            b, a = signal.iirnotch(notch_freq, args.quality_factor, sfreq)
            data = signal.filtfilt(b, a, data, axis=-1)
            notch_freq += args.notch_filter

    return data


def resample_data(data: np.ndarray, source_sfreq: float, target_sfreq: int) -> np.ndarray:
    if int(round(source_sfreq)) == int(target_sfreq):
        return data
    ratio = Fraction(target_sfreq / source_sfreq).limit_denominator(1000)
    return signal.resample_poly(data, ratio.numerator, ratio.denominator, axis=-1)


def segment_data(args, sfreq: int, data: np.ndarray, intervals: Iterable[Tuple[float, float]]):
    ch_num, ch_len = data.shape
    label_pts = np.zeros(ch_len, dtype=np.float32)
    for start, end in intervals:
        start_pt = max(0, int(round(start * sfreq)))
        end_pt = min(ch_len, int(round(end * sfreq)))
        label_pts[start_pt:end_pt] = 1.0

    patch_len = int(args.patch_secs * sfreq)
    seq_pts = int(args.seq_len * patch_len)
    seq_num = ch_len // seq_pts
    if seq_num <= 0:
        return None, None

    data = data[:, : seq_num * seq_pts].reshape(ch_num, seq_num, seq_pts)
    data = data.transpose(1, 0, 2)
    labels = label_pts[: seq_num * seq_pts].reshape(seq_num, seq_pts).mean(axis=-1)
    labels = (labels > args.label_thres).astype(np.int64)
    return data.astype(np.float32), labels


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def build_subject_groups(participants: pd.DataFrame, columns: Sequence[str]) -> Dict[str, Dict[str, List[str]]]:
    groups = {}
    for column in columns:
        if column not in participants.columns:
            continue
        values = defaultdict(list)
        for _, row in participants.iterrows():
            values[str(row[column])].append(row["participant_id"])
        groups[column] = {key: sorted(value) for key, value in sorted(values.items())}
    return groups


def sample_subject_data(args, subject: str):
    data = np.load(os.path.join(args.data_save_dir, subject, "data.npy"))
    label = np.load(os.path.join(args.data_save_dir, subject, "label.npy"))
    seizure_pos = np.where(label == 1)[0]
    normal_pos = np.where(label == 0)[0]
    normal_num = min(len(normal_pos), int(len(seizure_pos) * args.normal_ratio))
    rng = np.random.default_rng(0)
    normal_pos = rng.permutation(normal_pos)[:normal_num]
    pos = np.concatenate([seizure_pos, normal_pos])
    return data[pos], label[pos]
