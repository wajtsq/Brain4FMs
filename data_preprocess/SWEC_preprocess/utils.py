import json
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal


def subject_ids(subject_num: int) -> List[str]:
    return [f"ID{i:02d}" for i in range(1, subject_num + 1)]


def subject_dir(data_root: str, subject: str) -> str:
    return os.path.join(data_root, subject)


def total_file(data_root: str, subject: str) -> str:
    return os.path.join(subject_dir(data_root, subject), f"{subject}_total.h5")


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def make_subject_groups(subjects: Sequence[str], group_num: int) -> List[List[str]]:
    groups = [[] for _ in range(group_num)]
    for idx, subject in enumerate(subjects):
        groups[idx % group_num].append(subject)
    return groups


def read_seizures(h5_file) -> List[Tuple[float, float]]:
    if "data/seizures" not in h5_file:
        return []
    seizures = h5_file["data/seizures"][:]
    intervals = []
    for row in seizures:
        start = float(row["onsets"])
        end = float(row["offsets"])
        if end > start:
            intervals.append((start, end))
    return intervals


def interval_overlap_fraction(start: int, end: int, intervals: Iterable[Tuple[int, int]]) -> float:
    overlap = 0
    for int_start, int_end in intervals:
        overlap += max(0, min(end, int_end) - max(start, int_start))
    return overlap / max(1, end - start)


def resample_window(data: np.ndarray, source_sfreq: int, target_sfreq: int) -> np.ndarray:
    if int(source_sfreq) == int(target_sfreq):
        return data
    gcd = np.gcd(int(source_sfreq), int(target_sfreq))
    up = int(target_sfreq // gcd)
    down = int(source_sfreq // gcd)
    return signal.resample_poly(data, up, down, axis=-1)


def select_window_starts(
    total_len: int,
    sfreq: int,
    window_len: int,
    seizure_intervals_sec: Sequence[Tuple[float, float]],
    label_thres: float,
    normal_ratio: Optional[float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    seizure_intervals = [
        (max(0, int(round(start * sfreq))), min(total_len, int(round(end * sfreq))))
        for start, end in seizure_intervals_sec
    ]
    seizure_starts = set()
    for start, end in seizure_intervals:
        first = max(0, (start // window_len) * window_len)
        last = min(total_len - window_len, end)
        for win_start in range(first, last + 1, window_len):
            win_end = win_start + window_len
            if interval_overlap_fraction(win_start, win_end, seizure_intervals) > label_thres:
                seizure_starts.add(win_start)

    all_starts = np.arange(0, total_len - window_len + 1, window_len, dtype=np.int64)
    if len(all_starts) == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    seizure_starts = np.asarray(sorted(seizure_starts), dtype=np.int64)
    if normal_ratio is None:
        seizure_set = set(seizure_starts.tolist())
        labels = np.asarray([1 if start in seizure_set else 0 for start in all_starts], dtype=np.int64)
        return all_starts, labels

    seizure_set = set(seizure_starts.tolist())
    normal_starts = np.asarray([start for start in all_starts if start not in seizure_set], dtype=np.int64)
    normal_num = min(len(normal_starts), int(len(seizure_starts) * normal_ratio))
    if normal_num > 0:
        normal_starts = rng.permutation(normal_starts)[:normal_num]
    else:
        normal_starts = np.asarray([], dtype=np.int64)

    starts = np.concatenate([seizure_starts, normal_starts])
    labels = np.concatenate(
        [
            np.ones(len(seizure_starts), dtype=np.int64),
            np.zeros(len(normal_starts), dtype=np.int64),
        ]
    )
    order = rng.permutation(len(starts))
    return starts[order], labels[order]
