import argparse
import json
import os
import sys
import warnings

sys.path.append("/bench-mark")

import hdf5plugin
import h5py
import numpy as np

from data_preprocess.SWEC_preprocess.config import PreprocessArgs
from data_preprocess.SWEC_preprocess.utils import (
    read_seizures,
    resample_window,
    select_window_starts,
    subject_ids,
    total_file,
    write_json,
)


warnings.filterwarnings("ignore", category=RuntimeWarning)


def _subject_save_dir(args, subject):
    return os.path.join(args.data_save_dir, subject)


def _channel_names(ch_num):
    return [f"iEEG_{idx:03d}" for idx in range(ch_num)]


def _make_patient_groups(subjects):
    return [[subject] for subject in sorted(subjects)]


def _available_subjects(args, subjects=None):
    subjects = subjects or subject_ids(args.subject_num)
    return [subject for subject in subjects if os.path.exists(total_file(args.data_root, subject))]


def _decode_h5_strings(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _total_part_files(h5_file):
    if "info/files" not in h5_file:
        return None
    return _decode_h5_strings(h5_file["info/files"][:])


def _vds_sources_match_parts(ieeg, part_files):
    if not getattr(ieeg, "is_virtual", False) or not part_files:
        return True
    source_files = [os.path.basename(source.file_name) for source in ieeg.virtual_sources()]
    return source_files == part_files


class _DatasetIEEGSource:
    def __init__(self, dataset):
        self.dataset = dataset
        self.shape = dataset.shape

    def read_window(self, start, length):
        return self.dataset[:, start : start + length]

    def close(self):
        return


class _PartIEEGSource:
    def __init__(self, args, subject, part_files):
        self.handles = []
        self.datasets = []
        self.lengths = []
        for part_file in part_files:
            part_path = os.path.join(args.data_root, subject, part_file)
            if not os.path.exists(part_path):
                raise FileNotFoundError(f"{part_path} not found")
            handle = h5py.File(part_path, "r")
            dataset = handle["data/ieeg"]
            self.handles.append(handle)
            self.datasets.append(dataset)
            self.lengths.append(dataset.shape[1])

        if not self.datasets:
            raise ValueError(f"{subject}: no part files")
        ch_nums = {dataset.shape[0] for dataset in self.datasets}
        if len(ch_nums) != 1:
            raise ValueError(f"{subject}: inconsistent channel counts across part files: {sorted(ch_nums)}")

        self.cum_lengths = np.cumsum(self.lengths)
        self.shape = (self.datasets[0].shape[0], int(self.cum_lengths[-1]))

    def read_window(self, start, length):
        end = start + length
        out = np.empty((self.shape[0], length), dtype=np.float32)
        out_pos = 0
        part_idx = int(np.searchsorted(self.cum_lengths, start, side="right"))
        while start < end and part_idx < len(self.datasets):
            part_start = 0 if part_idx == 0 else int(self.cum_lengths[part_idx - 1])
            local_start = start - part_start
            local_end = min(self.lengths[part_idx], local_start + (end - start))
            read_len = local_end - local_start
            out[:, out_pos : out_pos + read_len] = self.datasets[part_idx][:, local_start:local_end]
            out_pos += read_len
            start += read_len
            part_idx += 1
        if out_pos != length:
            raise ValueError(f"Unable to read full window, requested={length}, read={out_pos}")
        return out

    def close(self):
        for handle in self.handles:
            handle.close()


def _open_ieeg_source(args, subject, h5_file):
    ieeg = h5_file["data/ieeg"]
    part_files = _total_part_files(h5_file)
    if _vds_sources_match_parts(ieeg, part_files):
        return _DatasetIEEGSource(ieeg)

    source_files = [os.path.basename(source.file_name) for source in ieeg.virtual_sources()]
    print(
        f"{subject}: total.h5 VDS sources do not match info/files; "
        f"reading part files directly. VDS starts with {source_files[:3]}, info/files starts with {part_files[:3]}",
        flush=True,
    )
    return _PartIEEGSource(args, subject, part_files)


def _is_valid_window(window):
    if not np.isfinite(window).all():
        return False
    if not np.any(window):
        return False
    return bool(np.min(np.std(window, axis=-1)) > 0)


def generate_subject_data(args, subject):
    path = total_file(args.data_root, subject)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Download the full subject folder first.")

    rng = np.random.default_rng(args.random_seed)
    with h5py.File(path, "r") as f:
        ieeg_source = _open_ieeg_source(args, subject, f)
        ch_num, total_len = ieeg_source.shape
        source_sfreq = int(f.attrs["sampling_rate"])
        seizures = read_seizures(f)
        if not seizures:
            print(f"{subject}: no seizure intervals, skipped before HDF5 window loading", flush=True)
            ieeg_source.close()
            return False
        source_window_len = int(round(args.seq_len * args.patch_secs * source_sfreq))
        target_window_len = int(round(args.seq_len * args.patch_secs * args.sfreq))
        normal_ratio = args.normal_ratio if args.sample_normal_windows else None
        starts, labels = select_window_starts(
            total_len,
            source_sfreq,
            source_window_len,
            seizures,
            args.label_thres,
            normal_ratio,
            rng,
        )

        data, out_label = [], []
        skipped_bad_window = 0
        for start, label in zip(starts, labels):
            window = ieeg_source.read_window(int(start), source_window_len)
            window = np.asarray(window, dtype=np.float32)
            if not _is_valid_window(window):
                skipped_bad_window += 1
                continue
            window = resample_window(window, source_sfreq, args.sfreq)
            window = window[:, :target_window_len]
            if window.shape[-1] < target_window_len:
                continue
            if not _is_valid_window(window):
                skipped_bad_window += 1
                continue
            data.append(window)
            out_label.append(int(label))
        ieeg_source.close()

    if not data:
        print(f"{subject}: no valid windows, skipped; bad windows={skipped_bad_window}")
        return False

    data = np.stack(data, axis=0).astype(np.float32)
    label = np.asarray(out_label, dtype=np.int64)
    if int(label.sum()) == 0:
        print(f"{subject}: no seizure windows, skipped")
        return False

    save_dir = _subject_save_dir(args, subject)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "data.npy"), data)
    np.save(os.path.join(save_dir, "label.npy"), label)
    write_json(os.path.join(save_dir, "channel_lst.json"), _channel_names(ch_num))
    write_json(
        os.path.join(save_dir, "summary.json"),
        {
            "subject": subject,
            "source_sfreq": source_sfreq,
            "target_sfreq": args.sfreq,
            "channel_num": ch_num,
            "window_num": int(label.shape[0]),
            "seizure_window_num": int(label.sum()),
            "seizure_event_num": len(seizures),
            "skipped_bad_window_num": int(skipped_bad_window),
            "sample_normal_windows": bool(args.sample_normal_windows),
            "normal_ratio": args.normal_ratio if args.sample_normal_windows else None,
        },
    )
    print(f"{subject}: saved {data.shape}, seizure windows={int(label.sum())}, skipped bad windows={skipped_bad_window}")
    return True


def generate_data(args, subjects=None):
    os.makedirs(args.data_save_dir, exist_ok=True)
    processed_subjects = []
    for subject in _available_subjects(args, subjects):
        try:
            if generate_subject_data(args, subject):
                processed_subjects.append(subject)
        except Exception as exc:
            print(f"{subject}: failed ({exc})", flush=True)
    return _make_patient_groups(processed_subjects)


def generate_subject_groups(args, subjects=None):
    subjects = subjects or [
        name for name in os.listdir(args.data_save_dir)
        if os.path.exists(os.path.join(args.data_save_dir, name, "data.npy"))
    ]
    groups = _make_patient_groups(subjects)
    path = os.path.join(args.data_save_dir, "subject_groups.json")
    write_json(path, groups)
    print(f"subject groups saved to {path}")
    return groups


def group_data(args, groups=None):
    if groups is None:
        group_json = os.path.join(args.data_save_dir, "subject_groups.json")
        if os.path.exists(group_json):
            with open(group_json) as f:
                groups = json.load(f)
        else:
            groups = generate_subject_groups(args)
    group_dir = os.path.join(args.data_save_dir, "group_data")
    os.makedirs(group_dir, exist_ok=True)

    for group_id, subjects in enumerate(groups):
        data, label = [], []
        for subject in subjects:
            sub_dir = _subject_save_dir(args, subject)
            data_path = os.path.join(sub_dir, "data.npy")
            label_path = os.path.join(sub_dir, "label.npy")
            if not os.path.exists(data_path):
                continue
            sub_data = np.load(data_path)
            sub_label = np.load(label_path)
            if args.sample_normal_windows:
                if int(sub_label.sum()) == 0:
                    print(f"{subject}: no seizure windows in saved data, skipped")
                    continue
                data.append(sub_data)
                label.append(sub_label)
            else:
                seizure_pos = np.where(sub_label == 1)[0]
                normal_pos = np.where(sub_label == 0)[0]
                if len(seizure_pos) == 0:
                    print(f"{subject}: no seizure windows in saved data, skipped")
                    continue
                rng = np.random.default_rng(args.random_seed + group_id)
                normal_num = min(len(normal_pos), int(len(seizure_pos) * args.normal_ratio))
                normal_pos = rng.permutation(normal_pos)[:normal_num]
                pos = np.concatenate([seizure_pos, normal_pos])
                if len(pos) == 0:
                    continue
                pos = rng.permutation(pos)
                data.append(sub_data[pos])
                label.append(sub_label[pos])
        if not data:
            print(f"group {group_id}: no data, skipped")
            continue
        data = np.concatenate(data, axis=0)
        label = np.concatenate(label, axis=0)
        np.save(os.path.join(group_dir, f"group_{group_id}_data.npy"), data)
        np.save(os.path.join(group_dir, f"group_{group_id}_label.npy"), label)
        if len(subjects) == 1:
            channel_json = os.path.join(_subject_save_dir(args, subjects[0]), "channel_lst.json")
            if os.path.exists(channel_json):
                with open(channel_json) as f:
                    channels = json.load(f)
                write_json(os.path.join(group_dir, f"group_{group_id}_channel_lst.json"), channels)
        print(f"group {group_id}: saved {data.shape}, seizure={int(label.sum())}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess SWEC iEEG Dataset.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Subject IDs, e.g. ID01 ID04.")
    parser.add_argument("--generate-subjects", action="store_true")
    parser.add_argument("--generate-groups", action="store_true")
    parser.add_argument("--group-data", action="store_true")
    parser.add_argument(
        "--save-all-windows",
        action="store_true",
        help="Save all fixed windows before group-level normal sampling. This can be very large for SWEC.",
    )
    parsed = parser.parse_args()

    args = PreprocessArgs()
    if parsed.save_all_windows:
        args.sample_normal_windows = False
    subjects = parsed.subjects or subject_ids(args.subject_num)
    did_action = False

    if parsed.generate_subjects:
        did_action = True
        generate_data(args, subjects)
    if parsed.generate_groups:
        did_action = True
        generate_subject_groups(args, subjects)
    if parsed.group_data:
        did_action = True
        group_data(args)
    if not did_action:
        groups = generate_data(args, subjects)
        group_data(args, groups)


if __name__ == "__main__":
    main()
