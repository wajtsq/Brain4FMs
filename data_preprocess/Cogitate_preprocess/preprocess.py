import argparse
import copy
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append('/bench-mark')

import numpy as np
import pandas as pd

from data_preprocess.Cogitate_preprocess.config import PreprocessArgs


warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import mne

    mne.set_log_level("ERROR")
except ModuleNotFoundError:
    mne = None


EVENT_COLUMNS = [
    "event type",
    "block",
    "miniblock",
    "category",
    "identity",
    "orientation",
    "duration",
    "task_relevance",
    "response",
]


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _dataset_save_dir(args):
    return os.path.join(args.data_save_root, args.dataset_name)


def _subject_save_dir(args, subject):
    return os.path.join(_dataset_save_dir(args), subject)


def _subject_done(args, subject):
    save_dir = _subject_save_dir(args, subject)
    if not (
        os.path.exists(os.path.join(save_dir, "data.npy"))
        and os.path.exists(os.path.join(save_dir, "label.npy"))
        and os.path.exists(os.path.join(save_dir, "channel_lst.json"))
    ):
        return False
    summary_path = os.path.join(save_dir, "summary.json")
    if not os.path.exists(summary_path):
        return False
    try:
        with open(summary_path) as f:
            summary = json.load(f)
        return (
            summary.get("label_mode") == args.label_mode
            and summary.get("label_values") == list(args.label_values)
            and summary.get("task_relevance_values") == list(args.task_relevance_values)
            and summary.get("response_values") == list(args.response_values)
            and summary.get("balance_category_duration") == bool(args.balance_category_duration)
            and summary.get("epoch_tmin") == args.epoch_tmin
            and summary.get("epoch_tmax") == args.epoch_tmax
            and summary.get("baseline") == list(args.baseline)
            and summary.get("keep_unreferenced_channels") == bool(args.keep_unreferenced_channels)
            and summary.get("reject_by_annotation") is True
        )
    except (OSError, json.JSONDecodeError):
        return False


def list_subjects(args):
    prefix = f"sub-{args.center_prefix}" if args.center_prefix else "sub-"
    subjects = [
        name for name in os.listdir(args.data_root)
        if name.startswith(prefix) and os.path.isdir(os.path.join(args.data_root, name))
    ]
    return sorted(subjects)[: args.subject_num]


def _ieeg_dir(args, subject):
    return os.path.join(args.data_root, subject, f"ses-{args.session}", "ieeg")


def _bids_base(args, subject):
    return os.path.join(_ieeg_dir(args, subject), f"{subject}_ses-{args.session}_task-{args.task}")


def _vhdr_path(args, subject):
    return f"{_bids_base(args, subject)}_ieeg.vhdr"


def _events_path(args, subject):
    return f"{_bids_base(args, subject)}_events.tsv"


def _channels_path(args, subject):
    return f"{_bids_base(args, subject)}_channels.tsv"


def _laplace_mapping_path(args, subject):
    return os.path.join(_ieeg_dir(args, subject), f"{subject}_ses-{args.session}_laplace_mapping_ieeg.json")


def _bad_description(value, bad_descriptions):
    if pd.isna(value):
        return False
    value = str(value)
    return any(desc in value for desc in bad_descriptions)


def _bad_annotation_intervals(raw):
    """Return MNE BAD/EDGE annotation intervals in recording-time seconds."""
    intervals = []
    for onset, duration, description in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
    ):
        if str(description).lower().startswith(("bad", "edge")):
            intervals.append((float(onset), float(onset + duration)))
    return intervals


def _epoch_overlaps_bad_annotation(epoch_start, epoch_end, intervals):
    return any(epoch_start < bad_end and epoch_end > bad_start for bad_start, bad_end in intervals)


def read_good_channels(args, subject):
    path = _channels_path(args, subject)
    channels = pd.read_csv(path, sep="\t")
    keep_type = channels["type"].astype(str).str.upper().isin(args.channel_types)
    keep_status = channels["status"].fillna("good").astype(str).str.lower() == "good"
    keep_desc = ~channels["status_description"].apply(
        lambda value: _bad_description(value, args.bad_channel_descriptions)
    )
    selected = channels.loc[keep_type & keep_status & keep_desc, "name"].astype(str).tolist()
    return selected


def _parse_trial_type(trial_type):
    parts = str(trial_type).split("/")
    if len(parts) != len(EVENT_COLUMNS):
        return None
    return dict(zip(EVENT_COLUMNS, parts))


def _factor_counts(rows, factors):
    counts = {}
    for row in rows:
        key = " × ".join(str(row[factor]) for factor in factors)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _balance_category_duration(args, subject, rows):
    """Balance the 4 category x 3 duration cells within one subject."""
    if not args.balance_category_duration or args.label_mode != "category" or not rows:
        return rows, {}

    factors = ("category", "duration")
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[factor] for factor in factors), []).append(row)
    durations = sorted({row["duration"] for row in rows})
    expected = {(category, duration) for category in args.label_values for duration in durations}
    missing = sorted(expected - set(grouped))
    if missing:
        raise ValueError(
            f"{subject}: cannot balance category x duration; missing cells: {missing}"
        )
    if len(grouped) < 2:
        return rows, _factor_counts(rows, factors)

    target_count = min(len(group) for group in grouped.values())
    # Use a stable subject-specific seed; Python's hash() is intentionally
    # randomized between processes and would not reproduce the sampling.
    subject_seed = sum((index + 1) * ord(char) for index, char in enumerate(subject))
    rng = np.random.default_rng(args.random_seed + subject_seed)
    balanced = []
    for key in sorted(grouped):
        group = grouped[key]
        selected = rng.choice(len(group), size=target_count, replace=False)
        balanced.extend(group[index] for index in sorted(selected))

    # Keep trial order chronological after sampling.
    balanced.sort(key=lambda row: row["onset"])
    return balanced, _factor_counts(balanced, factors)


def read_trial_metadata(args, subject):
    events = pd.read_csv(_events_path(args, subject), sep="\t", encoding="utf-8-sig")
    parsed_rows = []
    for row in events.itertuples(index=False):
        meta = _parse_trial_type(row.trial_type)
        if meta is None:
            continue
        if meta["event type"] != args.event_type:
            continue
        if args.task_relevance_values and meta["task_relevance"] not in args.task_relevance_values:
            continue
        if args.response_values and meta["response"] not in args.response_values:
            continue
        if args.label_mode == "category" and meta["category"] not in args.label_values:
            continue
        meta["onset"] = float(row.onset)
        parsed_rows.append(meta)
    balanced_rows, balance_counts = _balance_category_duration(args, subject, parsed_rows)
    return balanced_rows, {
        "trial_counts_before_balance": _factor_counts(parsed_rows, ("category", "duration")),
        "trial_counts_after_balance": balance_counts,
    }


def _label_for_trial(args, meta):
    """Map the experimental factor to a label without mixing factors."""
    relevance = meta["task_relevance"]
    if args.label_mode == "target_detection":
        # Relevant targets are the infrequent stimuli participants had to
        # report. Both relevant non-targets and irrelevant stimuli are
        # non-targets for this contrast.
        return "target" if relevance == "Relevant target" else "non_target"
    if args.label_mode == "task_relevance":
        return (
            "task_relevant"
            if relevance in ("Relevant target", "Relevant non-target")
            else "task_irrelevant"
        )
    if args.label_mode == "category":
        return meta["category"]
    raise ValueError(f"Unsupported label mode: {args.label_mode}")


def _notch_freqs(args, sfreq):
    if not args.notch_filter:
        return None
    nyquist = sfreq / 2.0
    if not args.notch_harmonics:
        return [args.notch_filter] if args.notch_filter < nyquist else None
    freqs = []
    freq = float(args.notch_filter)
    while freq < nyquist:
        freqs.append(freq)
        freq += float(args.notch_filter)
    return freqs or None


def _apply_laplace_reference(args, subject, data, channels):
    path = _laplace_mapping_path(args, subject)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{subject}: missing Laplace mapping file: {path}")

    with open(path) as f:
        mapping = json.load(f)

    ch_to_idx = {ch: idx for idx, ch in enumerate(channels)}
    referenced = np.zeros(len(channels), dtype=bool)
    out = data.copy()
    missing_ref_channels = {}
    unmapped_channels = []

    for ch_idx, ch_name in enumerate(channels):
        if ch_name not in mapping:
            unmapped_channels.append(ch_name)
            continue

        refs = [
            ref_name for ref_name in (mapping[ch_name].get("ref_1"), mapping[ch_name].get("ref_2"))
            if ref_name is not None and ref_name in ch_to_idx
        ]
        missing_refs = [
            ref_name for ref_name in (mapping[ch_name].get("ref_1"), mapping[ch_name].get("ref_2"))
            if ref_name is not None and ref_name not in ch_to_idx
        ]
        if missing_refs:
            missing_ref_channels[ch_name] = missing_refs
        if not refs:
            continue

        ref_data = np.stack([data[ch_to_idx[ref_name]] for ref_name in refs], axis=0)
        out[ch_idx] = data[ch_idx] - ref_data.mean(axis=0)
        referenced[ch_idx] = True

    if not args.keep_unreferenced_channels:
        out = out[referenced]
        channels = [ch for ch, keep in zip(channels, referenced) if keep]

    stats = {
        "reference": "laplace",
        "laplace_mapping_path": path,
        "laplace_referenced_channel_num": int(referenced.sum()),
        "laplace_unreferenced_channel_num": int((~referenced).sum()),
        "laplace_unmapped_channels": unmapped_channels,
        "laplace_missing_ref_channels": missing_ref_channels,
        "keep_unreferenced_channels": bool(args.keep_unreferenced_channels),
    }
    return out, channels, stats


def _load_raw_data(args, subject, channels):
    if mne is None:
        raise ImportError("mne is required to preprocess Cogitate BrainVision files")

    raw = mne.io.read_raw_brainvision(_vhdr_path(args, subject), preload=False, verbose=False)
    picks = [ch for ch in channels if ch in raw.ch_names]
    missing = sorted(set(channels) - set(picks))
    if not picks:
        raise ValueError(f"{subject}: no selected channels found in raw file")
    raw.pick_channels(picks, ordered=True)
    raw.load_data(verbose=False)

    sfreq = float(raw.info["sfreq"])
    if args.resample_before_filter and args.sfreq is not None and int(round(sfreq)) != int(args.sfreq):
        raw.resample(args.sfreq, n_jobs=args.mne_n_jobs, verbose=False)
        sfreq = float(raw.info["sfreq"])

    freqs = _notch_freqs(args, sfreq)
    if freqs is not None:
        raw.notch_filter(freqs=freqs, n_jobs=args.mne_n_jobs, verbose=False)
    if args.high_pass_filter is not None or args.low_pass_filter is not None:
        raw.filter(
            l_freq=args.high_pass_filter,
            h_freq=args.low_pass_filter,
            n_jobs=args.mne_n_jobs,
            verbose=False,
        )
    if not args.resample_before_filter and args.sfreq is not None and int(round(sfreq)) != int(args.sfreq):
        raw.resample(args.sfreq, n_jobs=args.mne_n_jobs, verbose=False)

    bad_annotation_intervals = _bad_annotation_intervals(raw)
    data = raw.get_data().astype(np.float32)
    reference = args.reference.lower()
    reference_stats = {"reference": reference}
    if reference == "car":
        data = data - data.mean(axis=0, keepdims=True)
        reference_stats = {"reference": "car"}
    elif reference == "laplace":
        data, picks, reference_stats = _apply_laplace_reference(args, subject, data, picks)
    elif reference in ("none", ""):
        pass
    else:
        raise ValueError(f"Unsupported reference: {args.reference}")
    return (
        data,
        int(round(raw.info["sfreq"])),
        list(picks),
        missing,
        reference_stats,
        bad_annotation_intervals,
    )


def _is_valid_epoch(epoch):
    if not np.isfinite(epoch).all():
        return False
    if not np.any(epoch):
        return False
    return bool(np.min(np.std(epoch, axis=-1)) > 0)


def _extract_epochs(args, data, sfreq, trial_rows, bad_annotation_intervals=None):
    label_map = {value: idx for idx, value in enumerate(args.label_values)}
    start_offset = int(round(args.epoch_tmin * sfreq))
    epoch_len = int(round((args.epoch_tmax - args.epoch_tmin) * sfreq))
    baseline_start = 0
    baseline_end = int(round((0.0 - args.epoch_tmin) * sfreq))
    if args.baseline == (None, 0.0) and not (0 < baseline_end <= epoch_len):
        raise ValueError(
            "Baseline (None, 0.0) must overlap the epoch: "
            f"epoch=({args.epoch_tmin}, {args.epoch_tmax})"
        )
    epochs, labels, kept_meta = [], [], []
    skipped_out_of_bounds = 0
    skipped_bad_epoch = 0
    skipped_bad_annotation = 0
    bad_annotation_intervals = bad_annotation_intervals or []

    for meta in trial_rows:
        start = int(round(meta["onset"] * sfreq)) + start_offset
        end = start + epoch_len
        if start < 0 or end > data.shape[1]:
            skipped_out_of_bounds += 1
            continue
        epoch_start_sec = start / sfreq
        epoch_end_sec = end / sfreq
        if _epoch_overlaps_bad_annotation(
            epoch_start_sec, epoch_end_sec, bad_annotation_intervals
        ):
            skipped_bad_annotation += 1
            continue
        epoch = data[:, start:end]
        if epoch.shape[1] != epoch_len or not _is_valid_epoch(epoch):
            skipped_bad_epoch += 1
            continue
        if args.baseline == (None, 0.0):
            epoch = epoch - epoch[:, baseline_start:baseline_end].mean(
                axis=1, keepdims=True
            )
        epochs.append(epoch)
        label = _label_for_trial(args, meta)
        if label not in label_map:
            raise ValueError(f"Label {label!r} is not present in label_values")
        labels.append(label_map[label])
        kept_meta.append({**meta, "label": label})

    stats = {
        "skipped_out_of_bounds": skipped_out_of_bounds,
        "skipped_bad_epoch": skipped_bad_epoch,
        "skipped_bad_annotation": skipped_bad_annotation,
    }
    if not epochs:
        return None, None, [], stats
    return np.stack(epochs, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64), kept_meta, stats


def generate_subject_data(args, subject):
    if args.skip_existing and _subject_done(args, subject):
        print(f"{subject}: existing preprocessed files found, skipped", flush=True)
        return True

    required = [_vhdr_path(args, subject), _events_path(args, subject), _channels_path(args, subject)]
    missing_files = [path for path in required if not os.path.exists(path)]
    if missing_files:
        print(f"{subject}: missing files, skipped: {missing_files}", flush=True)
        return False

    channels = read_good_channels(args, subject)
    if not channels:
        print(f"{subject}: no good {args.channel_types} channels, skipped", flush=True)
        return False

    trial_rows, trial_balance_stats = read_trial_metadata(args, subject)
    if not trial_rows:
        print(f"{subject}: no matching trials, skipped", flush=True)
        return False

    (
        data,
        sfreq,
        raw_channels,
        missing_channels,
        reference_stats,
        bad_annotation_intervals,
    ) = _load_raw_data(args, subject, channels)
    epochs, labels, kept_meta, stats = _extract_epochs(
        args, data, sfreq, trial_rows, bad_annotation_intervals
    )
    if epochs is None:
        print(f"{subject}: no valid epochs, skipped", flush=True)
        return False

    save_dir = _subject_save_dir(args, subject)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "data.npy"), epochs)
    np.save(os.path.join(save_dir, "label.npy"), labels)
    _write_json(os.path.join(save_dir, "channel_lst.json"), raw_channels)
    _write_json(
        os.path.join(save_dir, "summary.json"),
        {
            "subject": subject,
            "center_prefix": args.center_prefix,
            "dataset_name": args.dataset_name,
            "source_task": args.task,
            "channel_types": list(args.channel_types),
            "reference": args.reference,
            "resample_before_filter": bool(args.resample_before_filter),
            "sfreq": sfreq,
            "channel_num": len(raw_channels),
            "missing_channel_num": len(missing_channels),
            "epoch_num": int(labels.shape[0]),
            "label_values": list(args.label_values),
            "label_mode": args.label_mode,
            "label_counts": {
                args.label_values[idx]: int((labels == idx).sum())
                for idx in range(len(args.label_values))
            },
            "target_column": args.target_column,
            "event_type": args.event_type,
            "task_relevance_values": list(args.task_relevance_values),
            "response_values": list(args.response_values),
            "reject_by_annotation": True,
            "bad_annotation_interval_num": len(bad_annotation_intervals),
            "balance_category_duration": bool(args.balance_category_duration),
            "epoch_tmin": args.epoch_tmin,
            "epoch_tmax": args.epoch_tmax,
            "baseline": list(args.baseline),
            **reference_stats,
            **trial_balance_stats,
            **stats,
        },
    )
    metadata = [
        {key: value for key, value in meta.items() if key != "onset"} | {"onset": meta["onset"]}
        for meta in kept_meta
    ]
    _write_json(os.path.join(save_dir, "trial_metadata.json"), metadata)
    print(f"{subject}: saved {epochs.shape}, labels={np.bincount(labels, minlength=len(args.label_values)).tolist()}")
    return True


def _generate_subject_worker(payload):
    args, subject = payload
    try:
        return subject, generate_subject_data(args, subject), None
    except Exception as exc:
        return subject, False, str(exc)


def generate_data(args, subjects=None):
    os.makedirs(_dataset_save_dir(args), exist_ok=True)
    processed_subjects = []
    subjects = list(subjects or list_subjects(args))
    if args.n_jobs <= 1:
        for subject in subjects:
            try:
                if generate_subject_data(args, subject):
                    processed_subjects.append(subject)
            except Exception as exc:
                print(f"{subject}: failed ({exc})", flush=True)
        return [[subject] for subject in processed_subjects]

    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = [executor.submit(_generate_subject_worker, (args, subject)) for subject in subjects]
        for future in as_completed(futures):
            subject, ok, error = future.result()
            if ok:
                processed_subjects.append(subject)
            else:
                print(f"{subject}: failed ({error})", flush=True)
    processed_subjects = sorted(processed_subjects)
    return [[subject] for subject in processed_subjects]


def generate_subject_groups(args, subjects=None):
    if subjects is None:
        subjects = [
            name for name in os.listdir(_dataset_save_dir(args))
            if name.startswith("sub-") and os.path.exists(os.path.join(_dataset_save_dir(args), name, "data.npy"))
        ]
    groups = [[subject] for subject in sorted(subjects)]
    _write_json(os.path.join(_dataset_save_dir(args), "subject_groups.json"), groups)
    return groups


def group_data(args, groups=None):
    if groups is None:
        group_path = os.path.join(_dataset_save_dir(args), "subject_groups.json")
        if os.path.exists(group_path):
            with open(group_path) as f:
                groups = json.load(f)
        else:
            groups = generate_subject_groups(args)

    group_dir = os.path.join(_dataset_save_dir(args), "group_data")
    os.makedirs(group_dir, exist_ok=True)
    for group_id, subjects in enumerate(groups):
        data, label = [], []
        channels = None
        for subject in subjects:
            sub_dir = _subject_save_dir(args, subject)
            data_path = os.path.join(sub_dir, "data.npy")
            label_path = os.path.join(sub_dir, "label.npy")
            if not os.path.exists(data_path) or not os.path.exists(label_path):
                continue
            data.append(np.load(data_path))
            label.append(np.load(label_path))
            channel_path = os.path.join(sub_dir, "channel_lst.json")
            if channels is None and os.path.exists(channel_path):
                with open(channel_path) as f:
                    channels = json.load(f)

        if not data:
            print(f"group {group_id}: no data, skipped")
            continue

        group_x = np.concatenate(data, axis=0)
        group_y = np.concatenate(label, axis=0)
        rng = np.random.default_rng(args.random_seed + group_id)
        perm = rng.permutation(group_y.shape[0])
        group_x = group_x[perm]
        group_y = group_y[perm]

        np.save(os.path.join(group_dir, f"group_{group_id}_data.npy"), group_x)
        np.save(os.path.join(group_dir, f"group_{group_id}_label.npy"), group_y)
        if channels is not None:
            _write_json(os.path.join(group_dir, f"group_{group_id}_channel_lst.json"), channels)
        print(f"group {group_id}: saved {group_x.shape}, labels={np.bincount(group_y, minlength=len(args.label_values)).tolist()}")

    _write_json(
        os.path.join(group_dir, "preprocess_info.json"),
        {
            "dataset": args.dataset_name,
            "center_prefix": args.center_prefix,
            "channel_types": list(args.channel_types),
            "target_column": args.target_column,
            "label_values": list(args.label_values),
            "label_mode": args.label_mode,
            "event_type": args.event_type,
            "task_relevance_values": list(args.task_relevance_values),
            "response_values": list(args.response_values),
            "balance_category_duration": bool(args.balance_category_duration),
            "epoch_tmin": args.epoch_tmin,
            "epoch_tmax": args.epoch_tmax,
            "baseline": list(args.baseline),
            "sfreq": args.sfreq,
            "reference": args.reference,
            "resample_before_filter": bool(args.resample_before_filter),
            "reject_by_annotation": True,
            "keep_unreferenced_channels": bool(args.keep_unreferenced_channels),
        },
    )


def _modality_args(args, modality):
    modality = modality.upper()
    out = copy.copy(args)
    out.channel_types = (modality,)
    modality_label = {"ECOG": "ECoG", "SEEG": "SEEG"}[modality]
    out.dataset_name = f"Cogitate-{args.center_prefix}-{modality_label}"
    return out


def main():
    parser = argparse.ArgumentParser(description="Preprocess Cogitate iEEG visual perception task.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Subject IDs, e.g. sub-CF103 sub-CE108.")
    parser.add_argument("--center-prefix", type=str, default=None, help="Subject center prefix, e.g. CF, CE, CG.")
    parser.add_argument("--modality", choices=("ecog", "seeg", "both"), default="both")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel subject workers.")
    parser.add_argument("--mne-n-jobs", type=int, default=None, help="MNE workers inside each subject.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute subjects even when npy files already exist.")
    parser.add_argument("--generate-subjects", action="store_true")
    parser.add_argument("--generate-groups", action="store_true")
    parser.add_argument("--group-data", action="store_true")
    parser.add_argument(
        "--label-mode",
        choices=("target-detection", "task-relevance", "category"),
        default=None,
        help="Label contrast: target vs non-target (default), task-relevant vs irrelevant, or stimulus category.",
    )
    parser.add_argument(
        "--correct-only",
        action="store_true",
        help="Keep only Hit/CorrRej trials; useful when decoding behaviorally correct target detection.",
    )
    parser.add_argument(
        "--four-class-category",
        action="store_true",
        help="Compatibility alias for --label-mode category.",
    )
    parsed = parser.parse_args()

    args = PreprocessArgs()
    if parsed.center_prefix is not None:
        args.center_prefix = parsed.center_prefix
    if parsed.n_jobs is not None:
        args.n_jobs = parsed.n_jobs
    if parsed.mne_n_jobs is not None:
        args.mne_n_jobs = parsed.mne_n_jobs
    if parsed.overwrite:
        args.skip_existing = False
    if parsed.label_mode is not None:
        args.label_mode = parsed.label_mode.replace("-", "_")
    if parsed.correct_only:
        args.response_values = ("Hit", "CorrRej")
    if parsed.four_class_category:
        args.label_mode = "category"
        args.target_column = "category"
        args.label_values = ("face", "object", "letter", "false")
        args.task_relevance_values = ()
    elif args.label_mode == "task_relevance":
        args.target_column = "task_relevance"
        args.label_values = ("task_relevant", "task_irrelevant")
    elif args.label_mode == "category":
        # Keep the configured category labels. The default is the official
        # face-vs-object binary benchmark; four-class mode is explicit via
        # --four-class-category.
        args.target_column = "category"

    modalities = ("ECOG", "SEEG") if parsed.modality == "both" else (parsed.modality.upper(),)
    for modality in modalities:
        modality_args = _modality_args(args, modality)
        subjects = parsed.subjects or list_subjects(modality_args)
        did_action = False
        if parsed.generate_subjects:
            did_action = True
            generate_data(modality_args, subjects)
        if parsed.generate_groups:
            did_action = True
            generate_subject_groups(modality_args, subjects)
        if parsed.group_data:
            did_action = True
            group_data(modality_args)
        if not did_action:
            groups = generate_data(modality_args, subjects)
            group_data(modality_args, groups)


if __name__ == "__main__":
    main()
