from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ai_assistant import (
    collect_context,
    dataset_prompt,
    heuristic_dataset_proposal,
    load_json,
    merge_proposals,
    openai_json,
    print_proposal_summary,
    save_json,
)
from common import (
    DATA_PREPROCESS_DIR,
    ROOT,
    UPDATE_SKILL_DIR,
    UTILS_DIR,
    dict_has_key,
    insert_dict_entry,
    parse_split,
    preprocess_dir_name,
    prompt_bool,
    prompt_float,
    prompt_int,
    prompt_text,
    py_compile,
    quote,
    safe_identifier,
    write_new,
)


def dataset_config(args: argparse.Namespace, split: list[int]) -> str:
    data_root = args.data_root or f"/raw_datasets/{args.name}"
    data_save_dir = args.data_save_dir or f"/datasets/{args.name}"
    high_pass = args.high_pass if args.high_pass is not None else 0.01
    low_pass = args.low_pass if args.low_pass is not None else args.sfreq / 3
    return f'''class PreprocessArgs:
    # Source: {args.source_url}
    dataset_name = {quote(args.name)}
    source_url = {quote(args.source_url)}

    data_root = {quote(data_root)}
    data_save_dir = {quote(data_save_dir)}

    modality = {quote(args.modality)}
    sfreq = {args.sfreq}
    raw_sfreq = {args.raw_sfreq if args.raw_sfreq else "None"}
    channel = {args.channel}
    n_class = {args.n_class}
    seq_len = {args.seq_len}
    patch_secs = 1

    group_num = {args.group_num}
    split = {split}
    various_ch_num = {args.various_ch_num}

    high_pass_filter = {high_pass}
    low_pass_filter = {low_pass}
    notch_filter = {args.notch_filter}
    quality_factor = {args.quality_factor}

    label_map = {{}}
    selected_channels = None
'''


def dataset_preprocess(args: argparse.Namespace, package_name: str) -> str:
    return f'''import json
import urllib.request
from pathlib import Path

import numpy as np

from data_preprocess.{package_name}.config import PreprocessArgs
from data_preprocess.utils import _band_pass_filter, _notch_filter


def download_raw_data(args):
    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "SOURCE.json"
    marker.write_text(json.dumps({{"source_url": args.source_url}}, indent=2), encoding="utf-8")
    if args.source_url.startswith(("http://", "https://")):
        target = root / Path(args.source_url.rstrip("/").split("/")[-1]).name
        if target.name and "." in target.name and not target.exists():
            urllib.request.urlretrieve(args.source_url, target)
            print(f"downloaded {{args.source_url}} -> {{target}}")
    else:
        print(f"manual source recorded in {{marker}}")


def _select_brain_channels(raw):
    picks = []
    for idx, _ in enumerate(raw.info["ch_names"]):
        ch_type = raw.get_channel_types(picks=[idx])[0].lower()
        if ch_type in {{"eeg", "seeg", "ecog"}}:
            picks.append(idx)
    if not picks:
        raise ValueError("No EEG/iEEG channels found. Add dataset-specific channel selection.")
    return raw.get_data(picks=picks), [raw.info["ch_names"][i] for i in picks]


def _normalize_sample_rate_and_filter(args, data, sfreq):
    if sfreq != args.sfreq:
        from math import gcd
        from scipy.signal import resample_poly

        source = int(round(sfreq))
        target = int(round(args.sfreq))
        factor = gcd(source, target)
        data = resample_poly(data, target // factor, source // factor, axis=-1)
        sfreq = args.sfreq

    low = min(args.low_pass_filter, sfreq / 3)
    data = _band_pass_filter(data, sfreq, args.high_pass_filter, low)
    if args.notch_filter:
        data = _notch_filter(data, sfreq, args.notch_filter, args.quality_factor)
    return data, sfreq


def _segment_labeled_data(args, data):
    ch_num, ch_len = data.shape
    seq_pts = int(args.patch_secs * args.sfreq) * args.seq_len
    seq_num = ch_len // seq_pts
    if seq_num == 0:
        return np.empty((0, ch_num, seq_pts), dtype=np.float32)
    data = data[:, :seq_num * seq_pts].reshape(ch_num, seq_num, seq_pts)
    return data.transpose(1, 0, 2).astype(np.float32)


def iter_labeled_recordings(args):
    """Yield (subject_id, data, label) items.

    TODO: implement dataset-specific parsing:
    - read raw signals from args.data_root
    - select EEG/iEEG channels only
    - align events/tasks with labels
    - yield data as (channel, time) and label as int
    """
    raise NotImplementedError("Implement iter_labeled_recordings for this dataset.")


def generate_subject_data(args):
    save_dir = Path(args.data_save_dir) / "subject_data"
    save_dir.mkdir(parents=True, exist_ok=True)
    channels_path = Path(args.data_save_dir) / "channels_lst.json"

    for subject_id, data, label in iter_labeled_recordings(args):
        sfreq = args.raw_sfreq or args.sfreq
        data, sfreq = _normalize_sample_rate_and_filter(args, data, sfreq)
        samples = _segment_labeled_data(args, data)
        labels = np.full(samples.shape[0], int(label), dtype=np.int64)
        np.save(save_dir / f"{{subject_id}}_data.npy", samples.astype(np.float32))
        np.save(save_dir / f"{{subject_id}}_label.npy", labels)
        if not channels_path.exists():
            channels_path.write_text(json.dumps(args.selected_channels or []), encoding="utf-8")
        print(f"saved subject {{subject_id}}: {{samples.shape}}")


def generate_group_data(args):
    subject_dir = Path(args.data_save_dir) / "subject_data"
    group_dir = Path(args.data_save_dir) / "group_data"
    group_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted(p.name[:-9] for p in subject_dir.glob("*_data.npy"))
    if not subjects:
        raise FileNotFoundError(f"No subject data found in {{subject_dir}}")

    groups = [[] for _ in range(args.group_num)]
    for idx, subject_id in enumerate(subjects):
        groups[idx % args.group_num].append(subject_id)

    for group_id, group_subjects in enumerate(groups):
        data_parts, label_parts = [], []
        pos_parts = []
        for subject_id in group_subjects:
            data_parts.append(np.load(subject_dir / f"{{subject_id}}_data.npy"))
            label_parts.append(np.load(subject_dir / f"{{subject_id}}_label.npy"))
            pos_path = subject_dir / f"{{subject_id}}_pos.npy"
            if pos_path.exists():
                pos_parts.append(np.load(pos_path))
        data = np.concatenate(data_parts, axis=0)
        label = np.concatenate(label_parts, axis=0)
        np.save(group_dir / f"group_{{group_id}}_data.npy", data)
        np.save(group_dir / f"group_{{group_id}}_label.npy", label)
        if args.various_ch_num:
            if not pos_parts:
                raise FileNotFoundError(
                    f"args.various_ch_num=True but no subject position files were found for group {{group_id}}"
                )
            pos = np.concatenate(pos_parts, axis=0)
            np.save(group_dir / f"group_{{group_id}}_pos.npy", pos)
        print(f"group {{group_id}}: data={{data.shape}}, label={{label.shape}}")


def validate_group_data(args):
    group_dir = Path(args.data_save_dir) / "group_data"
    expected_t = args.sfreq * args.seq_len
    for group_id in range(args.group_num):
        data = np.load(group_dir / f"group_{{group_id}}_data.npy")
        label = np.load(group_dir / f"group_{{group_id}}_label.npy")
        assert data.ndim == 3, data.shape
        assert label.ndim == 1, label.shape
        assert data.shape[0] == label.shape[0]
        assert data.shape[1] == args.channel
        assert data.shape[2] == expected_t, (data.shape, expected_t)
        assert np.min(label) >= 0
        assert np.max(label) < args.n_class
        if args.various_ch_num:
            pos = np.load(group_dir / f"group_{{group_id}}_pos.npy")
            assert pos.shape[0] == data.shape[0], (pos.shape, data.shape)
    print("group data validation passed")


def main():
    args = PreprocessArgs()
    download_raw_data(args)
    generate_subject_data(args)
    generate_group_data(args)
    validate_group_data(args)


if __name__ == "__main__":
    main()
'''


def add_dataset(args: argparse.Namespace) -> None:
    dataset_name = safe_identifier(args.name)
    package_name = preprocess_dir_name(dataset_name)
    split = parse_split(args.split, args.group_num)
    preprocess_dir = DATA_PREPROCESS_DIR / package_name

    write_new(preprocess_dir / "config.py", dataset_config(args, split), args.overwrite)
    write_new(preprocess_dir / "preprocess.py", dataset_preprocess(args, package_name), args.overwrite)
    write_new(preprocess_dir / "__init__.py", "", args.overwrite)

    data_path = args.data_path or f"/datasets/{dataset_name}/group_data"
    info_value = (
        "{'data_path': " + quote(data_path) + ",\n"
        f"        'group_num': {args.group_num},\n"
        f"        'split': {split},\n"
        f"        'various_ch_num': {args.various_ch_num},\n"
        f"        'n_class': {args.n_class},\n"
        f"        'sfreq': {args.sfreq},\n"
        f"        'channel': {args.channel},\n"
        f"        'seq_len': {args.seq_len},\n"
        f"        'downstream': {quote(args.downstream)},\n"
        "    }"
    )
    insert_dict_entry(UTILS_DIR / "data_info.py", "data_info_dict", dataset_name, info_value)
    loader_name = "default_get_data_with_pos" if args.various_ch_num else "default_get_data"
    insert_dict_entry(UTILS_DIR / "meta_info.py", "get_data_dict", dataset_name, loader_name)
    metric = "BinaryClassMetrics" if args.n_class == 2 else "MultiClassMetrics"
    insert_dict_entry(UTILS_DIR / "meta_info.py", "metrics_dict", dataset_name, metric)

    print(f"dataset scaffold created: {preprocess_dir}")
    print(f"registered dataset key: {dataset_name}")


def wizard(args: argparse.Namespace) -> None:
    name = prompt_text("Dataset name")
    modality = prompt_text("Modality EEG/iEEG", "EEG")
    default_sfreq = 500 if modality.upper() == "EEG" else 1000
    ns = argparse.Namespace(
        name=name,
        source_url=prompt_text("Download/source URL"),
        data_root=prompt_text("Raw data root", f"/raw_datasets/{name}"),
        data_save_dir=prompt_text("Processed data save dir", f"/datasets/{name}"),
        data_path=None,
        modality=modality,
        sfreq=prompt_int("Target sfreq", default_sfreq),
        raw_sfreq=None,
        channel=prompt_int("Final EEG/iEEG channel count"),
        seq_len=prompt_int("Seconds per label/sample"),
        n_class=prompt_int("Number of classes"),
        downstream=prompt_text("Downstream task", "disorder"),
        group_num=prompt_int("Group number", 5),
        various_ch_num=prompt_bool("Different channels across patients", False),
        split=None,
        high_pass=prompt_float("High-pass filter Hz", 0.01),
        low_pass=None,
        notch_filter=prompt_float("Notch filter Hz, use 0 to disable", 50.0),
        quality_factor=30,
        overwrite=prompt_bool("Overwrite existing scaffold", False),
    )
    if ns.notch_filter == 0:
        ns.notch_filter = None
    ns.raw_sfreq_text = prompt_text("Raw sfreq if known, blank if unknown", "")
    ns.raw_sfreq = float(ns.raw_sfreq_text) if ns.raw_sfreq_text else None
    add_dataset(ns)


def inspect_dataset(args: argparse.Namespace) -> None:
    root = Path(args.data_root)
    report: dict[str, object] = {
        "data_root": str(root),
        "exists": root.exists(),
        "files": 0,
        "extensions": {},
        "metadata_files": [],
        "raw_candidates": [],
        "npy_shapes": [],
        "mne_probe": [],
        "suggestions": {},
    }
    if not root.exists():
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    raw_exts = {".edf", ".bdf", ".fif", ".set", ".vhdr", ".eeg"}
    metadata_names = {"participants.tsv", "events.tsv", "channels.tsv", "participants.csv", "events.csv", "channels.csv"}
    files = [p for p in root.rglob("*") if p.is_file()]
    report["files"] = len(files)
    exts: dict[str, int] = {}
    raw_candidates = []
    metadata_files = []
    for path in files:
        ext = path.suffix.lower()
        exts[ext] = exts.get(ext, 0) + 1
        if ext in raw_exts:
            raw_candidates.append(str(path))
        if path.name.lower() in metadata_names:
            metadata_files.append(str(path))
    report["extensions"] = exts
    report["raw_candidates"] = raw_candidates[:20]
    report["metadata_files"] = metadata_files[:50]

    npy_shapes = []
    for path in files:
        if path.suffix.lower() == ".npy" and len(npy_shapes) < 10:
            try:
                arr = np.load(path, mmap_mode="r")
                npy_shapes.append({"path": str(path), "shape": list(arr.shape), "dtype": str(arr.dtype)})
            except Exception as exc:
                npy_shapes.append({"path": str(path), "error": str(exc)})
    report["npy_shapes"] = npy_shapes

    try:
        import mne

        probes = []
        for raw_path in raw_candidates[:5]:
            path = Path(raw_path)
            try:
                if path.suffix.lower() in {".edf", ".bdf"}:
                    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
                elif path.suffix.lower() == ".fif":
                    raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
                elif path.suffix.lower() == ".set":
                    raw = mne.io.read_raw_eeglab(path, preload=False, verbose=False)
                else:
                    continue
                types = raw.get_channel_types()
                brain_count = sum(t.lower() in {"eeg", "seeg", "ecog"} for t in types)
                probes.append({
                    "path": raw_path,
                    "sfreq": float(raw.info["sfreq"]),
                    "n_channels": len(raw.info["ch_names"]),
                    "brain_channels": brain_count,
                    "channel_types": sorted(set(types)),
                    "first_channels": raw.info["ch_names"][:20],
                })
            except Exception as exc:
                probes.append({"path": raw_path, "error": str(exc)})
        report["mne_probe"] = probes
        if probes and "sfreq" in probes[0]:
            sfreq = probes[0]["sfreq"]
            modality = "iEEG" if any("seeg" in p.get("channel_types", []) or "ecog" in p.get("channel_types", []) for p in probes) else "EEG"
            report["suggestions"] = {
                "raw_sfreq": sfreq,
                "target_sfreq": min(int(sfreq), 1000 if modality == "iEEG" else 500),
                "modality": modality,
                "channel": probes[0].get("brain_channels"),
            }
    except Exception as exc:
        report["mne_probe"] = [{"error": f"mne unavailable or failed: {exc}"}]

    output = Path(args.output) if args.output else UPDATE_SKILL_DIR / "reports" / "dataset_inspect_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report saved: {output}")


def validate_dataset(args: argparse.Namespace) -> None:
    name = safe_identifier(args.name)
    issues = []
    checks = []
    preprocess_dir = DATA_PREPROCESS_DIR / preprocess_dir_name(name)
    config_path = preprocess_dir / "config.py"
    preprocess_path = preprocess_dir / "preprocess.py"

    for path in [config_path, preprocess_path]:
        if path.exists():
            ok, output = py_compile(path)
            checks.append(f"py_compile {path.name}: {'ok' if ok else 'failed'}")
            if not ok:
                issues.append(output)
        else:
            issues.append(f"missing {path}")

    for dict_name, file_name in [
        ("data_info_dict", "data_info.py"),
        ("get_data_dict", "meta_info.py"),
        ("metrics_dict", "meta_info.py"),
    ]:
        path = UTILS_DIR / file_name
        ok = dict_has_key(path, dict_name, name)
        checks.append(f"{dict_name} registration: {'ok' if ok else 'missing'}")
        if not ok:
            issues.append(f"{name} missing from {dict_name}")

    group_path = Path(args.group_path) if args.group_path else None
    if group_path:
        if not group_path.exists():
            issues.append(f"group path does not exist: {group_path}")
        else:
            data_files = sorted(group_path.glob("group_*_data.npy"))
            label_files = sorted(group_path.glob("group_*_label.npy"))
            checks.append(f"group files: {len(data_files)} data, {len(label_files)} label")
            for data_file in data_files:
                group_id = data_file.name.replace("_data.npy", "")
                label_file = group_path / f"{group_id}_label.npy"
                if not label_file.exists():
                    issues.append(f"missing label for {data_file.name}")
                    continue
                data = np.load(data_file, mmap_mode="r")
                label = np.load(label_file, mmap_mode="r")
                if data.ndim != 3:
                    issues.append(f"{data_file.name} shape is not 3D: {data.shape}")
                if label.ndim != 1:
                    issues.append(f"{label_file.name} shape is not 1D: {label.shape}")
                if data.shape[0] != label.shape[0]:
                    issues.append(f"{data_file.name} sample count differs from label: {data.shape[0]} vs {label.shape[0]}")
                pos_file = group_path / f"{group_id}_pos.npy"
                if pos_file.exists():
                    pos = np.load(pos_file, mmap_mode="r")
                    if pos.shape[0] != data.shape[0]:
                        issues.append(f"{pos_file.name} sample count differs from data: {pos.shape[0]} vs {data.shape[0]}")

    print("\n".join(checks))
    if issues:
        print("issues:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("dataset validation passed")


def ai_inspect_dataset(args: argparse.Namespace) -> None:
    context = collect_context(args.doc)
    report = load_json(args.inspect_report)
    proposal = heuristic_dataset_proposal(args.name, args.source_url, context, report)
    if args.use_ai:
        ai_result = openai_json(
            dataset_prompt(args.name, args.source_url, context, report),
            model=args.ai_model,
        )
        proposal = merge_proposals(proposal, ai_result)

    output = Path(args.output) if args.output else UPDATE_SKILL_DIR / "reports" / f"{safe_identifier(args.name or 'dataset')}_ai_proposal.json"
    save_json(proposal, output)
    print_proposal_summary(proposal, "Dataset AI Proposal")
    print(f"proposal saved: {output}")


def _proposal_to_dataset_args(proposal: dict[str, object], args: argparse.Namespace) -> argparse.Namespace:
    name = args.name or proposal.get("dataset_name")
    if not name:
        raise ValueError("--name is required when proposal has no dataset_name")
    source_url = args.source_url or proposal.get("source_url") or "TODO_SOURCE_URL"
    filtering = proposal.get("filtering", {}) if isinstance(proposal.get("filtering"), dict) else {}
    modality = args.modality or proposal.get("modality") or "EEG"
    target_sfreq = args.sfreq or proposal.get("target_sfreq")
    channel = args.channel or proposal.get("channel")
    seq_len = args.seq_len or proposal.get("seq_len")
    n_class = args.n_class or proposal.get("n_class")
    missing = []
    for key, value in [("sfreq", target_sfreq), ("channel", channel), ("seq_len", seq_len), ("n_class", n_class)]:
        if value is None:
            missing.append(key)
    if missing:
        raise ValueError(f"proposal is missing required fields: {', '.join(missing)}")

    return argparse.Namespace(
        name=str(name),
        source_url=str(source_url),
        data_root=args.data_root,
        data_save_dir=args.data_save_dir,
        data_path=args.data_path,
        modality=str(modality),
        sfreq=int(target_sfreq),
        raw_sfreq=proposal.get("raw_sfreq"),
        channel=int(channel),
        seq_len=int(seq_len),
        n_class=int(n_class),
        downstream=str(args.downstream or proposal.get("downstream") or "disorder"),
        group_num=args.group_num,
        various_ch_num=args.various_ch_num,
        split=args.split,
        high_pass=filtering.get("high_pass"),
        low_pass=filtering.get("low_pass"),
        notch_filter=filtering.get("notch"),
        quality_factor=args.quality_factor,
        overwrite=args.overwrite,
    )


def ai_add_dataset(args: argparse.Namespace) -> None:
    proposal = load_json(args.proposal)
    ns = _proposal_to_dataset_args(proposal, args)
    print_proposal_summary(proposal, "Using Dataset Proposal")
    if not args.yes and not prompt_bool("Apply this proposal and create scaffold", False):
        print("cancelled")
        return
    add_dataset(ns)
    validate_dataset(argparse.Namespace(name=ns.name, group_path=None))


def ai_review_dataset(args: argparse.Namespace) -> None:
    name = safe_identifier(args.name)
    preprocess_dir = DATA_PREPROCESS_DIR / preprocess_dir_name(name)
    context = collect_context([str(preprocess_dir), str(UTILS_DIR / "data_info.py"), str(UTILS_DIR / "meta_info.py")])
    report = {
        "dataset_name": name,
        "preprocess_dir": str(preprocess_dir),
    }
    proposal = heuristic_dataset_proposal(name, None, context, report)
    if args.use_ai:
        ai_result = openai_json(
            dataset_prompt(name, None, context, report)
            + "\nReview the current scaffold. Focus on missing fields, unsafe assumptions, and validation risks.",
            model=args.ai_model,
        )
        proposal = merge_proposals(proposal, ai_result)
    output = Path(args.output) if args.output else UPDATE_SKILL_DIR / "reports" / f"{name}_dataset_ai_review.json"
    save_json(proposal, output)
    print_proposal_summary(proposal, "Dataset AI Review")
    print(f"review saved: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset updater for Brain4FMs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add = subparsers.add_parser("add", help="scaffold and register a dataset")
    add.add_argument("--name", required=True)
    add.add_argument("--source-url", required=True)
    add.add_argument("--data-root")
    add.add_argument("--data-save-dir")
    add.add_argument("--data-path")
    add.add_argument("--modality", choices=["EEG", "iEEG"], default="EEG")
    add.add_argument("--sfreq", type=int, required=True)
    add.add_argument("--raw-sfreq", type=float)
    add.add_argument("--channel", type=int, required=True)
    add.add_argument("--seq-len", type=int, required=True)
    add.add_argument("--n-class", type=int, required=True)
    add.add_argument("--downstream", default="disorder")
    add.add_argument("--group-num", type=int, default=5)
    add.add_argument("--various-ch-num", action="store_true", help="register dataset with various_ch_num=True and use default_get_data_with_pos")
    add.add_argument("--split", help="comma-separated train,val,test group counts")
    add.add_argument("--high-pass", type=float)
    add.add_argument("--low-pass", type=float)
    add.add_argument("--notch-filter", type=float, default=50)
    add.add_argument("--quality-factor", type=float, default=30)
    add.add_argument("--overwrite", action="store_true")
    add.set_defaults(func=add_dataset)

    wiz = subparsers.add_parser("wizard", help="interactive dataset scaffold")
    wiz.set_defaults(func=wizard)

    inspect = subparsers.add_parser("inspect", help="inspect a raw dataset folder and write a report")
    inspect.add_argument("--data-root", required=True)
    inspect.add_argument("--output")
    inspect.set_defaults(func=inspect_dataset)

    validate = subparsers.add_parser("validate", help="validate dataset scaffold and optional group files")
    validate.add_argument("--name", required=True)
    validate.add_argument("--group-path", help="optional path to generated group_data")
    validate.set_defaults(func=validate_dataset)

    ai_inspect = subparsers.add_parser("ai-inspect", help="read docs/reports and write an AI-assisted dataset proposal")
    ai_inspect.add_argument("--name")
    ai_inspect.add_argument("--source-url")
    ai_inspect.add_argument("--doc", action="append", help="paper, README, metadata file, or folder; can repeat")
    ai_inspect.add_argument("--inspect-report", help="JSON report from dataset inspect")
    ai_inspect.add_argument("--output")
    ai_inspect.add_argument("--use-ai", action="store_true", help="call OpenAI when OPENAI_API_KEY is set")
    ai_inspect.add_argument("--ai-model")
    ai_inspect.set_defaults(func=ai_inspect_dataset)

    ai_add = subparsers.add_parser("ai-add", help="create dataset scaffold from a proposal JSON")
    ai_add.add_argument("--proposal", required=True)
    ai_add.add_argument("--name")
    ai_add.add_argument("--source-url")
    ai_add.add_argument("--data-root")
    ai_add.add_argument("--data-save-dir")
    ai_add.add_argument("--data-path")
    ai_add.add_argument("--modality")
    ai_add.add_argument("--sfreq", type=int)
    ai_add.add_argument("--channel", type=int)
    ai_add.add_argument("--seq-len", type=int)
    ai_add.add_argument("--n-class", type=int)
    ai_add.add_argument("--downstream")
    ai_add.add_argument("--group-num", type=int, default=5)
    ai_add.add_argument("--various-ch-num", action="store_true")
    ai_add.add_argument("--split")
    ai_add.add_argument("--quality-factor", type=float, default=30)
    ai_add.add_argument("--overwrite", action="store_true")
    ai_add.add_argument("--yes", action="store_true")
    ai_add.set_defaults(func=ai_add_dataset)

    ai_review = subparsers.add_parser("ai-review", help="review an existing dataset scaffold")
    ai_review.add_argument("--name", required=True)
    ai_review.add_argument("--output")
    ai_review.add_argument("--use-ai", action="store_true")
    ai_review.add_argument("--ai-model")
    ai_review.set_defaults(func=ai_review_dataset)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
