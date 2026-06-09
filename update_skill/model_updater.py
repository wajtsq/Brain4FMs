from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from ai_assistant import (
    collect_context,
    heuristic_model_proposal,
    load_json,
    merge_proposals,
    model_prompt,
    openai_json,
    print_proposal_summary,
    save_json,
)
from common import (
    DATASETS_DIR,
    MODEL_DIR,
    UPDATE_SKILL_DIR,
    UTILS_DIR,
    dict_has_key,
    ensure_import,
    insert_dict_entry,
    prompt_bool,
    prompt_int,
    prompt_text,
    py_compile,
    safe_identifier,
    write_new,
)


PRETRAINED_ROOT = Path("/benchmark/pretrained_weights")
MODEL_CONFIG_PATH = MODEL_DIR / "model_config.py"


def _checkpoint_field_name(model_name: str) -> str:
    return f"{safe_identifier(model_name)}_path"


def _checkpoint_target_dir(model_name: str) -> Path:
    return PRETRAINED_ROOT / safe_identifier(model_name)


def _checkpoint_filename(checkpoint_url: str, model_name: str) -> str:
    parsed = urllib.parse.urlparse(checkpoint_url)
    filename = Path(urllib.parse.unquote(parsed.path)).name
    if not filename:
        return f"{safe_identifier(model_name)}_checkpoint"
    return filename


def _checkpoint_target_path(model_name: str, checkpoint_url: str) -> Path:
    return _checkpoint_target_dir(model_name) / _checkpoint_filename(checkpoint_url, model_name)


def _upsert_model_path_arg(model_name: str, checkpoint_url: str) -> tuple[str, Path, bool]:
    field_name = _checkpoint_field_name(model_name)
    target_path = _checkpoint_target_path(model_name, checkpoint_url)
    text = MODEL_CONFIG_PATH.read_text(encoding="utf-8")
    field_pattern = re.compile(rf"^(\s*){re.escape(field_name)}:\s*str\s*=\s*(['\"])(.*?)\2\s*$", re.MULTILINE)
    replacement_line = f"    {field_name}: str = {target_path.as_posix()!r}"
    if field_pattern.search(text):
        updated = field_pattern.sub(replacement_line, text, count=1)
        changed = updated != text
        if changed:
            MODEL_CONFIG_PATH.write_text(updated, encoding="utf-8")
        return field_name, target_path, changed

    marker = "@dataclass\nclass ModelPathArgs:\n"
    if marker not in text:
        raise ValueError(f"Could not find ModelPathArgs in {MODEL_CONFIG_PATH}")
    updated = text.replace(marker, marker + replacement_line + "\n", 1)
    MODEL_CONFIG_PATH.write_text(updated, encoding="utf-8")
    return field_name, target_path, True


def _download_checkpoint(model_name: str, checkpoint_url: str, overwrite: bool = False) -> Path:
    target_path = _checkpoint_target_path(model_name, checkpoint_url)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not overwrite:
        return target_path
    with urllib.request.urlopen(checkpoint_url, timeout=600) as response:
        with target_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    return target_path


def model_file(model_name: str, final_dim: int, github_url: str) -> str:
    return f'''import torch
from torch import nn, optim
from argparse import Namespace


class {model_name}_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.final_dim = {final_dim}
        args.tune_a_part = True
        return args

    @staticmethod
    def clsf_loss_func(args, model):
        if args.weights is None:
            return nn.CrossEntropyLoss()
        return nn.CrossEntropyLoss(
            torch.tensor(args.weights, dtype=torch.float32, device=torch.device(args.gpu_id))
        )

    @staticmethod
    def optimizer(args, model, clsf):
        return optim.AdamW([
            {{"params": list(model.parameters()), "lr": args.model_lr}},
            {{"params": list(clsf.parameters()), "lr": args.clsf_lr}},
        ], betas=(0.9, 0.99), eps=1e-8)

    @staticmethod
    def scheduler(optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)


class {model_name}(nn.Module):
    """Brain4FMs wrapper for {github_url}.

    This file is only a temporary stub. Replace it with a real integration
    before treating the model as added successfully.
    """

    def __init__(self, args: Namespace):
        super().__init__()
        raise NotImplementedError(
            "Real integration required for {model_name}. "
            "Read the source repository and replace this stub before use."
        )

    def _build_encoder(self, args):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        raise NotImplementedError
'''


def model_dataset_file(model_name: str) -> str:
    return f'''import torch
from torch.utils.data import Dataset, DataLoader

from data_preprocess.utils import _std_data_segment


class {model_name}Dataset(Dataset):
    def __init__(self, args, x, y):
        # x: (seq_num, ch_num, N)
        # y: (seq_num, )
        if isinstance(x, dict):
            x = x["x"]
        self.seq_num, self.ch_num, _ = x.shape
        self.x = _std_data_segment(x)
        self.y = y

        # Add model-specific metadata here when needed:
        # channel positions, masks, spectra, prompts, token ids, etc.
        self.nProcessLoader = args.n_process_loader
        self.reload_pool = torch.multiprocessing.Pool(self.nProcessLoader)

    def __getitem__(self, index):
        return self.x[index, :, :], self.y[index]

    def __len__(self):
        return self.seq_num

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle=shuffle,
        )
'''


def add_model(args: argparse.Namespace) -> None:
    model_name = safe_identifier(args.name)
    model_dir = MODEL_DIR / model_name
    write_new(model_dir / "__init__.py", "", args.overwrite)
    write_new(model_dir / f"{model_name}.py", model_file(model_name, args.final_dim, args.github_url), args.overwrite)

    if args.clone:
        vendor_dir = model_dir / "vendor"
        if vendor_dir.exists() and not args.overwrite:
            raise FileExistsError(f"{vendor_dir} already exists; pass --overwrite to replace it")
        subprocess.run(["git", "clone", "--depth", "1", args.github_url, str(vendor_dir)], check=True)

    dataset_class = "DefaultDataset"
    if args.custom_dataset:
        dataset_class = f"{model_name}Dataset"
        write_new(DATASETS_DIR / f"{model_name}_dataset.py", model_dataset_file(model_name), args.overwrite)
        ensure_import(UTILS_DIR / "meta_info.py", f"from datasets.{model_name}_dataset import {dataset_class}")

    ensure_import(UTILS_DIR / "meta_info.py", f"from model.{model_name}.{model_name} import {model_name}, {model_name}_Trainer")
    insert_dict_entry(UTILS_DIR / "meta_info.py", "dataset_class_dict", model_name, dataset_class)
    insert_dict_entry(UTILS_DIR / "meta_info.py", "trainer_dict", model_name, f"{model_name}_Trainer")
    insert_dict_entry(UTILS_DIR / "meta_info.py", "model_dict", model_name, model_name)

    checkpoint_url = getattr(args, "checkpoint_url", None)
    if checkpoint_url:
        field_name, target_path, changed = _upsert_model_path_arg(model_name, checkpoint_url)
        if changed:
            print(f"registered checkpoint path: ModelPathArgs.{field_name} -> {target_path}")
        else:
            print(f"checkpoint path already up to date: ModelPathArgs.{field_name} -> {target_path}")

        if getattr(args, "download_checkpoint", False):
            downloaded_path = _download_checkpoint(model_name, checkpoint_url, overwrite=args.overwrite)
            print(f"checkpoint downloaded: {downloaded_path}")

    print(f"model scaffold created: {model_dir}")
    print(f"registered model key: {model_name}")


def wizard(args: argparse.Namespace) -> None:
    name = prompt_text("Model name")
    checkpoint_url = prompt_text("Checkpoint download URL", "")
    ns = argparse.Namespace(
        name=name,
        github_url=prompt_text("GitHub URL"),
        checkpoint_url=checkpoint_url or None,
        final_dim=prompt_int("Final encoder dim", 768),
        custom_dataset=prompt_bool("Need a custom dataset wrapper", False),
        clone=prompt_bool("Clone GitHub repo into model/<name>/vendor", False),
        download_checkpoint=prompt_bool("Download checkpoint to /benchmark/pretrained_weights", bool(checkpoint_url)),
        overwrite=prompt_bool("Overwrite existing scaffold", False),
    )
    add_model(ns)


def _clone_for_inspection(github_url: str) -> Path:
    cache = UPDATE_SKILL_DIR / "_model_inspect_cache"
    cache.mkdir(parents=True, exist_ok=True)
    repo_name = safe_identifier(Path(github_url.rstrip("/")).stem or "repo")
    target = cache / repo_name
    if target.exists():
        return target
    subprocess.run(["git", "clone", "--depth", "1", github_url, str(target)], check=True)
    return target


def _grep_patterns(repo: Path) -> dict[str, object]:
    py_files = list(repo.rglob("*.py"))
    readme_files = [p for p in repo.rglob("*") if p.is_file() and p.name.lower().startswith("readme")]
    config_files = [
        p for p in repo.rglob("*")
        if p.is_file() and p.suffix.lower() in {".yaml", ".yml", ".json", ".toml"}
    ]
    report: dict[str, object] = {
        "python_files": len(py_files),
        "readme_files": [str(p) for p in readme_files[:10]],
        "config_files": [str(p) for p in config_files[:20]],
        "nn_module_classes": [],
        "brain_model_classes": [],
        "class_path_hints": [],
        "forward_signatures": [],
        "dimension_hints": [],
        "input_shape_hints": [],
        "checkpoint_hints": [],
        "requirements": [],
    }

    class_pat = re.compile(r"class\s+([A-Za-z_]\w*)\s*\(([^)]*(?:nn\.Module|Module)[^)]*)\)")
    brain_model_pat = re.compile(r"class\s+([A-Za-z_]\w*)\s*\(([^)]*(?:BrainModel|LightningModule)[^)]*)\)")
    forward_pat = re.compile(r"def\s+forward\s*\(([^)]*)\)")
    dim_pat = re.compile(
        r"(?:hidden_size|d_model|embed_dim|emb_size|final_dim|n_embd|size_output|size_input|segment_n|segment_size|n_channels)\s*[=:]\s*(\d+)",
        re.I,
    )
    shape_pat = re.compile(r"(?:input|shape|tensor)[^\n]{0,80}(?:B|batch)[^\n]{0,120}", re.I)
    ckpt_pat = re.compile(r"(?:checkpoint|ckpt|pretrain|pretrained|load_state_dict|torch\.load)[^\n]{0,160}", re.I)
    class_path_pat = re.compile(r"class_path:\s*([A-Za-z_][\w.]*)")
    batch_shape_pat = re.compile(r"x\s*:\s*\[[^\]]*(?:bsz|batch)[^\]]*\]", re.I)

    for path in py_files[:300]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for match in class_pat.finditer(text):
            report["nn_module_classes"].append({"file": str(path), "class": match.group(1)})
        for match in brain_model_pat.finditer(text):
            report["brain_model_classes"].append({"file": str(path), "class": match.group(1)})
        for match in forward_pat.finditer(text):
            report["forward_signatures"].append({"file": str(path), "signature": match.group(1)})
        for match in dim_pat.finditer(text):
            report["dimension_hints"].append({"file": str(path), "line": match.group(0)})
        for match in ckpt_pat.finditer(text):
            report["checkpoint_hints"].append({"file": str(path), "line": match.group(0).strip()})
        for match in batch_shape_pat.finditer(text):
            report["input_shape_hints"].append({"file": str(path), "line": match.group(0).strip()})

    for path in readme_files[:10]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in shape_pat.finditer(text):
            report["input_shape_hints"].append({"file": str(path), "line": match.group(0).strip()})
        for match in dim_pat.finditer(text):
            report["dimension_hints"].append({"file": str(path), "line": match.group(0)})
        for match in ckpt_pat.finditer(text):
            report["checkpoint_hints"].append({"file": str(path), "line": match.group(0).strip()})

    for path in config_files[:50]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in class_path_pat.finditer(text):
            report["class_path_hints"].append({"file": str(path), "class_path": match.group(1)})
        for match in dim_pat.finditer(text):
            report["dimension_hints"].append({"file": str(path), "line": match.group(0)})
        for match in ckpt_pat.finditer(text):
            report["checkpoint_hints"].append({"file": str(path), "line": match.group(0).strip()})
        for match in batch_shape_pat.finditer(text):
            report["input_shape_hints"].append({"file": str(path), "line": match.group(0).strip()})

    for req_name in ["requirements.txt", "environment.yml", "pyproject.toml", "setup.py"]:
        for path in repo.rglob(req_name):
            report["requirements"].append(str(path))

    final_dims = []
    preferred_dims = []
    dims = []
    for item in report["dimension_hints"]:
        line = item["line"]
        if re.search(r"\bfinal_dim\b", line, flags=re.I):
            final_dims.extend(int(n) for n in re.findall(r"\d+", line))
        if re.search(r"\b(?:n_embd|size_output|hidden_size|d_model|embed_dim|emb_size)\b", line, flags=re.I):
            preferred_dims.extend(int(n) for n in re.findall(r"\d+", line))
        numbers = re.findall(r"\d+", item["line"])
        dims.extend(int(n) for n in numbers)
    if final_dims:
        report["suggestions"] = {"possible_final_dim": final_dims[0]}
    elif preferred_dims:
        report["suggestions"] = {"possible_final_dim": max(preferred_dims)}
    elif dims:
        common_dims = sorted(set(dims), key=lambda x: (-dims.count(x), x))
        report["suggestions"] = {"possible_final_dim": common_dims[0]}
    else:
        report["suggestions"] = {"possible_final_dim": None}
    report["suggestions"]["custom_dataset_needed"] = "unknown; inspect input_shape_hints and forward_signatures"
    return report


def inspect_model(args: argparse.Namespace) -> None:
    if args.repo_path:
        repo = Path(args.repo_path)
    else:
        repo = _clone_for_inspection(args.github_url)
    report = {"repo": str(repo), "exists": repo.exists()}
    if repo.exists():
        report.update(_grep_patterns(repo))
    output = Path(args.output) if args.output else UPDATE_SKILL_DIR / "reports" / "model_inspect_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report saved: {output}")


def validate_model(args: argparse.Namespace) -> None:
    name = safe_identifier(args.name)
    model_path = MODEL_DIR / name / f"{name}.py"
    dataset_path = DATASETS_DIR / f"{name}_dataset.py"
    issues = []
    checks = []

    if model_path.exists():
        ok, output = py_compile(model_path)
        checks.append(f"py_compile {model_path.name}: {'ok' if ok else 'failed'}")
        if not ok:
            issues.append(output)
    else:
        issues.append(f"missing {model_path}")

    if dataset_path.exists():
        ok, output = py_compile(dataset_path)
        checks.append(f"py_compile {dataset_path.name}: {'ok' if ok else 'failed'}")
        if not ok:
            issues.append(output)

    meta = UTILS_DIR / "meta_info.py"
    for dict_name in ["dataset_class_dict", "trainer_dict", "model_dict"]:
        ok = dict_has_key(meta, dict_name, name)
        checks.append(f"{dict_name} registration: {'ok' if ok else 'missing'}")
        if not ok:
            issues.append(f"{name} missing from {dict_name}")

    if model_path.exists():
        text = model_path.read_text(encoding="utf-8")
        required = [f"class {name}_Trainer", f"class {name}", "def set_config", "def clsf_loss_func", "def optimizer", "def scheduler", "def forward_propagate"]
        for marker in required:
            ok = marker in text
            checks.append(f"{marker}: {'ok' if ok else 'missing'}")
            if not ok:
                issues.append(f"{marker} missing in {model_path}")
        placeholder_markers = [
            "Real integration required for",
            "Replace it with a real integration",
            "nn.LazyLinear(",
            "Replace `_build_encoder` and `forward`",
        ]
        for marker in placeholder_markers:
            if marker in text:
                issues.append(f"placeholder marker found in {model_path}: {marker}")

    print("\n".join(checks))
    if issues:
        print("issues:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("model validation passed")


def ai_inspect_model(args: argparse.Namespace) -> None:
    context_paths = list(args.doc or [])
    report = load_json(args.inspect_report)
    report_repo = report.get("repo") if isinstance(report, dict) else None
    if report_repo and Path(report_repo).exists():
        context_paths.append(str(report_repo))
    if args.repo_path:
        context_paths.append(args.repo_path)
        if not report:
            report = {"repo": args.repo_path, "exists": Path(args.repo_path).exists()}
            if Path(args.repo_path).exists():
                report.update(_grep_patterns(Path(args.repo_path)))
    context = collect_context(context_paths)
    proposal = heuristic_model_proposal(args.name, args.github_url, context, report)
    if args.checkpoint_url:
        proposal["checkpoint_url"] = args.checkpoint_url
    if args.use_ai:
        ai_result = openai_json(
            model_prompt(args.name, args.github_url, context, report),
            model=args.ai_model,
        )
        proposal = merge_proposals(proposal, ai_result)

    output = Path(args.output) if args.output else UPDATE_SKILL_DIR / "reports" / f"{safe_identifier(args.name or 'model')}_ai_proposal.json"
    save_json(proposal, output)
    print_proposal_summary(proposal, "Model AI Proposal")
    print(f"proposal saved: {output}")


def _proposal_to_model_args(proposal: dict[str, object], args: argparse.Namespace) -> argparse.Namespace:
    name = args.name or proposal.get("model_name")
    if not name:
        raise ValueError("--name is required when proposal has no model_name")
    github_url = args.github_url or proposal.get("github_url") or "TODO_GITHUB_URL"
    final_dim = args.final_dim or proposal.get("final_dim")
    if final_dim is None:
        raise ValueError("proposal is missing final_dim; pass --final-dim to override")
    custom_dataset = args.custom_dataset
    if custom_dataset is None:
        custom_dataset = bool(proposal.get("custom_dataset_needed"))
    checkpoint_url = args.checkpoint_url or proposal.get("checkpoint_url")
    if not checkpoint_url:
        raise ValueError("checkpoint_url is required; pass --checkpoint-url or include it in the proposal JSON")
    return argparse.Namespace(
        name=str(name),
        github_url=str(github_url),
        checkpoint_url=str(checkpoint_url) if checkpoint_url else None,
        final_dim=int(final_dim),
        custom_dataset=bool(custom_dataset),
        clone=args.clone,
        download_checkpoint=args.download_checkpoint,
        overwrite=args.overwrite,
    )


def ai_add_model(args: argparse.Namespace) -> None:
    proposal = load_json(args.proposal)
    ns = _proposal_to_model_args(proposal, args)
    print_proposal_summary(proposal, "Using Model Proposal")
    if not args.yes and not prompt_bool("Apply this proposal and create scaffold", False):
        print("cancelled")
        return
    add_model(ns)
    validate_model(argparse.Namespace(name=ns.name))


def ai_review_model(args: argparse.Namespace) -> None:
    name = safe_identifier(args.name)
    model_dir = MODEL_DIR / name
    context_paths = [str(model_dir), str(UTILS_DIR / "meta_info.py")]
    dataset_path = DATASETS_DIR / f"{name}_dataset.py"
    if dataset_path.exists():
        context_paths.append(str(dataset_path))
    context = collect_context(context_paths)
    report = {"repo": str(model_dir), "exists": model_dir.exists()}
    if model_dir.exists():
        report.update(_grep_patterns(model_dir))
    proposal = heuristic_model_proposal(name, None, context, report)
    if args.use_ai:
        ai_result = openai_json(
            model_prompt(name, None, context, report)
            + "\nReview the current scaffold. Focus on interface mismatch, final_dim, custom dataset need, and missing checkpoint logic.",
            model=args.ai_model,
        )
        proposal = merge_proposals(proposal, ai_result)
    output = Path(args.output) if args.output else UPDATE_SKILL_DIR / "reports" / f"{name}_model_ai_review.json"
    save_json(proposal, output)
    print_proposal_summary(proposal, "Model AI Review")
    print(f"review saved: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model updater for Brain4FMs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add = subparsers.add_parser("add", help="register a model entry and create a temporary stub that must be replaced by a real integration")
    add.add_argument("--name", required=True)
    add.add_argument("--github-url", required=True)
    add.add_argument("--checkpoint-url", required=True, help="download URL for the primary pretrained checkpoint")
    add.add_argument("--final-dim", type=int, default=768)
    add.add_argument("--custom-dataset", action="store_true")
    add.add_argument("--clone", action="store_true")
    add.add_argument("--download-checkpoint", action="store_true", help="download the checkpoint into /benchmark/pretrained_weights/<ModelName>/")
    add.add_argument("--overwrite", action="store_true")
    add.set_defaults(func=add_model)

    wiz = subparsers.add_parser("wizard", help="interactive model stub generator")
    wiz.set_defaults(func=wizard)

    inspect = subparsers.add_parser("inspect", help="inspect a GitHub repo or local model repo")
    inspect.add_argument("--github-url")
    inspect.add_argument("--repo-path")
    inspect.add_argument("--output")
    inspect.set_defaults(func=inspect_model)

    validate = subparsers.add_parser("validate", help="validate model integration and registration; fails on placeholder stubs")
    validate.add_argument("--name", required=True)
    validate.set_defaults(func=validate_model)

    ai_inspect = subparsers.add_parser("ai-inspect", help="read docs/code reports and write an AI-assisted model proposal")
    ai_inspect.add_argument("--name")
    ai_inspect.add_argument("--github-url")
    ai_inspect.add_argument("--checkpoint-url", help="download URL for the primary pretrained checkpoint")
    ai_inspect.add_argument("--repo-path")
    ai_inspect.add_argument("--doc", action="append", help="README, paper, source folder, or file; can repeat")
    ai_inspect.add_argument("--inspect-report", help="JSON report from model inspect")
    ai_inspect.add_argument("--output")
    ai_inspect.add_argument("--use-ai", action="store_true", help="call OpenAI when OPENAI_API_KEY is set")
    ai_inspect.add_argument("--ai-model")
    ai_inspect.set_defaults(func=ai_inspect_model)

    ai_add = subparsers.add_parser("ai-add", help="create model registration and temporary stub from a proposal JSON")
    ai_add.add_argument("--proposal", required=True)
    ai_add.add_argument("--name")
    ai_add.add_argument("--github-url")
    ai_add.add_argument("--checkpoint-url")
    ai_add.add_argument("--final-dim", type=int)
    custom_dataset_group = ai_add.add_mutually_exclusive_group()
    custom_dataset_group.add_argument("--custom-dataset", dest="custom_dataset", action="store_true")
    custom_dataset_group.add_argument("--no-custom-dataset", dest="custom_dataset", action="store_false")
    ai_add.set_defaults(custom_dataset=None)
    ai_add.add_argument("--clone", action="store_true")
    ai_add.add_argument("--download-checkpoint", action="store_true")
    ai_add.add_argument("--overwrite", action="store_true")
    ai_add.add_argument("--yes", action="store_true")
    ai_add.set_defaults(func=ai_add_model)

    ai_review = subparsers.add_parser("ai-review", help="review an existing model scaffold")
    ai_review.add_argument("--name", required=True)
    ai_review.add_argument("--output")
    ai_review.add_argument("--use-ai", action="store_true")
    ai_review.add_argument("--ai-model")
    ai_review.set_defaults(func=ai_review_model)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect" and not args.github_url and not args.repo_path:
        parser.error("inspect requires --github-url or --repo-path")
    args.func(args)


if __name__ == "__main__":
    main()
