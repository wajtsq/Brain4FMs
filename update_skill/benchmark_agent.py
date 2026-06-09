from __future__ import annotations

import argparse
import ast
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_assistant import read_text_file
from common import ROOT, UPDATE_SKILL_DIR, UTILS_DIR, safe_identifier


DEFAULT_KNOWLEDGE_DIR = ROOT.parent / "knowledge"
DEFAULT_KB_PATH = DEFAULT_KNOWLEDGE_DIR / "benchmark_kb.json"

METRIC_PRIORITY = ["AUROC", "Acc", "F1", "Kappa", "F2"]
DOWNSTREAM_ALIASES = {
    "sleep": "disorder",
    "stage": "disorder",
    "adhd": "disorder",
    "depression": "disorder",
    "mdd": "disorder",
    "schizophrenia": "disorder",
    "alzheimer": "disorder",
    "seizure": "disorder",
    "emotion": "emotion",
    "valence": "emotion",
    "arousal": "emotion",
    "motor": "Motor imagine",
    "imagery": "Motor imagine",
    "bci": "Motor imagine",
    "concept": "concept",
    "language": "concept",
}


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def lookup_dataset_info(name: str, datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if name in datasets:
        return datasets[name]
    normalized = normalize_name(name)
    for key, info in datasets.items():
        key_norm = normalize_name(key)
        if normalized == key_norm or normalized in key_norm or key_norm in normalized:
            return info
    return {}


def load_data_info() -> dict[str, dict[str, Any]]:
    text = (UTILS_DIR / "data_info.py").read_text(encoding="utf-8")
    match = re.search(r"data_info_dict\s*=\s*(\{.*\})\s*$", text, flags=re.DOTALL)
    if not match:
        return {}
    return ast.literal_eval(match.group(1))


def parse_markdown_tables(readme_path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    text = readme_path.read_text(encoding="utf-8", errors="ignore")
    models: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    current_header: list[str] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            current_header = None
            continue

        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue

        if "Model Name" in cells and "Dataset" in cells:
            current_header = cells
            continue
        if "Mode Name" in cells and "paper" in cells:
            current_header = cells
            continue
        if current_header is None:
            continue

        if "Mode Name" in current_header and len(cells) >= 2:
            model_name = cells[current_header.index("Mode Name")]
            models[model_name] = {
                "name": model_name,
                "paper": cells[current_header.index("paper")] if "paper" in current_header else "",
                "code": cells[current_header.index("code")] if "code" in current_header else "",
            }
            continue

        if "Model Name" in current_header and "Dataset" in current_header:
            row = {key: cells[idx] if idx < len(cells) else "" for idx, key in enumerate(current_header)}
            row["metrics"] = {
                key: parse_metric_value(value)
                for key, value in row.items()
                if key not in {"Model Name", "Dataset"} and parse_metric_value(value) is not None
            }
            if row["metrics"]:
                results.append(row)

    return models, results


def parse_metric_value(value: str) -> float | None:
    cleaned = re.sub(r"[*_`]", "", value)
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return float(match.group(0)) if match else None


def collect_manual_notes(knowledge_dir: Path) -> list[dict[str, Any]]:
    notes_dir = knowledge_dir / "notes"
    if not notes_dir.exists():
        return []
    notes = []
    for path in sorted(notes_dir.glob("*.md")):
        notes.append({"path": str(path), "text": path.read_text(encoding="utf-8", errors="ignore")})
    return notes


def build_kb(args: argparse.Namespace) -> None:
    knowledge_dir = Path(args.knowledge_dir)
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    readme_path = Path(args.readme)
    models, results = parse_markdown_tables(readme_path)
    data_info = load_data_info()

    for row in results:
        dataset_name = row.get("Dataset")
        info = lookup_dataset_info(dataset_name, data_info)
        if info:
            row["dataset_info"] = info

    paper_text = ""
    paper_note = None
    if args.paper:
        paper_path = Path(args.paper)
        paper_text = read_text_file(paper_path, limit=args.paper_chars)
        if paper_text.startswith("[PDF text extraction unavailable"):
            paper_note = paper_text
            paper_text = ""

    kb = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sources": {
            "readme": str(readme_path),
            "paper": str(args.paper) if args.paper else None,
            "paper_note": paper_note,
            "data_info": str(UTILS_DIR / "data_info.py"),
        },
        "datasets": data_info,
        "models": models,
        "results": results,
        "paper_excerpt": paper_text,
        "manual_notes": collect_manual_notes(knowledge_dir),
    }

    output = Path(args.output) if args.output else knowledge_dir / "benchmark_kb.json"
    output.write_text(json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"knowledge base saved: {output}")
    print(f"datasets: {len(data_info)}, models: {len(models)}, result rows: {len(results)}")
    if paper_note:
        print(paper_note)


def add_note(args: argparse.Namespace) -> None:
    knowledge_dir = Path(args.knowledge_dir)
    notes_dir = knowledge_dir / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    slug = safe_identifier(args.title).lower()
    path = notes_dir / f"{slug}.md"
    tags = ", ".join(args.tag or [])
    content = f"# {args.title}\n\nTags: {tags}\n\n{args.text.strip()}\n"
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.write_text(content, encoding="utf-8")
    print(f"note saved: {path}")


def load_kb(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {path}. Run `agent build-kb` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def infer_downstream(question: str) -> str | None:
    q = question.lower()
    for key, value in DOWNSTREAM_ALIASES.items():
        if key in q:
            return value
    return None


def infer_dataset_from_question(question: str, datasets: dict[str, Any]) -> str | None:
    q = question.lower()
    for name in sorted(datasets, key=len, reverse=True):
        if name.lower() in q:
            return name
    return None


def dataset_profile(args: argparse.Namespace, kb: dict[str, Any]) -> dict[str, Any]:
    datasets = kb.get("datasets", {})
    name = args.dataset_name or infer_dataset_from_question(args.question, datasets)
    profile = dict(datasets.get(name, {})) if name else {}
    if name:
        profile["name"] = name

    overrides = {
        "downstream": args.downstream or infer_downstream(args.question),
        "n_class": args.n_class,
        "sfreq": args.sfreq,
        "channel": args.channel,
        "seq_len": args.seq_len,
    }
    for key, value in overrides.items():
        if value is not None:
            profile[key] = value
    profile["keywords"] = [
        word for word in DOWNSTREAM_ALIASES
        if word in args.question.lower()
    ]
    return profile


def choose_metric(row: dict[str, Any]) -> tuple[str, float] | None:
    metrics = row.get("metrics", {})
    for metric in METRIC_PRIORITY:
        if metric in metrics:
            return metric, float(metrics[metric])
    if metrics:
        key = next(iter(metrics))
        return key, float(metrics[key])
    return None


def similarity_score(
    target: dict[str, Any],
    candidate_info: dict[str, Any],
    candidate_name: str | None = None,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    candidate_text = (candidate_name or "").lower()
    keyword_hits = [kw for kw in target.get("keywords", []) if kw in candidate_text]
    if keyword_hits:
        score += 2.5
        reasons.append(f"dataset keyword match: {', '.join(keyword_hits)}")
    if target.get("downstream") and target.get("downstream") == candidate_info.get("downstream"):
        score += 3.0
        reasons.append(f"same downstream={target['downstream']}")
    if target.get("n_class") and target.get("n_class") == candidate_info.get("n_class"):
        score += 1.4
        reasons.append(f"same n_class={target['n_class']}")
    for key, weight, scale in [("sfreq", 1.0, 500), ("channel", 1.0, 64), ("seq_len", 0.8, 30)]:
        if target.get(key) is None or candidate_info.get(key) is None:
            continue
        diff = abs(float(target[key]) - float(candidate_info[key]))
        partial = max(0.0, weight * (1.0 - diff / scale))
        score += partial
        if partial > weight * 0.65:
            reasons.append(f"similar {key}: {candidate_info[key]}")
    return score, reasons


def rank_models(kb: dict[str, Any], target: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    datasets = kb.get("datasets", {})
    for row in kb.get("results", []):
        metric = choose_metric(row)
        if not metric:
            continue
        metric_name, metric_value = metric
        dataset_name = row.get("Dataset")
        info = row.get("dataset_info") or lookup_dataset_info(dataset_name, datasets)
        sim, reasons = similarity_score(target, info, dataset_name)
        if sim <= 0 and target:
            continue
        score = metric_value * (1.0 + sim / 6.0)
        model = row.get("Model Name")
        item = buckets.setdefault(model, {"model": model, "score": 0.0, "evidence": []})
        item["evidence"].append({
            "dataset": dataset_name,
            "metric": metric_name,
            "value": metric_value,
            "similarity": round(sim, 3),
            "weighted_score": round(score, 3),
            "reasons": reasons,
        })

    for item in buckets.values():
        item["evidence"] = sorted(
            item["evidence"],
            key=lambda x: (x["weighted_score"], x["similarity"], x["value"]),
            reverse=True,
        )
        top_evidence = item["evidence"][:3]
        item["score"] = round(
            sum(ev["weighted_score"] for ev in top_evidence) / max(len(top_evidence), 1),
            3,
        )
        item["evidence"] = top_evidence

    ranked = sorted(buckets.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def relevant_notes(kb: dict[str, Any], question: str, limit: int = 3) -> list[dict[str, str]]:
    words = {w for w in re.findall(r"[A-Za-z0-9_\-]+", question.lower()) if len(w) >= 3}
    scored = []
    for note in kb.get("manual_notes", []):
        text = note.get("text", "")
        score = sum(1 for word in words if word in text.lower())
        if score:
            scored.append((score, note))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"path": n["path"], "snippet": n["text"][:500]} for _, n in scored[:limit]]


def answer(args: argparse.Namespace) -> None:
    kb = load_kb(Path(args.kb))
    target = dataset_profile(args, kb)
    ranked = rank_models(kb, target, args.top_k)
    notes = relevant_notes(kb, args.question)

    response = {
        "question": args.question,
        "target_dataset_profile": target,
        "recommendations": ranked,
        "relevant_notes": notes,
        "missing_information": missing_target_fields(target),
        "method": (
            "Rule-based benchmark agent: match the new dataset profile to existing benchmark datasets, "
            "weight model performance by task/data similarity, then attach evidence rows."
        ),
    }
    if args.json:
        print(json.dumps(response, indent=2, ensure_ascii=False))
    else:
        print_human_answer(response)


def missing_target_fields(target: dict[str, Any]) -> list[str]:
    return [key for key in ["downstream", "n_class", "sfreq", "channel", "seq_len"] if target.get(key) is None]


def print_human_answer(response: dict[str, Any]) -> None:
    target = response["target_dataset_profile"]
    print("Benchmark Agent Answer")
    print("======================")
    print(f"Question: {response['question']}")
    print(f"Target profile: {json.dumps(target, ensure_ascii=False)}")
    if response["missing_information"]:
        print(f"Missing fields to improve the answer: {', '.join(response['missing_information'])}")
    print("")

    recommendations = response["recommendations"]
    if not recommendations:
        print("No recommendation could be scored. Add downstream/n_class/sfreq/channel/seq_len or more knowledge notes.")
        return

    print("Recommended models:")
    for idx, item in enumerate(recommendations, start=1):
        print(f"{idx}. {item['model']} (score={item['score']})")
        for ev in item["evidence"]:
            reasons = "; ".join(ev["reasons"]) if ev["reasons"] else "general benchmark performance"
            print(f"   - {ev['dataset']}: {ev['metric']}={ev['value']} ({reasons})")

    if response["relevant_notes"]:
        print("\nRelevant manual notes:")
        for note in response["relevant_notes"]:
            print(f"- {note['path']}: {note['snippet'].replace(chr(10), ' ')[:180]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Knowledge-base benchmark agent for Brain4FMs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-kb", help="build/update the local benchmark knowledge base")
    build.add_argument("--readme", default=str(ROOT / "README.md"))
    build.add_argument("--paper", default=str(ROOT.parent / "f7_icml2026_benchmark.pdf"))
    build.add_argument("--paper-chars", type=int, default=30000)
    build.add_argument("--knowledge-dir", default=str(DEFAULT_KNOWLEDGE_DIR))
    build.add_argument("--output")
    build.set_defaults(func=build_kb)

    note = subparsers.add_parser("add-note", help="add a manual knowledge note")
    note.add_argument("--title", required=True)
    note.add_argument("--text", required=True)
    note.add_argument("--tag", action="append")
    note.add_argument("--knowledge-dir", default=str(DEFAULT_KNOWLEDGE_DIR))
    note.add_argument("--overwrite", action="store_true")
    note.set_defaults(func=add_note)

    ask = subparsers.add_parser("ask", help="ask the benchmark agent for a recommendation")
    ask.add_argument("question")
    ask.add_argument("--kb", default=str(DEFAULT_KB_PATH))
    ask.add_argument("--dataset-name")
    ask.add_argument("--downstream")
    ask.add_argument("--n-class", type=int)
    ask.add_argument("--sfreq", type=float)
    ask.add_argument("--channel", type=int)
    ask.add_argument("--seq-len", type=float)
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(func=answer)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
