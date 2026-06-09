from __future__ import annotations

import json
import os
os.environ["OPENAI_API_KEY"] = "sk-xxxx"
# proxy = "http://127.0.0.1:xx"

# os.environ["http_proxy"] = proxy
# os.environ["https_proxy"] = proxy
# os.environ["HTTP_PROXY"] = proxy
# os.environ["HTTPS_PROXY"] = proxy
import re
import urllib.request
from pathlib import Path
from typing import Any


TEXT_EXTS = {
    ".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".csv", ".tsv",
    ".py", ".m", ".matinfo", ".edfinfo",
}


def read_text_file(path: Path, limit: int = 12000) -> str:
    try:
        if path.suffix.lower() == ".pdf":
            return _read_pdf_text(path, limit)
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text[:limit]
    except Exception as exc:
        return f"[Could not read {path}: {exc}]"


def _read_pdf_text(path: Path, limit: int) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader
        except Exception as exc:
            return f"[PDF text extraction unavailable for {path}: {exc}]"

    reader = PdfReader(str(path))
    chunks = []
    for page in reader.pages[:20]:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            continue
        if sum(len(c) for c in chunks) >= limit:
            break
    return "\n".join(chunks)[:limit]


def collect_context(paths: list[str] | None, limit_per_file: int = 12000) -> str:
    if not paths:
        return ""
    chunks = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files = [
                p for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in TEXT_EXTS
            ][:30]
        else:
            files = [path]
        for file_path in files:
            chunks.append(f"\n### FILE: {file_path}\n{read_text_file(file_path, limit_per_file)}")
    return "\n".join(chunks)


def load_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {"error": f"report not found: {path}"}
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def openai_json(prompt: str, model: str | None = None) -> dict[str, Any]:
    """Call an OpenAI-compatible Responses API and parse JSON from the output.

    This function is intentionally tiny and optional. It needs only
    OPENAI_API_KEY, and falls back cleanly when no key is configured.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"available": False, "error": "OPENAI_API_KEY is not set"}

    endpoint = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = model or os.environ.get("BRAIN4FMS_AI_MODEL", "gpt-4.1-mini")
    payload = {
        "model": model,
        "input": prompt,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{endpoint}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"available": False, "error": f"AI request failed: {exc}"}

    text = _extract_response_text(raw)
    parsed = _parse_json_object(text)
    if parsed is None:
        return {"available": True, "error": "AI output was not valid JSON", "raw_text": text}
    parsed["_ai_provider"] = {"model": model, "endpoint": endpoint}
    return parsed


def _extract_response_text(raw: dict[str, Any]) -> str:
    if isinstance(raw.get("output_text"), str):
        return raw["output_text"]
    chunks = []
    for item in raw.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                chunks.append(content.get("text", ""))
    return "\n".join(chunks)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def dataset_prompt(name: str | None, source_url: str | None, context: str, report: dict[str, Any]) -> str:
    return f"""You are helping extend the Brain4FMs EEG/iEEG benchmark.

Return ONLY valid JSON. Do not use markdown.

Infer a dataset extension proposal from the dataset paper, README, metadata,
and local inspection report. Be conservative: unknown fields must be null and
must be listed in missing_information.

Expected JSON keys:
{{
  "dataset_name": string|null,
  "source_url": string|null,
  "modality": "EEG"|"iEEG"|null,
  "raw_sfreq": number|null,
  "target_sfreq": integer|null,
  "channel": integer|null,
  "seq_len": integer|null,
  "n_class": integer|null,
  "downstream": string|null,
  "filtering": {{"high_pass": number|null, "low_pass": number|null, "notch": number|null}},
  "label_mapping": object,
  "channel_selection_notes": string,
  "event_alignment_notes": string,
  "implementation_plan": [string],
  "missing_information": [string],
  "confidence": number
}}

Benchmark policy:
- final data shape must be (N, C, T)
- T = target_sfreq * seq_len
- EEG target_sfreq should be <= 500
- iEEG target_sfreq should be <= 1000
- if no paper filter is specified, use 0.01 Hz to target_sfreq / 3
- select only EEG/iEEG channels

User supplied name: {name}
Source URL: {source_url}

Inspection report JSON:
{json.dumps(report, ensure_ascii=False)[:18000]}

Context:
{context[:36000]}
"""


def model_prompt(name: str | None, github_url: str | None, context: str, report: dict[str, Any]) -> str:
    return f"""You are helping extend the Brain4FMs EEG/iEEG benchmark.

Return ONLY valid JSON. Do not use markdown.

Infer a model extension proposal from GitHub README/code summaries and local
inspection report. Be conservative: unknown fields must be null and must be
listed in missing_information.

Expected JSON keys:
{{
  "model_name": string|null,
  "github_url": string|null,
  "checkpoint_url": string|null,
  "main_module_class": string|null,
  "final_dim": integer|null,
  "input_shape": string|null,
  "use_source_head": boolean|null,
  "source_head_reason": string|null,
  "head_decision_evidence": [string],
  "uses_default_dataset": boolean|null,
  "custom_dataset_needed": boolean|null,
  "custom_dataset_reason": string|null,
  "dataset_decision_evidence": [string],
  "checkpoint_notes": string,
  "required_files": [string],
  "forward_adaptation_plan": [string],
  "implementation_plan": [string],
  "missing_information": [string],
  "confidence": number
}}

Brain4FMs model interface:
- model folder: temp_private/model/<ModelName>/
- class <ModelName>_Trainer with set_config, clsf_loss_func, optimizer, scheduler
- class <ModelName>(nn.Module)
- static forward_propagate(args, data_packet, model, clsf, loss_func=None)
- DefaultDataset returns x: (seq_num, ch_num, N), y: (seq_num,)

Important:
- The target output is a working integration, not a compile-only scaffold.
- Do not suggest placeholder encoder blocks such as LazyLinear stubs as a final implementation.
- If the model cannot be completed because a checkpoint or required source file is missing, say so explicitly in missing_information and implementation_plan.
- By default, if the source model already has a usable task head, keep and use that head.
- Use the benchmark's external clsf only when the source model has no suitable head for the benchmark task or when the source head cannot be reused safely.

Dataset decision rule:
- custom_dataset_needed=false only if all required model inputs can be derived
  from x=(batch, channel, time) and y by simple reshape/crop/pad/FFT inside
  forward_propagate.
- custom_dataset_needed=true if the model requires channel names, channel
  positions, masks, token ids, prompts, subject/session metadata, graph edges,
  precomputed spectra, or any extra tensor that cannot be derived from x/y.
- Always include evidence for the decision.

User supplied name: {name}
GitHub URL: {github_url}

Inspection report JSON:
{json.dumps(report, ensure_ascii=False)[:22000]}

Context:
{context[:36000]}
"""


def heuristic_dataset_proposal(
    name: str | None,
    source_url: str | None,
    context: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    text = context.lower()
    suggestions = report.get("suggestions", {}) if isinstance(report, dict) else {}
    modality = suggestions.get("modality")
    if not modality:
        modality = "iEEG" if any(k in text for k in ["seeg", "ecog", "intracranial", "ieeg"]) else "EEG"

    raw_sfreq = suggestions.get("raw_sfreq") or _config_number(context, "raw_sfreq")
    target_sfreq = suggestions.get("target_sfreq") or _config_number(context, "sfreq")
    if target_sfreq is None and raw_sfreq:
        target_sfreq = min(int(raw_sfreq), 1000 if modality == "iEEG" else 500)
    channel = suggestions.get("channel") or _config_number(context, "channel")
    n_class = _config_number(context, "n_class") or _guess_n_class(text)
    seq_len = _config_number(context, "seq_len") or _guess_seq_len(text)

    missing = []
    for key, value in [
        ("raw_sfreq", raw_sfreq),
        ("channel", channel),
        ("seq_len", seq_len),
        ("n_class", n_class),
    ]:
        if value is None:
            missing.append(key)

    high_pass = _first_number_after(text, ["high-pass", "high pass", "band-pass", "bandpass"])
    if high_pass is None:
        high_pass = 0.01
    low_pass = None
    if target_sfreq:
        low_pass = target_sfreq / 3
    notch = 50 if "50 hz" in text or "50hz" in text else (60 if "60 hz" in text or "60hz" in text else 50)

    return {
        "dataset_name": name,
        "source_url": source_url,
        "modality": modality,
        "raw_sfreq": raw_sfreq,
        "target_sfreq": target_sfreq,
        "channel": channel,
        "seq_len": seq_len,
        "n_class": n_class,
        "downstream": _guess_downstream(text),
        "filtering": {"high_pass": high_pass, "low_pass": low_pass, "notch": notch},
        "label_mapping": {},
        "channel_selection_notes": "Select EEG/iEEG channels only; confirm against channels.tsv or raw headers.",
        "event_alignment_notes": "Confirm event/label alignment from dataset-specific annotation files.",
        "implementation_plan": [
            "Run dataset inspect on the raw folder.",
            "Fill iter_labeled_recordings(args) using the dataset annotation files.",
            "Generate subject_data, then group_data.",
            "Run validate with --group-path.",
        ],
        "missing_information": missing,
        "confidence": 0.35 if missing else 0.65,
        "_assistant_mode": "heuristic",
    }


def heuristic_model_proposal(
    name: str | None,
    github_url: str | None,
    context: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    suggestions = report.get("suggestions", {}) if isinstance(report, dict) else {}
    classes = report.get("nn_module_classes", []) if isinstance(report, dict) else []
    main_class = classes[0]["class"] if classes else None
    final_dim = suggestions.get("possible_final_dim")
    dataset_decision = decide_model_dataset_need(context, report)
    head_decision = decide_model_head_usage(context, report)
    custom_needed = dataset_decision["custom_dataset_needed"]
    missing = []
    if not main_class:
        missing.append("main_module_class")
    if final_dim is None:
        missing.append("final_dim")
    return {
        "model_name": name,
        "github_url": github_url,
        "checkpoint_url": None,
        "main_module_class": main_class,
        "final_dim": final_dim,
        "input_shape": dataset_decision["input_shape"],
        "use_source_head": head_decision["use_source_head"],
        "source_head_reason": head_decision["reason"],
        "head_decision_evidence": head_decision["evidence"],
        "uses_default_dataset": not custom_needed,
        "custom_dataset_needed": custom_needed,
        "custom_dataset_reason": dataset_decision["reason"],
        "dataset_decision_evidence": dataset_decision["evidence"],
        "checkpoint_notes": "Inspect checkpoint loading code and add ModelPathArgs only if needed.",
        "required_files": [],
        "forward_adaptation_plan": [
            "Adapt DefaultDataset x=(batch, channel, time) inside forward_propagate when possible.",
            "Create a custom dataset only if the model requires metadata beyond x and y.",
        ],
        "implementation_plan": [
            "Copy or port the minimal real source files needed for a working integration.",
            "Implement the actual forward path and checkpoint loading logic.",
            "Run validate and a minimal import or forward smoke test.",
        ],
        "missing_information": missing,
        "confidence": 0.35 if missing else 0.6,
        "_assistant_mode": "heuristic",
    }


def decide_model_head_usage(context: str, report: dict[str, Any]) -> dict[str, Any]:
    """Decide whether to use the source model's own head or benchmark external clsf."""
    blob = (context + "\n" + json.dumps(report, ensure_ascii=False)).lower()
    evidence: list[str] = []

    source_head_markers = [
        "classification head",
        "classifier head",
        "head_model",
        "num_classes",
        "n_classes",
        "class_path: models.",
        "mvpformerhead",
        "classifier(",
        "self.head",
        "final_layer",
        "prediction head",
        "binaryaccuracy",
        "crossentropyloss",
    ]
    external_head_blockers = [
        "return_patch_tokens",
        "return_output=true",
        "feature extractor",
        "encoder only",
        "backbone only",
        "pretrain only",
        "without classification head",
        "linear probe",
    ]

    source_hits = [m for m in source_head_markers if _contains_marker(blob, m)]
    blocker_hits = [m for m in external_head_blockers if _contains_marker(blob, m)]

    if source_hits:
        evidence.extend([f"source-head-marker: {m}" for m in source_hits[:6]])
        return {
            "use_source_head": True,
            "reason": "Source repository appears to expose a usable task head; prefer reusing it by default.",
            "evidence": evidence,
        }

    if blocker_hits:
        evidence.extend([f"external-clsf-marker: {m}" for m in blocker_hits[:6]])
        return {
            "use_source_head": False,
            "reason": "Repository appears to expose a backbone or feature path without a clearly reusable task head.",
            "evidence": evidence,
        }

    evidence.append("default-policy: prefer source head when present")
    return {
        "use_source_head": True,
        "reason": "Default policy prefers the source model head unless inspection proves only a backbone is available.",
        "evidence": evidence,
    }


def merge_proposals(base: dict[str, Any], ai: dict[str, Any]) -> dict[str, Any]:
    if not ai or ai.get("available") is False or ai.get("error"):
        base["_ai_error"] = ai.get("error") if isinstance(ai, dict) else "AI unavailable"
        return base
    merged = dict(base)
    for key, value in ai.items():
        if key.startswith("_"):
            merged[key] = value
        elif value not in (None, "", [], {}):
            merged[key] = value
    merged["_assistant_mode"] = "ai+heuristic"
    return merged


def decide_model_dataset_need(context: str, report: dict[str, Any]) -> dict[str, Any]:
    """Decide whether DefaultDataset is enough for a new model.

    DefaultDataset returns only (x, y), where x has shape (batch, channel, time).
    A custom dataset is needed when the model appears to require information
    that cannot be derived from x by a small reshape/crop/FFT in forward_propagate.
    """
    blob = (context + "\n" + json.dumps(report, ensure_ascii=False)).lower()
    evidence: list[str] = []

    metadata_markers = [
        "channel position", "channel_positions", "channel_pos", "channel ids",
        "channel_id", "montage", "electrode", "sensor location", "spatial coordinate",
        "positional coordinate", "leadfield",
    ]
    token_markers = [
        "input_ids", "attention_mask", "token_type_ids", "tokenizer", "prompt",
        "text prompt", "labels_mask",
    ]
    external_feature_markers = [
        "precomputed spectrogram", "precomputed stft", "precomputed psd",
        "connectivity matrix", "adjacency matrix", "graph", "edge_index",
    ]
    subject_markers = [
        "subject id", "subject_id", "session id", "session_id", "demographic",
        "metadata",
    ]

    marker_groups = [
        ("channel metadata", metadata_markers),
        ("token/prompt inputs", token_markers),
        ("precomputed external features", external_feature_markers),
        ("subject/session metadata", subject_markers),
    ]
    for label, markers in marker_groups:
        hits = [m for m in markers if _contains_marker(blob, m)]
        if hits:
            evidence.append(f"{label}: {', '.join(hits[:5])}")

    weak_feature_hits = [
        marker for marker in ["spectrogram", "stft", "power spectral", "band power", "psd", "fft"]
        if _contains_marker(blob, marker)
    ]

    forward_signatures = report.get("forward_signatures", []) if isinstance(report, dict) else []
    suspicious_args: list[str] = []
    for item in forward_signatures:
        signature = item.get("signature", "") if isinstance(item, dict) else ""
        args = _parse_signature_args(signature)
        required = [
            arg for arg in args
            if arg not in {"self", "x", "input", "inputs", "data"}
            and "=" not in arg
            and not arg.startswith("*")
        ]
        if required:
            suspicious_args.append(f"{signature} requires {required}")
    if suspicious_args:
        evidence.append("forward requires extra non-default args: " + "; ".join(suspicious_args[:5]))

    input_shape = _infer_input_shape(blob, report)
    simple_shape = _looks_like_default_compatible(input_shape, blob)

    if evidence:
        return {
            "custom_dataset_needed": True,
            "reason": "Model appears to require metadata or extra tensors beyond DefaultDataset (x, y).",
            "evidence": evidence,
            "input_shape": input_shape,
        }
    if simple_shape:
        simple_evidence = ["no required metadata/extra tensor markers found"]
        if weak_feature_hits:
            simple_evidence.append(
                "spectral/time-frequency markers found, but these can usually be computed from x inside forward_propagate"
            )
        return {
            "custom_dataset_needed": False,
            "reason": "Detected input appears compatible with DefaultDataset x=(batch, channel, time), possibly with reshape in forward_propagate.",
            "evidence": simple_evidence,
            "input_shape": input_shape,
        }
    if weak_feature_hits and not evidence:
        return {
            "custom_dataset_needed": False,
            "reason": "Only derived signal features were detected. Keep DefaultDataset and compute them from x in forward_propagate unless the repo explicitly requires precomputed tensors.",
            "evidence": [f"derived feature markers: {', '.join(weak_feature_hits[:5])}"],
            "input_shape": input_shape,
        }
    return {
        "custom_dataset_needed": False,
        "reason": "No hard evidence for a custom dataset. Keep DefaultDataset, then verify forward_propagate with a synthetic batch.",
        "evidence": ["input requirement unclear; validator/review should confirm"],
        "input_shape": input_shape,
    }


def _parse_signature_args(signature: str) -> list[str]:
    args = []
    for raw in signature.split(","):
        arg = raw.strip()
        if not arg:
            continue
        arg = arg.split(":", 1)[0].strip()
        args.append(arg)
    return args


def _infer_input_shape(blob: str, report: dict[str, Any]) -> str | None:
    shape_hints = report.get("input_shape_hints", []) if isinstance(report, dict) else []
    for item in shape_hints:
        line = item.get("line", "") if isinstance(item, dict) else ""
        if line:
            return line[:240]

    patterns = [
        r"\((?:batch|b|bsz)[,\s]+(?:channel|channels|c|nvars|variables)[,\s]+(?:time|seq|length|t)\)",
        r"\((?:batch|b|bsz)[,\s]+(?:time|seq|length|t)[,\s]+(?:channel|channels|c|nvars|variables)\)",
        r"\[(?:batch|b|bsz)[,\s]+(?:channel|channels|c|nvars|variables)[,\s]+(?:time|seq|length|t)\]",
        r"\[(?:batch|b|bsz)[,\s]+(?:time|seq|length|t)[,\s]+(?:channel|channels|c|nvars|variables)\]",
    ]
    for pattern in patterns:
        match = re.search(pattern, blob)
        if match:
            return match.group(0)
    return None


def _looks_like_default_compatible(input_shape: str | None, blob: str) -> bool:
    if input_shape:
        text = input_shape.lower()
        has_batch = any(k in text for k in ["batch", "bsz", " b", "(b", "[b"])
        has_channel = any(k in text for k in ["channel", "channels", "nvars", "variables", " c", ",c"])
        has_time = any(k in text for k in ["time", "seq", "length", " t", ",t"])
        if has_batch and has_channel and has_time:
            return True
    return any(k in blob for k in [
        "batch, channel, time",
        "batch, channels, time",
        "batch, nvars, seq",
        "batch_size, nvars, seq_len",
        "b, c, t",
        "b,c,t",
    ])


def _contains_marker(text: str, marker: str) -> bool:
    if re.search(r"[a-zA-Z0-9_]", marker):
        pattern = r"(?<![a-zA-Z0-9_])" + re.escape(marker) + r"(?![a-zA-Z0-9_])"
        return re.search(pattern, text) is not None
    return marker in text


def _guess_n_class(text: str) -> int | None:
    for pat in [r"(\d+)[-\s]?class", r"(\d+)\s+classes", r"(\d+)\s+categories"]:
        match = re.search(pat, text)
        if match:
            return int(match.group(1))
    if "binary" in text or "control" in text and "patient" in text:
        return 2
    return None


def _guess_seq_len(text: str) -> int | None:
    for pat in [r"(\d+)\s*s(?:ec|econd)?\s*(?:epoch|window|segment)", r"epoch[s]?\s+of\s+(\d+)\s*s"]:
        match = re.search(pat, text)
        if match:
            return int(match.group(1))
    return None


def _guess_downstream(text: str) -> str:
    if any(k in text for k in ["emotion", "valence", "arousal"]):
        return "emotion"
    if any(k in text for k in ["motor imagery", "motor imagine", "bci"]):
        return "Motor imagine"
    if any(k in text for k in ["sleep", "stage"]):
        return "disorder"
    return "disorder"


def _first_number_after(text: str, markers: list[str]) -> float | None:
    for marker in markers:
        idx = text.find(marker)
        if idx >= 0:
            match = re.search(r"(\d+(?:\.\d+)?)", text[idx: idx + 80])
            if match:
                return float(match.group(1))
    return None


def _config_number(text: str, name: str) -> int | float | None:
    match = re.search(rf"\b{name}\s*(?::\s*[\w.\[\]]+)?\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return None
    value = float(match.group(1))
    return int(value) if value.is_integer() else value


def print_proposal_summary(proposal: dict[str, Any], title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    for key, value in proposal.items():
        if key.startswith("_"):
            continue
        if isinstance(value, (dict, list)):
            print(f"{key}: {json.dumps(value, ensure_ascii=False)}")
        else:
            print(f"{key}: {value}")
    if proposal.get("_ai_error"):
        print(f"AI note: {proposal['_ai_error']}")
