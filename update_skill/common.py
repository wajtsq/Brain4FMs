from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPDATE_SKILL_DIR = ROOT / "update_skill"
DATA_PREPROCESS_DIR = ROOT / "data_preprocess"
DATASETS_DIR = ROOT / "datasets"
MODEL_DIR = ROOT / "model"
UTILS_DIR = ROOT / "utils"


def safe_identifier(name: str) -> str:
    value = re.sub(r"\W+", "_", name.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        raise ValueError("name cannot be empty")
    if value[0].isdigit():
        value = f"_{value}"
    return value


def preprocess_dir_name(dataset_name: str) -> str:
    return f"{safe_identifier(dataset_name).lower()}_preprocess"


def quote(value: str) -> str:
    return repr(value)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_new(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def find_dict_end(text: str, dict_name: str) -> int:
    match = re.search(rf"^{dict_name}\s*=\s*{{", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find dict {dict_name}")
    open_brace = text.find("{", match.start())
    depth = 0
    in_string: str | None = None
    escaped = False
    for i in range(open_brace, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_string:
                in_string = None
            continue
        if ch in {"'", '"'}:
            in_string = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"Could not find end of dict {dict_name}")


def insert_dict_entry(path: Path, dict_name: str, key: str, value: str) -> bool:
    text = read_text(path)
    start_match = re.search(rf"^{dict_name}\s*=\s*{{", text, flags=re.MULTILINE)
    if not start_match:
        raise ValueError(f"Could not find dict {dict_name}")
    end = find_dict_end(text, dict_name)
    dict_body = text[start_match.start() : end]
    if re.search(rf"{re.escape(quote(key))}\s*:", dict_body):
        return False
    entry = f"    {quote(key)}: {value},\n"
    path.write_text(text[:end] + entry + text[end:], encoding="utf-8")
    return True


def ensure_import(path: Path, import_line: str) -> bool:
    text = read_text(path)
    if import_line in text:
        return False
    marker = "from utils.metrics import BinaryClassMetrics, MultiClassMetrics\n"
    if marker in text:
        text = text.replace(marker, f"{import_line}\n{marker}")
    else:
        lines = text.splitlines(keepends=True)
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                insert_at = i + 1
        lines.insert(insert_at, f"{import_line}\n")
        text = "".join(lines)
    path.write_text(text, encoding="utf-8")
    return True


def dict_has_key(path: Path, dict_name: str, key: str) -> bool:
    text = read_text(path)
    start_match = re.search(rf"^{dict_name}\s*=\s*{{", text, flags=re.MULTILINE)
    if not start_match:
        return False
    end = find_dict_end(text, dict_name)
    body = text[start_match.start() : end]
    return re.search(rf"{re.escape(quote(key))}\s*:", body) is not None


def default_split(group_num: int) -> list[int]:
    if group_num == 6:
        return [3, 1, 2]
    if group_num == 4:
        return [2, 1, 1]
    return [max(group_num - 2, 1), 1, 1]


def parse_split(raw: str | None, group_num: int) -> list[int]:
    if raw is None:
        return default_split(group_num)
    split = [int(x.strip()) for x in raw.split(",")]
    if len(split) != 3 or sum(split) != group_num:
        raise ValueError("--split must contain three comma-separated ints that sum to --group-num")
    return split


def prompt_text(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{label}{suffix}: ").strip()
    if value == "" and default is not None:
        return default
    return value


def prompt_int(label: str, default: int | None = None) -> int:
    return int(prompt_text(label, None if default is None else str(default)))


def prompt_float(label: str, default: float | None = None) -> float:
    return float(prompt_text(label, None if default is None else str(default)))


def prompt_bool(label: str, default: bool = False) -> bool:
    default_text = "y" if default else "n"
    value = prompt_text(f"{label} (y/n)", default_text).lower()
    return value in {"y", "yes", "1", "true"}


def py_compile(path: Path) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output

