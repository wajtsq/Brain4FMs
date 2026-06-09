from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import dataset_updater
import benchmark_agent
import model_updater


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Unified updater entry for Brain4FMs datasets, models, and benchmark knowledge agent."
    )
    parser.add_argument("target", choices=["dataset", "model", "agent"], help="what to update")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)

    forwarded = parsed.args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    # Backward compatibility with the old interface:
    # benchmark_updater.py dataset --name ... -> dataset_updater.py add --name ...
    old_dataset_flags = {"--name", "--source-url", "--sfreq", "--channel", "--seq-len", "--n-class"}
    old_model_flags = {"--name", "--github-url", "--checkpoint-url", "--final-dim", "--custom-dataset", "--clone", "--download-checkpoint"}
    if parsed.target == "agent":
        benchmark_agent.main(forwarded)
    elif parsed.target == "dataset":
        if not forwarded or forwarded[0].startswith("-"):
            if any(flag in forwarded for flag in old_dataset_flags):
                forwarded = ["add"] + forwarded
        dataset_updater.main(forwarded)
    else:
        if not forwarded or forwarded[0].startswith("-"):
            if any(flag in forwarded for flag in old_model_flags):
                forwarded = ["add"] + forwarded
        model_updater.main(forwarded)


if __name__ == "__main__":
    main()
