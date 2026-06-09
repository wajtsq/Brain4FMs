---
name: add-benchmark-model
description: Use when adding or updating a Brain4FMs benchmark model from a GitHub repository and checkpoint URL, including update_skill/model_updater.py automation, model/model_config.py checkpoint registration, optional checkpoint download into /benchmark/pretrained_weights/<ModelName>/, model/<ModelName>/ integration, datasets/<ModelName>_dataset.py when needed, and utils/meta_info.py registration.
---

# Add Benchmark Model

Use this skill when the task is to add a new benchmark model or fix the model-adding workflow itself.

The default goal is a working benchmark integration, not a scaffold. Do not stop at a compile-only placeholder when the request is to add a paper model.

## Inputs You Must Collect

Before writing code, make sure you have all of these:

- `model_name`
- `github_url`
- `checkpoint_url`
- `final_dim` if it can be identified reliably
- Whether the model can use `DefaultDataset` or needs a custom dataset wrapper

Do not start the final integration flow without both `github_url` and `checkpoint_url`.
If checkpoint download is blocked, continue the code integration and tell the user exactly which file path must be populated before weight loading can work.

## Files To Read First

- `update_skill/model_updater.py`
- `update_skill/benchmark_updater.py`
- `model/model_config.py`
- `utils/meta_info.py`
- `datasets/default_dataset.py`
- Two similar model integrations under `model/*/*.py`
- The target repository README, main model file, config file, and checkpoint-loading code

If the task is about improving the workflow itself, read the current command parser and any helper functions before editing.

## Standard Workflow

1. Inspect the source repository.
   - Find the actual runnable model class, not only a base class.
   - Find the true input structure.
   - Find the real encoder output dimension used by the benchmark-facing head.
   - Find how checkpoints are loaded and whether one file or multiple files are needed.
   - By default, if the source model already has a usable task head, keep and use that head.
   - Use the benchmark's external `clsf` only when the source model has no suitable head for the benchmark task or when the source head cannot be reused safely.
   - Determine whether extra source files must be vendored into `model/<ModelName>/` for the model to run.

2. Use or improve `update_skill/model_updater.py`.
   - The `add` flow must accept `--name`, `--github-url`, and `--checkpoint-url`.
   - The script should update `model/model_config.py` automatically.
   - The script should support downloading the checkpoint into `/benchmark/pretrained_weights/<ModelName>/`.
   - Prefer deterministic helpers in `model_updater.py` over manual editing instructions in final notes.

3. Register checkpoint paths in `model/model_config.py`.
   - Add or update a `ModelPathArgs.<ModelName>_path` field.
   - The stored path should point to the downloaded file, not only the directory, unless the model truly loads a folder.
   - Keep the naming consistent with existing entries such as `LaBraM_path` or `BIOT_path`.

4. Download checkpoints when requested by the flow.
   - Target directory: `/benchmark/pretrained_weights/<ModelName>/`
   - Preserve the filename from the checkpoint URL when practical.
   - If the environment blocks network access, retry with escalation when allowed.
   - If the filesystem blocks writes to `/data`, explain that clearly and stop before pretending the download succeeded.

5. Create or update the benchmark model integration.
   - Add `model/<ModelName>/`.
   - Add `model/<ModelName>/<ModelName>.py`.
   - Define `class <ModelName>_Trainer`.
   - Define `class <ModelName>(nn.Module)`.
   - Keep model-specific reshape, padding, tokenization, or metadata handling inside the model wrapper or custom dataset.
   - Reproduce the actual source model behavior needed for benchmark inference or fine-tuning.
   - Vendor or port the minimal required source files instead of replacing them with placeholder layers.
   - Use `raise NotImplementedError` only when the source repository is genuinely blocked by a missing artifact or undocumented dependency and you cannot proceed safely.

6. Decide on dataset strategy.
   - Use `DefaultDataset` only when the source model can be adapted from `x: (seq_num, ch_num, N)` and `y: (seq_num,)`.
   - Create `datasets/<ModelName>_dataset.py` when the model needs segments, per-patient structure, channel metadata, masks, token IDs, prompts, or any extra tensors.

7. Register the new model.
   - Update `utils/meta_info.py`.
   - Ensure imports, `dataset_class_dict`, `trainer_dict`, and `model_dict` are all updated.

## Command Expectations

When editing `update_skill/model_updater.py`, keep these expectations true:

- `python update_skill/model_updater.py add --name <ModelName> --github-url <repo> --checkpoint-url <url> ...`
- `python update_skill/model_updater.py ai-add --proposal <json> --checkpoint-url <url> ...`
- Optional download flag:
  - `--download-checkpoint`

Recommended behavior:

- `add` requires `--checkpoint-url`
- `ai-add` must receive `checkpoint_url` either from the proposal JSON or the CLI
- The script should print the resolved `ModelPathArgs` field name and destination path

## Implementation Rules

- Keep folder name, class name, CLI name, and `meta_info.py` key identical.
- Prefer automatic edits in code over “remember to manually edit model_config.py”.
- Do not claim a model uses `DefaultDataset` unless the data interface truly matches.
- Do not silently skip checkpoint registration.
- Do not hardcode secrets or API keys.
- Do not ship placeholder implementations that only compile.
- If a model is incomplete because the checkpoint or a source-only dependency is missing, fail clearly or leave an explicit runtime error that names the missing artifact and expected path.

## Validation

At minimum, verify:

- `python -m py_compile update_skill/model_updater.py`
- `python -m py_compile update_skill/benchmark_updater.py`
- `python -m py_compile model/model_config.py`
- If model integration was added, also validate the model file and any dataset file

If a real checkpoint download was not run because of sandbox or network restrictions, say that explicitly in the final answer.
Validation should also try a minimal import or forward smoke test in an environment that has the model dependencies, when available.

## Final Answer Requirements

When you finish a model-adding task, report:

- Which commands were changed
- Whether `model/model_config.py` was updated automatically
- The expected checkpoint destination path
- Whether the checkpoint was actually downloaded or only wired in code
- Any remaining manual work, such as multi-file checkpoints or source-only dependencies
- Whether the added model is a real integration or still blocked by a named missing artifact
