---
name: add-benchmark-dataset
description: Use when adding a new EEG/iEEG dataset to Brain4FMs, including raw data download, data_preprocess/{dataset}_preprocess implementation, group_data generation, temp_private/utils/data_info.py registration, and temp_private/utils/meta_info.py metrics registration.
---

# Add Benchmark Dataset

Use this skill when the user asks to add or update a dataset in the Brain4FMs benchmark.

The default goal is a working benchmark dataset integration, not a metadata note or partial scaffold.
When the user gives:

- the local raw-data download path
- the dataset name
- the dataset paper link or official README / dataset page link

you should continue through the workflow and generate the benchmark dataset code directly.

## Inputs You Must Collect

Before writing code, make sure you have all of these:

- `dataset_name`
- `data_root`: the local path where the raw dataset has already been downloaded
- `source_url`: the paper page, official README, or official dataset website

Optional but useful:

- direct download URL if different from the paper / README page
- target processed output path
- known task name, label set, or modality

If the user provides only `dataset_name`, `data_root`, and `source_url`, treat that as enough to begin the dataset integration flow.

## Required Context

Read these files before editing:

- `update_skill/dataset_updater.py`
- `data_preprocess/utils.py`
- One or two similar preprocess examples under `data_preprocess/*_preprocess/`
- `data_process/data_info.py`
- `utils/meta_info.py`

Prefer examples with the same modality/task as the new dataset: EEG vs iEEG, sleep staging, disorder classification, motor imagery, emotion, or concept/task labels.

## Workflow

1. Identify dataset facts from the paper, official dataset page / README, and the downloaded local files under `data_root`:
   - raw download URL or manual download instructions, if available
   - modality: EEG or iEEG
   - raw sampling rate
   - channel names/types and the EEG/iEEG channel subset
   - label space and whether the task is binary or multi-class
   - subject/session/trial structure
   - task event timing and label-to-window alignment
   - recommended filtering from the paper
   - desired sample duration: `seq_len` seconds per label
   - whether the local raw-data folder already contains all files needed to implement preprocessing

2. Use the user-provided `data_root` as the source of truth for raw data.
   - Do not re-ask for raw-data location if the user has already provided it.
   - Do not commit raw data.
   - Put processed output under `data_save_dir=/datasets/<DATASET_NAME>` unless the repo uses another local convention.
   - If raw data is incomplete or requires manual extraction, document the expected directory layout clearly and continue implementing the pipeline around that layout.

3. Create `data_preprocess/<dataset_name>_preprocess/`.
   - Add `config.py` with a `PreprocessArgs` class following local examples.
   - Add `preprocess.py` that reads raw files, selects EEG/iEEG channels, filters, resamples if needed, segments, saves subject/session samples, and builds group files.
   - Add local `utils.py` only when dataset-specific grouping, label parsing, or metadata parsing is needed.
   - The generated code should be specific to the provided dataset, not a generic placeholder with an unimplemented dataset parser unless the local raw data is missing required files.

4. Preprocessing rules:
   - Final sample array must be 3D: `(N, C, T)`.
   - `T = patch_len * seq_len`.
   - `patch_len = sfreq`, where `sfreq` is the final sampling rate.
   - `seq_len` is the duration in seconds assigned to one label in the paper/protocol.
   - Downsample EEG to `<=500 Hz` unless the benchmark/paper requires lower.
   - Downsample iEEG to `<=1000 Hz` unless the benchmark/paper requires lower.
   - Use the paper's filter settings when specified.
   - If the paper gives no filter, use band-pass `0.01 Hz` to `sfreq / 3`.
   - Apply notch filtering when power-line noise is expected or already used by nearby examples.
   - Select only EEG/iEEG channels. Exclude EOG, EMG, ECG, stim/status, reference-only, annotation, and misc channels unless the task explicitly requires them.
   - Align task windows to labels/events before truncation. Never create windows that cross labels unless the paper defines that behavior.

5. Generate group data:
   - Save data files as `group_0_data.npy`, ..., `group_k_data.npy`.
   - Save labels as `group_0_label.npy`, ..., `group_k_label.npy`.
   - This cross-patient naming convention is required. Follow the existing preprocess folders and do not invent dataset-specific group filenames.
   - Each group data file shape must be `(seq_num, ch_num, seq_len * patch_len)`.
   - Default to 4-6 subject-disjoint groups when possible.
   - Keep train/validation/test split consistent with `group_num`, usually `[3, 1, 1]` for 5 groups or `[3, 1, 2]` for 6 groups.
   - If channel count varies across subjects, either reconcile channels during preprocessing or set `various_ch_num: True` only when reconciliation is impossible. The default is `False`.
   - When `various_ch_num: True`, also generate `group_<i>_pos.npy` files so the benchmark can use `default_get_data_with_pos`.

6. Update `data_process/data_info.py`.
   - Add an entry in `data_info_dict`:
     - `data_path`: `/datasets/<DATASET_NAME>/group_data` or task-specific group directory
     - `group_num`
     - `split`
     - `various_ch_num`: default `False`
     - `n_class`
     - `sfreq`
     - `channel`
     - `seq_len`
     - `downstream`
     - optional `label_path` if the dataset needs external class-name mapping

7. Update `utils/meta_info.py`.
   - Add `<DATASET_NAME>: default_get_data` in `get_data_dict` by default.
   - If patients have different channel layouts and the preprocess output includes `group_<i>_pos.npy`, register `<DATASET_NAME>: default_get_data_with_pos` instead.
   - Add `<DATASET_NAME>: BinaryClassMetrics` when `n_class == 2`.
   - Add `<DATASET_NAME>: MultiClassMetrics` when `n_class >= 3`.
   - Import any custom loader if needed.

## Implementation Rules

- Do not stop at a description of what should be implemented; generate the dataset benchmark code directly.
- Prefer dataset-specific parsing logic derived from the provided local files and official documentation.
- Do not leave `iter_labeled_recordings` as a generic placeholder unless the local raw-data layout is genuinely unavailable or incomplete.
- If the dataset cannot be completed because files are missing from `data_root`, say exactly which files or subfolders are missing and what layout the code expects.
- Keep the dataset CLI name, preprocess folder name, and `data_info_dict` key identical.
- Use the user-provided `dataset_name` consistently.
- Ensure `data_process/data_info.py` reflects the true `group_num`, `split`, `various_ch_num`, `n_class`, `sfreq`, `channel`, and `seq_len`.

## Validation

Run or document the preprocessing command. Then verify:

- All expected `group_*_data.npy` and `group_*_label.npy` files exist.
- Every data file is 3D and every label file is 1D.
- `data.shape[0] == label.shape[0]`.
- `data.shape[2] == sfreq * seq_len`.
- `data.shape[1] == channel`, unless `various_ch_num` is intentionally `True`.
- Labels are integer class IDs from `0` to `n_class - 1`.
- Groups are subject-disjoint unless the dataset has no subject identity.
- `data_info_dict` and `meta_info.py` keys exactly match the dataset name used by CLI `--dataset`.

If preprocessing cannot be run because the raw data requires credentials or manual download, still implement the pipeline and include exact manual download steps plus the expected `data_root` layout.
If preprocessing cannot be run because the local raw data at `data_root` is incomplete, still implement the pipeline and include the exact missing files or folders needed.

## Final Answer Requirements

When you finish a dataset-adding task, report:

- which files were created or updated
- which raw-data path the pipeline expects
- which paper / README / official site was used as the source reference
- whether the preprocessing code is dataset-specific and runnable or still blocked by missing raw files
- the expected preprocessing command to generate benchmark group data
