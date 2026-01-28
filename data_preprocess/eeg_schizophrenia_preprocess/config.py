from dataclasses import dataclass


@dataclass
class PreprocessArgs:

    min_duration: int = 60
    high_pass_filter: float = 0.01
    notch_filter: float = 50
    quality_factor: float = 30

    patch_secs: float = 1
    subject_num: int = 28
    seq_len: int = 5  # secs
    group_num: int = 5

    sfreq: int = 250

    # Please download the dataset from the official website and set `data_root` accordingly.
    data_root: str = '/EEG_schizophrenia/edf_files'
    data_save_dir: str = '/datasets/EEG_schizophrenia'
