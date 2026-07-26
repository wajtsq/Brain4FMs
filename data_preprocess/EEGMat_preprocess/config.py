from dataclasses import dataclass


@dataclass
class PreprocessArgs:

    min_duration: int = 60
    high_pass_filter: float = 0.1
    low_pass_filter: float = 75
    notch_filter: float = 50
    quality_factor: float = 30

    patch_secs: float = 1
    subject_num: int = 36
    seq_len: int = 10  # secs
    sfreq: int = 250
    group_num: int = 4

    # Please download the dataset from the official website and set `data_root` accordingly.
    data_root: str = '/EEGMat'
    data_save_dir: str = '/datasets/EEGMat'
