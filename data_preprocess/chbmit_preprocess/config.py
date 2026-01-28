from dataclasses import dataclass


@dataclass
class PreprocessArgs:

    min_duration: int = 60
    high_pass_filter: float = 0.01
    notch_filter: float = 60
    quality_factor: float = 30

    patch_secs: float = 1
    subject_num: int = 24
    seq_len: int = 10  # secs
    group_num: int = 5
    sfreq: int = 256
    normal_ratio: float = 3
    label_thres: float = 0.5

    # Please download the dataset from the official website and set `data_root` accordingly.
    data_root: str = '/files/chbmit/1.0.0/'
    data_save_dir: str = '/datasets/CHBMIT'
