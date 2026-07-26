from dataclasses import dataclass


@dataclass
class PreprocessArgs:

    high_pass_filter: float = 0.3
    low_pass_filter: float = 45
    notch_filter: float = 50
    quality_factor: float = 30

    patch_secs: float = 1
    subject_num: int = 100
    seq_len: int = 30  # secs
    sfreq: int = 200
    group_num: int = 5

    # Please download the dataset from the official website and set `data_root` accordingly.
    data_root: str = '/ISRUC_dataset/subgroup_1'
    data_save_dir: str = '/datasets/ISRUC'
