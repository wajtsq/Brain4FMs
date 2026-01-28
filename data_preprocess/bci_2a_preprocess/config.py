from dataclasses import dataclass


@dataclass
class PreprocessArgs:
    high_pass_filter: float = 0.5
    low_pass_filter: float = 100
    notch_filter: float = 50

    patch_secs: float = 1
    subject_num: int = 9
    seq_len: int = 3  # secs
    group_num: int = 4
    sfreq: int = 250

    # Please download the dataset from the official website and set `data_root` accordingly.
    data_root: str = '/BCICIV_2a_gdf'
    data_save_dir: str = '/datasets/BCICIV_2a'
