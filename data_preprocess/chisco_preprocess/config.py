from dataclasses import dataclass


@dataclass
class PreprocessArgs:

    high_pass_filter: float = 0.01
    notch_filter: float = 50
    quality_factor: float = 30

    patch_secs: float = 1
    subject_num: int = 5
    sfreq: int = 500
    seq_len: int = 1650   # 3.3 secs
    group_num: int = 5
    normal_ratio: float = 3

    # Please download the dataset from the official website and set `data_root` `label_path` and `class_path` accordingly.
    data_root: str = '/preprocessed_pkl'
    label_path: str = '/json/textmaps.json'
    data_save_dir: str = '/datasets/Chisco'
    class_path: str = '/json/classnumber.json'
