from dataclasses import dataclass


@dataclass
class PreprocessArgs:

    high_pass_filter: float = 0.01
    low_pass_filter: float = 70     # high gamma (<fs/2)
    notch_filter: float = 50
    quality_factor: float = 30

    patch_secs: float = 1
    subject_num: int = 109
    seq_len: int = 4  # secs
    sfreq: int = 160
    group_num: int = 6

    # Please download the dataset from the official website and set `data_root` accordingly.
    data_root: str = '/EEG_motor_movement_imagery/physionet.org/files/eegmmidb/1.0.0'
    data_save_dir: str = '/datasets/EEGMMIDB'
