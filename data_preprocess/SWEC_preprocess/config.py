from dataclasses import dataclass


@dataclass
class PreprocessArgs:
    # SWEC HDF5 files are already band-pass filtered to 0.5-150 Hz and
    # downsampled to 512 or 1024 Hz by the dataset curators. Do not filter
    # them again during preprocessing.
    high_pass_filter: float = 0.5
    low_pass_filter: float = 150.0
    sfreq: int = 512
    patch_secs: float = 1.0
    seq_len: int = 5
    label_thres: float = 0.5
    normal_ratio: float = 3.0
    group_num: int = 5
    random_seed: int = 1
    sample_normal_windows: bool = True

    subject_num: int = 68

    data_root: str = "/SWEC_iEEG_Dataset"
    data_save_dir: str = "/datasets/SWEC"
