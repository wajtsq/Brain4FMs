from dataclasses import dataclass


@dataclass
class PreprocessArgs:

    min_duration: int = 60
    high_pass_filter: float = 0.3
    low_pass_filter: float = 45
    notch_filter: float = 50
    quality_factor: float = 30

    patch_secs: float = 1
    sfreq: float = 100
    ch_num: int = 1
    seq_len: int = 30  # secs
    group_num: int = 5
    subject_num: int = 44

    data_root: str = '/SleepEDFx_dataset/physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry'
    data_save_dir: str = '/datasets/SleepEDFx'
