from dataclasses import dataclass


@dataclass
class PreprocessArgs:

    # Raw seizure/non-seizure benchmark settings.
    # HUP/ds004100 has variable native sampling rates; use one common rate for
    # fixed-length raw segments. Set to None to keep each file's native rate.
    sfreq: int = 512

    # Conservative signal cleanup for raw iEEG. The 60 Hz value is from the BIDS
    # sidecars. The 0.5-120 Hz band is configurable rather than treated as a
    # dataset-paper requirement.
    high_pass_filter: float = 0.5
    low_pass_filter: float = 120.0
    notch_filter: float = 60.0
    quality_factor: float = 30

    patch_secs: float = 1
    seq_len: int = 2  # secs
    group_num: int = 5
    label_thres: float = 0.5
    min_duration: int = 60
    normal_ratio: float = 3.0

    # Dataset-specific BIDS choices: remove channels marked bad and use clinical
    # iEEG channels only. Keep the original EDF reference by default; do not
    # apply bipolar/CAR unless explicitly needed for a separate experiment.
    pick_channel_status: str = "good"
    channel_types: tuple = ("ECOG", "SEEG")
    exclude_name_keywords: tuple = ("EKG", "ECG", "TRIG", "DC")
    common_average_reference: bool = False

    # Preserve metadata columns supplied by participants.tsv for later analysis.
    groupby_columns: tuple = ("outcome", "engel", "therapy", "implant", "target", "lesion_status")

    data_root: str = '/ds004100-download'
    data_save_dir: str = '/datasets/HUP'
