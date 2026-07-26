from dataclasses import dataclass


@dataclass
class PreprocessArgs:

    # Standardized benchmark band-pass after resampling to 512 Hz.
    high_pass_filter: float = 0.1
    low_pass_filter: float = 150
    notch_filter: float = 60
    quality_factor: float = 30
    notch_harmonics: bool = True
    reference: str = 'laplace'  # none, car, laplace
    # Match the official Laplace boundary handling: channels without a
    # usable reference are marked bad/removed from the processed signal.
    keep_unreferenced_channels: bool = False
    n_jobs: int = 1
    mne_n_jobs: int = 1
    skip_existing: bool = True
    resample_before_filter: bool = True

    patch_secs: float = 1
    subject_num: int = 19
    sfreq: int = 512
    seq_len: int = 2
    group_num: int = 19
    random_seed: int = 0

    session: str = '1'
    task: str = 'Dur'
    center_prefix: str = 'CF'
    channel_types: tuple = ('ECOG',)
    dataset_name: str = 'Cogitate-CF-ECoG'
    bad_channel_descriptions: tuple = (
        'epileptic_onset',
        'noisy_user1',
        'dead_user1',
        'wrong_channel_user1',
        'outside_brain',
    )

    event_type: str = 'stimulus onset'
    # The Dur paradigm has three task-relevance levels.  The old defaults
    # selected only Irrelevant trials and then decoded face vs object, which
    # is neither the target-detection nor the task-relevance contrast used in
    # the Cogitate protocol.
    # Official Cogitate categorical-decoding example: face vs object.
    # Keep the task-irrelevant condition so the benchmark does not encode
    # button-press/report-related differences in the category label.
    label_mode: str = 'category'  # target_detection, task_relevance, category
    target_column: str = 'category'
    label_values: tuple = ('face', 'object')
    task_relevance_values: tuple = ('Irrelevant',)
    response_values: tuple = ()
    # The official design balances duration over the experiment but retains
    # small per-condition imbalances; do not add benchmark-side resampling.
    balance_category_duration: bool = False
    epoch_tmin: float = -0.5
    epoch_tmax: float = 1.5
    baseline: tuple = (None, 0.0)

    data_root: str = '/Cogitate/mnt/beegfs/workspace/2023-0385-Cogitatedatarelease/CURATE/COG_ECOG_EXP1_BIDS'
    data_save_root: str = '/datasets'
