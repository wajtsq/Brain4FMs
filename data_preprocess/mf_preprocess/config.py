from dataclasses import dataclass

mayo_pat_ids = [0, 18, 21, 1, 9, 19, 2, 5, 16, 3, 4, 23, 6, 7, 8, 14, 17, 20]
mayo_groups = [[0, 18, 21], [1, 9, 19], [2, 5, 16], [3, 4, 23], [6, 7, 8], [14, 17, 20]]
# mayo_groups = [[0, 18, 21, 6], [1, 9, 19, 8], [2, 5, 16], [3, 4, 23], [14, 17, 20, 7]]
fnusa_pat_ids = [1, 5, 2, 9, 3, 4, 12, 6, 7, 8, 10, 11, 13]
fnusa_groups = [[1, 5], [3, 4, 12], [2, 6, 7], [8, 10], [9, 11, 13]]


@dataclass
class PreprocessArgs:

    min_duration: int = 60
    high_pass_filter: float = 0.01
    notch_filter: float = 60
    quality_factor: float = 30

    patch_secs: float = 1
    seq_len: int = 3  # secs
    sfreq: int = 1000
    group_num: int = 6

    # Please download the dataset from the official website and set `data_root` accordingly.
    data_root: str = ''
    data_save_dir: str = '/datasets'
