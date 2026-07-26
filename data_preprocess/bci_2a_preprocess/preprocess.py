import json
import os
import sys

sys.path.append('/bench-mark')

import mne
import numpy as np
import pickle
from scipy.io import loadmat

from data_preprocess.bci_2a_preprocess.config import PreprocessArgs
from data_preprocess.utils import _split_subjects


LABEL_DICT = {'769': 7, '770': 8, '771': 9, '772': 10}
CLASS_KEYS = ['769', '770', '771', '772']
CLASS_EVENT_CODES = {key: 769 + idx for idx, key in enumerate(CLASS_KEYS)}
CHANNEL_KEYS = {
    'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
    'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6',
    'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4',
    'EEG-14': 'P1', 'EEG-15': 'Pz', 'EEG-16': 'P2', 'EEG-Pz': 'POz'
}


def _save_channels(save_dir):
    channel_file = os.path.join(save_dir, 'channels_lst.json')
    if not os.path.exists(channel_file):
        with open(channel_file, 'w') as f:
            json.dump(list(CHANNEL_KEYS.values()), f)


def _load_eval_labels(data_root, subject_id):
    candidate_paths = [
        os.path.join(data_root, f'A{subject_id}E.mat'),
        os.path.join(data_root, f'A{subject_id}E.gdf.mat'),
        os.path.join(data_root, 'true_labels', f'A{subject_id}E.mat'),
        os.path.join(data_root, 'true_labels', f'A{subject_id}E.gdf.mat'),
        os.path.join(data_root, 'labels', f'A{subject_id}E.mat'),
        os.path.join(data_root, 'labels', f'A{subject_id}E.gdf.mat'),
        os.path.join(os.path.dirname(data_root), 'true_labels2a', f'A{subject_id}E.mat'),
        os.path.join(os.path.dirname(data_root), 'true_labels2a', f'A{subject_id}E.gdf.mat'),
    ]
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        mat = loadmat(path)
        if 'classlabel' in mat:
            return mat['classlabel'].reshape(-1).astype(np.int64) - 1
    raise FileNotFoundError(
        f'Cannot find evaluation labels for subject A{subject_id}. '
        f'Expected one of: {candidate_paths}'
    )


def _load_session_data(file_path, split, eval_labels=None):
    raw = mne.io.read_raw_gdf(
        file_path,
        stim_channel='auto',
        verbose='ERROR',
        exclude=(['EOG-left', 'EOG-central', 'EOG-right'])
    )
    raw.rename_channels(CHANNEL_KEYS)
    raw.load_data(verbose='ERROR')
    raw.filter(
        l_freq=args.high_pass_filter,
        h_freq=args.low_pass_filter,
        verbose='ERROR',
    )
    raw.notch_filter(freqs=[args.notch_filter], verbose='ERROR')

    events, events_id = mne.events_from_annotations(raw)
    event_codes = events[:, 2]

    artifact_code = events_id.get('1023')
    valid_trial_mask = np.ones(len(events), dtype=bool)
    if artifact_code is not None:
        artifact_positions = set(events[event_codes == artifact_code, 0].tolist())
        valid_trial_mask = np.array([pos not in artifact_positions for pos in events[:, 0]])

    tmin, tmax = 1.0, 4.0
    if split == 'train':
        available_keys = [key for key in CLASS_KEYS if key in events_id]
        new_dict = {key: events_id[key] for key in available_keys}
        event_ids = event_codes.astype(str)
        mask = np.isin(event_ids, [str(i) for i in new_dict.values()]) & valid_trial_mask
        selected_events = events[mask]
        epochs = mne.Epochs(
            raw,
            selected_events,
            event_id=new_dict,
            tmin=tmin,
            tmax=tmax,
            proj=True,
            baseline=None,
            preload=True,
            verbose='ERROR',
        )
        code_to_label = {events_id[key]: idx for idx, key in enumerate(CLASS_KEYS) if key in events_id}
        labels = np.array([code_to_label[code] for code in epochs.events[:, -1]], dtype=np.int64)
    else:
        unknown_code = events_id.get('783')
        if unknown_code is None:
            raise ValueError(f'Cannot find unknown cue marker 783 in {file_path}.')
        mask = (event_codes == unknown_code) & valid_trial_mask
        selected_events = events[mask].copy()
        if eval_labels is None:
            raise ValueError(f'Evaluation labels are required for {file_path}.')
        if len(selected_events) != len(eval_labels):
            raise ValueError(
                f'Label count mismatch for {file_path}: '
                f'{len(selected_events)} events vs {len(eval_labels)} labels.'
            )
        selected_events[:, 2] = eval_labels + CLASS_EVENT_CODES['769']
        event_id = CLASS_EVENT_CODES.copy()
        epochs = mne.Epochs(
            raw,
            selected_events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            proj=True,
            baseline=None,
            preload=True,
            verbose='ERROR',
        )
        labels = eval_labels

    datas = epochs.get_data()[:, :, :750].astype(np.float32)
    return datas, labels.astype(np.int64)


# left_hand = 769, right_hand = 770, foot = 771, tongue = 772
def get_data_all(args):
    root_path = args.data_root
    data_save_dir = args.data_save_dir
    group_dir = os.path.join(data_save_dir, 'group_data')
    os.makedirs(group_dir, exist_ok=True)
    _save_channels(group_dir)

    for subject in range(1, args.subject_num + 1):
        subject_id = f'{subject:02d}'
        subject_datas, subject_labels = [], []

        for session_id in args.cross_subject_sessions:
            file_path = os.path.join(root_path, f'A{subject_id}{session_id}.gdf')
            if session_id == 'T':
                datas, labels = _load_session_data(file_path, split='train')
            elif session_id == 'E':
                eval_labels = _load_eval_labels(root_path, subject_id)
                datas, labels = _load_session_data(file_path, split='test', eval_labels=eval_labels)
            else:
                raise ValueError(f'Unsupported cross-subject session: {session_id}')

            np.save(os.path.join(data_save_dir, f'A{subject_id}_session{session_id}_data.npy'), datas)
            np.save(os.path.join(data_save_dir, f'A{subject_id}_session{session_id}_label.npy'), labels)
            subject_datas.append(datas)
            subject_labels.append(labels)

        subject_datas = np.concatenate(subject_datas, axis=0)
        subject_labels = np.concatenate(subject_labels)
        np.save(os.path.join(data_save_dir, f'A{subject_id}_data.npy'), subject_datas)
        np.save(os.path.join(data_save_dir, f'A{subject_id}_label.npy'), subject_labels)
        print(f'cross-subject data of subject A{subject_id} saved')


def group_data(args):
    path = os.path.join(args.data_save_dir, 'subject_groups.pkl')
    if os.path.exists(path):
        groups = pickle.load(open(path, 'rb'))
    else:
        subject_list = [str(i).zfill(2) for i in range(1, args.subject_num + 1)]
        groups = _split_subjects(subject_list, args.group_num)
        pickle.dump(groups, open(path, 'wb'))

    group_dir = os.path.join(args.data_save_dir, 'group_data')
    os.makedirs(group_dir, exist_ok=True)
    for i, group in enumerate(groups):
        datas, labels = [], []
        for g in group:
            datas.append(np.load(os.path.join(args.data_save_dir, f'A{g}_data.npy')))
            labels.append(np.load(os.path.join(args.data_save_dir, f'A{g}_label.npy')))
        datas = np.concatenate(datas, axis=0)
        labels = np.concatenate(labels)

        np.save(os.path.join(group_dir, f'group_{i}_data.npy'), datas)
        np.save(os.path.join(group_dir, f'group_{i}_label.npy'), labels)
        print(f'data and label of group {i} saved')


args = PreprocessArgs()

# Cross-subject pipeline
get_data_all(args)
group_data(args)
