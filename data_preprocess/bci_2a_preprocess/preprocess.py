import os
import json
import sys
sys.path.append('/bench-mark')
import mne
import numpy as np
import pickle
from scipy.io import loadmat

from data_preprocess.bci_2a_preprocess.config import PreprocessArgs
from data_preprocess.utils import _split_subjects

# left_hand = 769, right_hand = 770, foot = 771, tongue = 772
def _find_eval_label_file(root_path, file):
    stem = os.path.splitext(file)[0]
    candidates = [
        os.path.join(root_path, f'{stem}.mat'),
        os.path.join(root_path, 'true_labels2a', f'{stem}.mat'),
        os.path.join(root_path, 'true_labels', f'{stem}.mat'),
        os.path.join(root_path, 'true_label', f'{stem}.mat'),
        os.path.join(root_path, 'labels', f'{stem}.mat'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f'Cannot find true label .mat for evaluation file {file}. '
        f'Expected one of: {candidates}'
    )


def _load_eval_labels(root_path, file):
    label_path = _find_eval_label_file(root_path, file)
    mat = loadmat(label_path)
    if 'classlabel' not in mat:
        raise KeyError(f'{label_path} does not contain "classlabel"')

    labels = np.asarray(mat['classlabel']).reshape(-1).astype(int) - 1
    return labels


def _load_gdf_epochs(root_path, file, channel_keys, label_dict):
    path = os.path.join(root_path, file)
    raw = mne.io.read_raw_gdf(path, stim_channel="auto", verbose='ERROR',
                            exclude=(["EOG-left", "EOG-central", "EOG-right"]))
    raw.rename_channels(channel_keys)

    events, events_id = mne.events_from_annotations(raw)
    event_ids = events[:, 2].astype(str)

    if 'T' in file:
        keys = events_id.keys() & label_dict.keys()
        new_dict = {}
        for key in keys:
            new_dict[key] = events_id[key]
        mask = np.isin(event_ids, [str(i) for i in list(new_dict.values())])
        labels = None
    elif 'E' in file:
        if '783' not in events_id:
            raise KeyError(f'{file} does not contain evaluation cue annotation "783"')
        new_dict = {'783': events_id['783']}
        mask = event_ids == str(events_id['783'])
        labels = _load_eval_labels(root_path, file)
    else:
        raise ValueError(f'Unsupported BCI-2a file name: {file}')

    events = events[mask]
    tmin, tmax = 1., 4.
    epochs = mne.Epochs(raw, events, event_id=new_dict, 
                            tmin=tmin, tmax=tmax, 
                            proj=True, baseline=None, preload=True)
    datas = epochs.get_data()[:, :, :750]
    datas = np.array(datas, dtype=np.float32)

    if labels is None:
        labels = epochs.events[:, -1]-events_id['769']
    elif len(labels) != len(datas):
        raise ValueError(
            f'{file} has {len(datas)} evaluation epochs but {len(labels)} labels in the .mat file'
        )

    return datas, labels


def get_data_all(root_path, data_save_dir):
    label_dict = {'769': 7, '770': 8, '771': 9, '772': 10}
    channel_keys = {'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
                'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6',
                'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4',
                'EEG-14': 'P1', 'EEG-15': 'Pz', 'EEG-16': 'P2', 'EEG-Pz': 'POz'}
    channel_file = os.path.join(data_save_dir, 'group_data', 'channels_lst.json')
    ch_names = list(channel_keys.values())
    if not os.path.exists(os.path.join(data_save_dir, 'group_data')):
        os.mkdir(os.path.join(data_save_dir, 'group_data'))
    if not os.path.exists(channel_file):
        with open(channel_file, 'w') as f:
            json.dump(ch_names, f)

    subject_data = {}
    subject_labels = {}
    for file in sorted(os.listdir(root_path)):
        if not file.endswith('.gdf') or ('T' not in file and 'E' not in file):
            continue

        subject = file[:3]
        datas, labels = _load_gdf_epochs(root_path, file, channel_keys, label_dict)
        subject_data.setdefault(subject, []).append(datas)
        subject_labels.setdefault(subject, []).append(labels)

    for subject in sorted(subject_data.keys()):
        datas = np.concatenate(subject_data[subject], axis=0)
        labels = np.concatenate(subject_labels[subject])
        np.save(os.path.join(data_save_dir, f'{subject}_data.npy'), datas)
        np.save(os.path.join(data_save_dir, f'{subject}_label.npy'), labels)
        print(f'data and label of subject {subject} saved')


def group_data(args):
    path = os.path.join(args.data_save_dir, 'subject_groups.pkl')
    if os.path.exists(path):
        groups = pickle.load(open(path, 'rb'))
    else:
        subject_list = [str(i).zfill(2) for i in range(1, args.subject_num + 1)]
        groups = _split_subjects(subject_list, args.group_num)
        pickle.dump(groups, open(path, 'wb'))

    for i, group in enumerate(groups):
        datas, labels = [], []
        for g in group:
            datas.append(np.load(os.path.join(args.data_save_dir, f'A{g}_data.npy')))
            labels.append(np.load(os.path.join(args.data_save_dir, f'A{g}_label.npy')))
        datas = np.concatenate(datas, axis=0)
        labels = np.concatenate(labels)

        np.save(os.path.join(args.data_save_dir, f'group_data/group_{i}_data.npy'), datas)
        np.save(os.path.join(args.data_save_dir, f'group_data/group_{i}_label.npy'), labels)
        print(f'data and label of group {i} saved')


args = PreprocessArgs()
get_data_all(args.data_root, args.data_save_dir)
group_data(args)
