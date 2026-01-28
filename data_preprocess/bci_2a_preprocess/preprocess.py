import os
import json
import sys
sys.path.append('/bench-mark')
import mne
import numpy as np
import pickle

from data_preprocess.bci_2a_preprocess.config import PreprocessArgs
from data_preprocess.utils import _split_subjects, _merge_data

# left_hand = 769, right_hand = 770, foot = 771, tongue = 772
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

    for file in os.listdir(root_path):
        if 'T' not in file:
            continue

        path = os.path.join(root_path, file)
        raw = mne.io.read_raw_gdf(path, stim_channel="auto", verbose='ERROR',
                                exclude=(["EOG-left", "EOG-central", "EOG-right"]))
        raw.rename_channels(channel_keys)

        events, events_id = mne.events_from_annotations(raw)
        event_ids = events[:, 2].astype(str)
        keys = events_id.keys() & label_dict.keys()
        new_dict = {}
        for key in keys:
            new_dict[key] = events_id[key]
        mask = np.isin(event_ids, [str(i) for i in list(new_dict.values())])
        events = events[mask]
        tmin, tmax = 1., 4.
        epochs = mne.Epochs(raw, events, event_id=new_dict, 
                                tmin=tmin, tmax=tmax, 
                                proj=True, baseline=None, preload=True)
        labels = epochs.events[:, -1]-events_id['769']
        datas = epochs.get_data()[:, :, :750]
        datas = np.array(datas, dtype=np.float32)
        np.save(os.path.join(data_save_dir, f'{file[:3]}_data.npy'), datas)
        np.save(os.path.join(data_save_dir, f'{file[:3]}_label.npy'), labels)


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