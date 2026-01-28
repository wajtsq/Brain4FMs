import os
import numpy as np

import pickle
from data_preprocess.utils import _split_subjects, _std_data_segment


def _generate_bdi_groups(args, bdi_dict):
    path = os.path.join(args.data_save_dir, 'bdi_groups.pkl')
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    else:
        l0, l1 = [], []
        for pid, bdi in bdi_dict.items():
            if bdi < 7:
                l0.append(pid)
            elif bdi > 13:
                l1.append(pid)

        g0 = _split_subjects(l0, args.group_num)
        g1 = _split_subjects(l1, args.group_num)

        groups = []
        for i in range(args.group_num):
            groups.append(g0[i] + g1[i])
        pickle.dump(groups, open(path, 'wb'))

    return groups


def _generate_stai_groups(args, stai_dict):
    path = os.path.join(args.data_save_dir, 'stai_groups.pkl')
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    else:
        l0, l1, l2 = [], [], []
        for pid, stai in stai_dict.items():
            if 20 <= stai <= 40:
                l0.append(pid)
            elif 40 < stai <= 60:
                l1.append(pid)
            elif 60 < stai <= 80:
                l2.append(pid)

        g0 = _split_subjects(l0, args.group_num)
        g1 = _split_subjects(l1, args.group_num)
        g2 = _split_subjects(l2, args.group_num)

        groups = []
        for i in range(args.group_num):
            groups.append(g0[i] + g1[i] + g2[i])
        pickle.dump(groups, open(path, 'wb'))

    return groups


def _get_level(score, group_std):
    level = None
    if group_std == 'BDI':
        if score < 7:
            level = 0
        elif score > 13:
            level = 1
    elif group_std == 'STAI':
        if 20 <= score <= 40:
            level = 0
        elif 40 < score <= 60:
            level = 1
        elif 60 < score <= 80:
            level = 2
    else:
        raise ValueError

    return level


def _merge_data(args, group, score_dict, group_std):
    data, label = [], []
    for sub in group:
        score = score_dict[sub]
        level = _get_level(score, group_std)
        sub_data = np.load(os.path.join(args.data_save_dir, f'{sub}_data_run1.npy'))
        sub_label = level * np.ones(sub_data.shape[0], dtype=int)

        # sub_data = _std_data_segment(sub_data)
        # bsz, ch_num, _ = sub_data.shape
        # sub_data = sub_data.reshape(bsz, ch_num, -1)

        data.append(sub_data)
        label.append(sub_label)

    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)

    return data, label

