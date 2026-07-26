import pickle
import os
import json
import numpy as np

from utils.data_info import data_info_dict


def _load_channel_names(data_load_path, group_id=None):
    channel_paths = []
    if group_id is not None:
        channel_paths.append(os.path.join(data_load_path, f'group_{group_id}_channel_lst.json'))
    channel_paths.extend([
        os.path.join(data_load_path, 'channels_lst.json'),
        os.path.join(os.path.dirname(data_load_path), 'channels_lst.json'),
    ])
    for channel_path in channel_paths:
        if os.path.exists(channel_path):
            with open(channel_path, 'r') as f:
                return json.load(f), channel_path
    return None, None


def _reshape_group_x(x, y, args):
    if len(x.shape) > 3:
        if x.shape[0] != len(y) and x.shape[1] == len(y):
            x = x.transpose(1, 0, 2, 3)
        bsz, ch_num, _, _ = x.shape
        x = x.reshape(bsz, ch_num, -1)
    if len(x.shape) < 3:
        x = np.expand_dims(x, axis=1)
    return x


def _make_group_packet(x, data_load_path, group_id, step, args, channel_names=None, channel_path=None, pos=None):
    ch_num = x.shape[1]
    perm = np.arange(ch_num)
    if (step == 'train' or step == 'valid') and args.exp_id == '-4':
        rng = np.random.default_rng(args.cv_id + 1 + int(group_id))
        perm = rng.permutation(ch_num)
        x = x[:, perm, :]
        if channel_names is not None:
            channel_names = [channel_names[i] for i in perm]
        if pos is not None:
            pos = pos[perm] if pos.shape[0] == ch_num else pos

    packet = {
        'x': x,
        'perm': perm,
        'group_id': group_id,
        'channel_names': channel_names,
        'channel_path': channel_path,
        'cnn_in_channels': ch_num,
        'full_data_path': data_load_path,
        'channels_already_ordered': True,
    }
    if pos is not None:
        packet['pos'] = pos
    return packet


def default_get_data(args, step,):
    # data_load_path = f'{args.data_load_dir}/{args.data_id}/'
    group_num      = data_info_dict[args.dataset]['group_num']
    split          = data_info_dict[args.dataset]['split']
    various_ch_num = data_info_dict[args.dataset]['various_ch_num']
    data_load_path = data_info_dict[args.dataset]['data_path']

    indices = list(range(group_num))
    shift = args.cv_id
    indices = indices[-shift:] + indices[:-shift]

    tr_indices = indices[ : split[0]]
    vl_indices = indices[split[0] : -split[2]]
    ts_indices = indices[-split[2] : ]

    if step == 'train':
        target_indices = tr_indices
    elif step == 'valid':
        target_indices = vl_indices
    elif step == 'test':
        target_indices = ts_indices
    else:
        raise NotImplementedError('Unknown step.')

    print(f'{step} group indices: {target_indices}')

    group_x_list, group_y_list = [], []
    for g_id in target_indices:
        y = np.load(os.path.join(data_load_path, f'group_{g_id}_label.npy'))
        x = np.load(os.path.join(data_load_path, f'group_{g_id}_data.npy'))
        x = _reshape_group_x(x, y, args)
        channel_names, channel_path = _load_channel_names(data_load_path, g_id)

        if various_ch_num:
            group_x_list.append(_make_group_packet(x, data_load_path, g_id, step, args, channel_names, channel_path))
        else:
            group_x_list.append(x)
        group_y_list.append(y)

    if not various_ch_num:
        group_x_list = [np.concatenate(group_x_list, axis=0)]
        group_y_list = [np.concatenate(group_y_list, axis=0)]

        perm = np.array([i for i in range(args.cnn_in_channels)])
        if (step == 'train' or step == 'valid') and args.exp_id == '-4':     # shuffled
            np.random.seed(args.cv_id+1)
            perm = np.random.permutation(args.cnn_in_channels)
            args.perm = perm
            group_x_list = [group_x[:, perm, :] for group_x in group_x_list]
        else:
            args.perm = None
        channel_names, channel_path = _load_channel_names(data_load_path)
        if channel_names is not None and perm is not None and len(channel_names) >= len(perm):
            channel_names = [channel_names[i] for i in perm]
        group_x_list = [{
            'x': group_x_list[0],
            'perm': perm,
            'group_id': None,
            'channel_names': channel_names,
            'channel_path': channel_path,
            'cnn_in_channels': args.cnn_in_channels,
            'full_data_path': data_load_path,
            'channels_already_ordered': True,
        }]
    else:
        args.perm = None

    return group_x_list, group_y_list



def clinical_get_data(args, step):
    data_load_path = f'{args.data_load_dir}/{args.data_id}/'

    group_num      = data_info_dict[args.dataset]['group_num']
    split          = data_info_dict[args.dataset]['split']
    various_ch_num = data_info_dict[args.dataset]['various_ch_num']

    indices = list(range(1, group_num+1))   # g1, g2, g3, g4
    shift = args.cv_id
    indices = indices[-shift:] + indices[:-shift]

    tr_indices = indices[ : split[0]]
    vl_indices = indices[split[0] : -split[2]]
    ts_indices = indices[-split[2] : ]

    if step == 'train':
        target_indices = tr_indices
    elif step == 'valid':
        target_indices = vl_indices
    elif step == 'test':
        target_indices = ts_indices
    else:
        raise NotImplementedError('Unknown step.')

    print(f'{step} group indices: {target_indices}')

    group_x_list, group_y_list = [], []
    for g_id in target_indices:
        if step != 'test':
            x = pickle.load(open(data_load_path + f'sampled_g{g_id}_x.pkl', 'rb'))
            y = pickle.load(open(data_load_path + f'sampled_g{g_id}_y.pkl', 'rb'))
        else:
            x = pickle.load(open(data_load_path + f'unsampled_g{g_id}_x.pkl', 'rb'))
            y = pickle.load(open(data_load_path + f'unsampled_g{g_id}_y.pkl', 'rb'))
        group_x_list += x
        group_y_list += y

    if not various_ch_num:
        group_x_list = [np.concatenate(group_x_list, axis=0)]
        group_y_list = [np.concatenate(group_y_list, axis=0)]

    return group_x_list, group_y_list


def default_get_data_with_pos(args, step,):
    # data_load_path = f'{args.data_load_dir}/{args.data_id}/'
    group_num      = data_info_dict[args.dataset]['group_num']
    split          = data_info_dict[args.dataset]['split']
    various_ch_num = data_info_dict[args.dataset]['various_ch_num']
    data_load_path = data_info_dict[args.dataset]['data_path']

    indices = list(range(group_num))
    shift = args.cv_id
    indices = indices[-shift:] + indices[:-shift]

    tr_indices = indices[ : split[0]]
    vl_indices = indices[split[0] : -split[2]]
    ts_indices = indices[-split[2] : ]

    if step == 'train':
        target_indices = tr_indices
    elif step == 'valid':
        target_indices = vl_indices
    elif step == 'test':
        target_indices = ts_indices
    else:
        raise NotImplementedError('Unknown step.')

    print(f'{step} group indices: {target_indices}')

    group_x_list, group_y_list, group_pos_list = [], [], []
    for g_id in target_indices:
        y = np.load(os.path.join(data_load_path, f'group_{g_id}_label.npy'))
        x = np.load(os.path.join(data_load_path, f'group_{g_id}_data.npy'))
        x = _reshape_group_x(x, y, args)
        pos = np.load(os.path.join(data_load_path, f'group_{g_id}_pos.npy'))
        channel_names, channel_path = _load_channel_names(data_load_path, g_id)
        if various_ch_num:
            group_x_list.append(_make_group_packet(x, data_load_path, g_id, step, args, channel_names, channel_path, pos))
        else:
            group_x_list.append(x)
        group_y_list.append(y)
        group_pos_list.append(pos)

    if not various_ch_num:
        perm = np.array([i for i in range(args.cnn_in_channels)])
        if (step == 'train' or step == 'valid') and args.exp_id == '-4':     # shuffled
            np.random.seed(args.cv_id+1)
            perm = np.random.permutation(args.cnn_in_channels)
            args.perm = perm
            group_x_list = [group_x[:, perm, :] for group_x in group_x_list]
        else:
            args.perm = None
        channel_names, channel_path = _load_channel_names(data_load_path)
        if channel_names is not None and perm is not None and len(channel_names) >= len(perm):
            channel_names = [channel_names[i] for i in perm]
        group_x_list = [{'x': np.concatenate(group_x_list, axis=0), 'perm': perm,
                         'pos': np.concatenate(group_pos_list, axis=0),
                         'group_id': None,
                         'channel_names': channel_names,
                         'channel_path': channel_path,
                         'cnn_in_channels': args.cnn_in_channels,
                         'full_data_path': data_load_path,
                         'channels_already_ordered': True}]
        group_y_list = [np.concatenate(group_y_list, axis=0)]
    else:
        args.perm = None
        
    return group_x_list, group_y_list


def sample_training_data(group_tr_x, group_tr_y, group_vl_x, group_vl_y, shot=8):
    if isinstance(group_tr_x[0], dict) and len(group_tr_x) > 1:
        raise ValueError('few-shot/prototype sampling is not supported for various_ch_num=True yet.')
    label = np.concatenate((group_tr_y[0], group_vl_y[0]), axis=0)
    if isinstance(group_tr_x[0], dict):
        group_tr_x = group_tr_x[0]['x']
        group_vl_x = group_vl_x[0]['x']
    data = np.concatenate((group_tr_x, group_vl_x), axis=0)
    category = np.unique(label)
    rng = np.random.default_rng()   # seed = random
    sample_idx = []
    for c in category:
        idx = np.where(label == c)[0]
        sample_num = shot
            
        sample_idx.append(
            idx[rng.permutation(len(idx))][:sample_num]
        )

    sample_idx = np.concatenate(sample_idx)

    return data[sample_idx], label[sample_idx]
