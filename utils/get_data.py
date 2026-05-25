import pickle
import os
import numpy as np

from utils.data_info import data_info_dict


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
        x = np.load(os.path.join(data_load_path, f'group_{g_id}_data.npy'))
        if len(x.shape) > 3:
            bsz, ch_num, _, _ = x.shape
            if ch_num != args.cnn_in_channels:
                x = x.transpose(1,0,2,3)
                bsz, ch_num, _, _ = x.shape
            x = x.reshape(bsz, ch_num, -1)      # (bsz, ch_num, N)
            
        y = np.load(os.path.join(data_load_path, f'group_{g_id}_label.npy'))
        group_x_list.append(x)
        group_y_list.append(y)

    perm = np.array([i for i in range(args.cnn_in_channels)])
    if (step == 'train' or step == 'valid') and args.exp_id == '-4':
        np.random.seed(args.cv_id + 1)
        perm = np.random.permutation(args.cnn_in_channels)
        args.perm = perm
        group_x_list = [group_x[:, perm, :] for group_x in group_x_list]
    else:
        args.perm = None

    if not various_ch_num:
        group_y_list = [np.concatenate(group_y_list, axis=0)]

    group_x_list = [{'x': np.concatenate(group_x_list, axis=0), 'perm': perm}]

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
        x = np.load(os.path.join(data_load_path, f'group_{g_id}_data.npy'))
        if len(x.shape) > 3:
            bsz, ch_num, _, _ = x.shape
            if ch_num != args.cnn_in_channels:
                x = x.transpose(1,0,2,3)
                bsz, ch_num, _, _ = x.shape
            x = x.reshape(bsz, ch_num, -1)      # (bsz, ch_num, N)
            
        y = np.load(os.path.join(data_load_path, f'group_{g_id}_label.npy'))
        pos = np.load(os.path.join(data_load_path, f'group_{g_id}_pos.npy'))
        group_x_list.append(x)
        group_y_list.append(y)
        group_pos_list.append(pos)

    perm = np.array([i for i in range(args.cnn_in_channels)])
    if (step == 'train' or step == 'valid') and args.exp_id == '-4':
        np.random.seed(args.cv_id + 1)
        perm = np.random.permutation(args.cnn_in_channels)
        args.perm = perm
        group_x_list = [group_x[:, perm, :] for group_x in group_x_list]
    else:
        args.perm = None

    if not various_ch_num:
        group_x_list = [{'x': np.concatenate(group_x_list, axis=0), 'perm': perm,
                         'pos': np.concatenate(group_pos_list, axis=0)}]
        group_y_list = [np.concatenate(group_y_list, axis=0)]
        
    return group_x_list, group_y_list


def sample_training_data(group_tr_x, group_tr_y, group_vl_x, group_vl_y, shot=8):
    label = np.concatenate((group_tr_y[0], group_vl_y[0]), axis=0)
    if isinstance(group_tr_x[0], dict):
        tr_dict = group_tr_x[0]
        vl_dict = group_vl_x[0]
        data = np.concatenate((tr_dict['x'], vl_dict['x']), axis=0)
        sampled = {
            'x': data,
            'perm': tr_dict.get('perm', None),
        }
        if 'pos' in tr_dict and 'pos' in vl_dict:
            sampled['pos'] = np.concatenate((tr_dict['pos'], vl_dict['pos']), axis=0)
    else:
        data = np.concatenate((group_tr_x, group_vl_x), axis=0)
        sampled = data

    categories = np.unique(label)
    rng = np.random.default_rng()
    sample_idx = []
    for category in categories:
        idx = np.where(label == category)[0]
        sample_idx.append(idx[rng.permutation(len(idx))][:shot])

    sample_idx = np.concatenate(sample_idx)

    if isinstance(sampled, dict):
        out = {
            'x': sampled['x'][sample_idx],
            'perm': sampled.get('perm', None),
        }
        if 'pos' in sampled:
            out['pos'] = sampled['pos'][sample_idx]
        return out, label[sample_idx]

    return sampled[sample_idx], label[sample_idx]
