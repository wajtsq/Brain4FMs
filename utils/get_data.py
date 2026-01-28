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

    if not various_ch_num:
        group_x_list = [np.concatenate(group_x_list, axis=0)]
        group_y_list = [np.concatenate(group_y_list, axis=0)]

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

    if not various_ch_num:
        group_x_list = [{'x': np.concatenate(group_x_list, axis=0),
                         'pos': np.concatenate(group_pos_list, axis=0)}]
        group_y_list = [np.concatenate(group_y_list, axis=0)]
        
    return group_x_list, group_y_list