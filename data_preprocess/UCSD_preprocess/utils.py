import os
import numpy as np
import random
import pickle


def count_subjects(path, kind, task):
    files = os.listdir(path)
    subjects = []
    for file in files:
        if '.npy' in file and kind in file and task in file:
            subjects.append(file.split('_')[0])
    return subjects
        

def _generate_subject_groups(args, task):
    path = os.path.join(args.data_save_dir, f'subject_list_{task}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            subject_list = pickle.load(f)
    
    else:
        subject_list = []
        hc_numbers = count_subjects(args.data_save_dir, 'hc', 'hc')
        pd_numbers = count_subjects(args.data_save_dir, 'pd', task)
        hc_num = len(hc_numbers)
        pd_num = len(pd_numbers)
        
        group_sub_num_hc = hc_num // args.group_num
        group_sub_num_pd = pd_num // args.group_num
        remainder_hc = hc_num % args.group_num
        remainder_pd = pd_num % args.group_num
        
        start_hc, start_pd = 0, 0
        for i in range(args.group_num):
            group_size_hc = group_sub_num_hc
            group_size_pd = group_sub_num_pd
            if i < remainder_hc:
                group_size_hc += 1
            if i < remainder_pd:
                group_size_pd += 1
                
            group = hc_numbers[start_hc: start_hc + group_size_hc] + pd_numbers[start_pd: start_pd + group_size_pd]
            subject_list.append(group)
            start_hc += group_size_hc
            start_pd += group_size_pd
            
        with open(os.path.join(path), 'wb') as f:
            pickle.dump(subject_list, f)
            
    return subject_list

selected_channel_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 
                              'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 
                              'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 
                              'F8', 'AF4', 'Fp2', 'Fz', 'Cz']

def _select_channels_from_edf(raw):
    all_channel_names = raw.info['ch_names']
    raw.pick_channels(selected_channel_names)
    return raw

def store_channels(args):
    import json
    channel_file_on = os.path.join(args.data_save_dir, f'group_data_on', 'channels_lst.json')
    channel_file_off = os.path.join(args.data_save_dir, f'group_data_off', 'channels_lst.json')
    if not os.path.exists(channel_file_on):
        with open(channel_file_on, 'w') as f:
            json.dump(selected_channel_names, f)
        with open(channel_file_off, 'w') as f:
            json.dump(selected_channel_names, f)


def _merge_data(args, group, task):
    datas, labels = [], []
    for sub in group:
        if 'hc' in sub:
            label = 0
            path = os.path.join(args.data_save_dir, f'{sub}_hc_data.npy')
        else:
            label = 1
            path = os.path.join(args.data_save_dir, f'{sub}_{task}_data.npy')
            
        data = np.load(path)           
        datas.append(data)
        labels.append([label]*data.shape[0])
        
    datas = np.concatenate(datas, axis=0)
    labels = np.concatenate(labels)
    return datas, labels
