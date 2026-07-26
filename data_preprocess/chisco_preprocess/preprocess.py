import os
import sys
sys.path.append('/bench-mark')
import numpy as np
import pickle
import json
from data_preprocess.chisco_preprocess.config import PreprocessArgs
from data_preprocess.utils import _split_subjects


def load_subject(args, subjects, task):
    valid_run_ids = set([f"0{i}" for i in range(1, 46)])
    datas, labels = [], []
    for subj in subjects:
        subj_path = os.path.join(args.data_root, f"sub-{subj}", "eeg")
        if not os.path.exists(subj_path):
            print(f"⚠️ Subject folder not found: {subj_path}")
            continue

        for fn in sorted(os.listdir(subj_path)):
            if not (fn.endswith(".pkl") and f"sub-{subj}" in fn and f"task-{task}" in fn):
                continue
            try:
                run_id = fn.split("run-")[1].split("_")[0]
            except IndexError:
                print(f"⚠️ Unable to resolve run number: {fn}")
                continue
            if run_id not in valid_run_ids:
                continue

            with open(os.path.join(subj_path, fn), "rb") as f:
                trials = pickle.load(f)
            with open(args.label_path, 'r') as f:
                textmap = json.load(f)
                
            for tr in trials:
                sentence = str(tr.get("text", "")).strip()
                eeg = tr["input_features"][0, :122, :args.seq_len].astype(np.float32) * 1e6
                datas.append(eeg)
                labels.append(textmap[sentence])
    return np.array(datas, dtype=np.float32), np.array(labels)
                    

def group_data(args, task):
    path = os.path.join(args.data_save_dir, 'subject_groups.pkl')
    class_path = os.path.join(args.data_save_dir, 'classnumber.json')
    if os.path.exists(path):
        groups = pickle.load(open(path, 'rb'))
    else:
        subject_list = [str(i).zfill(2) for i in range(1, args.subject_num + 1)]
        groups = _split_subjects(subject_list, args.group_num)
        pickle.dump(groups, open(path, 'wb'))
    if not os.path.exists(class_path):
        with open(args.class_path, "r", encoding='gbk') as f:
            classnumber = json.load(f)
        with open(class_path, "w") as f:
            json.dump(classnumber, f, ensure_ascii=False, indent=4)

    for i, g in enumerate(groups):
        data, label = load_subject(args, g, task)
        np.save(os.path.join(args.data_save_dir, f'group_data_{task}/group_{i}_data.npy'), data)
        np.save(os.path.join(args.data_save_dir, f'group_data_{task}/group_{i}_label.npy'), label)
        print(f'data and label of group {i} saved')

def split_data(data, label, num_groups, path, subject):
    save_path = os.path.join(path, f'sub-0{subject}', 'group_data')
    os.makedirs(save_path, exist_ok=True)  # Create the output directory

    data_per_group = len(data) // num_groups
    
    for i in range(num_groups):
        # Compute the start and end indices for the current group
        start_idx = i * data_per_group
        end_idx = (i + 1) * data_per_group if i != num_groups - 1 else len(data)
        
        # Split the data and labels
        group_data = data[start_idx:end_idx]
        group_label = label[start_idx:end_idx]
        
        # Save as .npy files
        np.save(os.path.join(save_path, f'group_{i}_data.npy'), group_data)
        np.save(os.path.join(save_path, f'group_{i}_label.npy'), group_label)


def store_channel(args, channels):
    import json
    channel_file = os.path.join(args.data_save_dir, 'group_data', 'channels_lst.json')
    ch_names = [name.split('-')[0]+'-'+name.split('-')[1] for name in channels]
    if not os.path.exists(channel_file):
        with open(channel_file, 'w') as f:
            json.dump(ch_names, f)
            

args = PreprocessArgs()
for task in ['read', 'imagine']:
    group_data(args, task)
