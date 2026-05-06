import numpy as np
import torch
import os
import re
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel

from data_preprocess.utils import _std_data_segment
from model.model_config import ModelPathArgs


def _canonicalize_ch_name(name: str) -> str:
    n = name.strip()
    if '-' in n:
        n = n.split('-')[0]
    n = re.sub(r"^(EEG|eeg)\s+", "", n)
    n = re.sub(r"[-_](REF|ref|AVG|avg|A1|A2|M1|M2|LE|RE)$", "", n)
    n = re.sub(r"\(.*\)", "", n)
    n = n.replace(" ", "")
    n = n.upper()
    return n


def get_safe_global_pos(pos_bank, channel_names):
    total_pos = list(pos_bank.position_names)
    name2idx = {name: i for i, name in enumerate(total_pos)}
    pos_weight = pos_bank.embedding  # [N_pos, 3]

    global_pos = []
    for ch in channel_names:
        if ch in name2idx:
            idx = name2idx[ch]
            global_pos.append(pos_weight[idx])
        else:
            global_pos.append(torch.zeros(3))
    return torch.stack(global_pos, dim=0)


class REVEDataset(Dataset):
    def __init__(self, args, x, y):
        # x: (seq_num, ch_num, N)
        # y: (seq_num, )
        ch_pos, perm = None, None
        if isinstance(x, dict):
            ch_pos = x.get("pos", None)
            perm = x.get("perm", None)
            x = x['x']
        
        pos_bank = AutoModel.from_pretrained(ModelPathArgs.REVE_pos_path, 
                                             trust_remote_code=True, torch_dtype="auto")
        
        channel_path = os.path.join(args.full_data_path, 'channels_lst.json')
        if os.path.exists(channel_path):
            with open(channel_path, 'r') as f:
                channel_names = json.load(f)
            channel_names = [_canonicalize_ch_name(ch) for ch in channel_names]
            if perm is not None:
                channel_names = [channel_names[i] for i in perm]

            global_pos = get_safe_global_pos(pos_bank, channel_names)
            if ch_pos is not None:
                N, C = ch_pos.shape
                pos_batch = np.zeros((N, C, 3), dtype=np.float32)
                for i in range(N):
                    for c in range(C):
                        idx = int(ch_pos[i, c])
                        if idx < 0 or idx >= global_pos.shape[0]:
                            pos_batch[i, c] = 0.0
                        else:
                            pos_batch[i, c] = global_pos[idx]

                self.pos = torch.from_numpy(pos_batch)          # (N, C, 3)
                
            else:
                self.pos = np.stack(global_pos, axis=0).astype(np.float32)  # (C,3)
                self.pos = torch.from_numpy(self.pos)
        else:
            self.pos = torch.from_numpy(np.zeros((x.shape[0], x.shape[1], 3), dtype=np.float32))
                
        self.seq_num, self.ch_num, N = x.shape
        x = _std_data_segment(x)    # time level normalization

        self.x = x
        self.y = y
            
        self.nProcessLoader = args.n_process_loader
        self.reload_pool = torch.multiprocessing.Pool(self.nProcessLoader)


    def __getitem__(self, idx):
        x = self.x[idx]  # shape: (ch_num, N)
        y = self.y[idx]
        pos = None
        if self.pos.dim() == 2:
            pos = self.pos  # (C,3)
        elif self.pos.dim() == 3:
            pos = self.pos[idx]  # (C,3)
        else:
            raise ValueError(f"Unexpected self.pos dim={self.pos.dim()}, shape={tuple(self.pos.shape)}")
        pos = pos.to(torch.float32)

        return x, y, pos


    def __len__(self):
        return self.seq_num

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          drop_last=False, pin_memory=True,
                          shuffle=shuffle)
