import numpy as np
import torch
import os
import re
import json
import mne
from torch.utils.data import Dataset, DataLoader

from data_preprocess.utils import _std_data_segment

SENSOR_TYPE_DICT = {"EEG": 0, "MAG": 1, "GRAD": 2}
DOWNSTREAM_DTYPE = torch.bfloat16

def _canonicalize_ch_name(name: str) -> str:
    """
    Standardize the various variant channel names in the dataset 
    as much as possible into a form that matches standard_1020
    examples:
    - 'EEG Fp1-REF' / 'Fp1-REF' / 'Fp1-Avg' / 'Fp1-M1' -> 'Fp1'
    - FP1 -> Fp1
    - Remove spaces, prefixes, suffixes, parentheses, etc
    """
    n = name.strip()
    if '-' in n:
        n = n.split('-')[0]
    n = re.sub(r"^(EEG|eeg)\s+", "", n)
    n = re.sub(r"[-_](REF|ref|AVG|avg|A1|A2|M1|M2|LE|RE)$", "", n)
    n = re.sub(r"\(.*\)", "", n)
    n = n.replace(" ", "")

    # # "FP1" -> "Fp1", "CPZ" -> "Cpz"
    # # end z/Z -> 'z'
    # if len(n) > 0:
    #     m = re.match(r"^([A-Za-z]+)([0-9]*)$", n)
    #     if m:
    #         letters, digits = m.group(1), m.group(2)
    #         letters = letters.capitalize()  # Fp, Cp, Af, Oz...
    #         letters = re.sub(r"Z$", "z", letters)
    #         n = letters + digits
    n = n.upper()
    return n


def _build_global_pos_table(channel_name, drop_unknown=True):
    """
    return:
      global_pos6: (C_global, 6) float32
      global_mask: (C_global,) bool
      unknown: list[(idx, raw_name, canonical)]
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_pos_dict_ori = montage.get_positions()["ch_pos"]
    ch_pos_dict = {}
    for key_ in ch_pos_dict_ori.keys():
        up_key = key_.upper()
        ch_pos_dict[up_key] = ch_pos_dict_ori[key_]

    global_pos = []
    global_mask = []
    unknown = []
    for i, ch in enumerate(channel_name):
        cname = _canonicalize_ch_name(ch)
        if cname in ch_pos_dict:
            xyz = np.asarray(ch_pos_dict[cname], dtype=np.float32)
            pos6 = np.concatenate([xyz, np.zeros(3, dtype=np.float32)], axis=0)
            global_pos.append(pos6)
            global_mask.append(True)
        else:
            global_pos.append(np.zeros(6, dtype=np.float32))
            unknown.append((i, ch, cname))
            if not drop_unknown:
                global_mask.append(True)
            else:
                global_mask.append(False)

    global_pos6 = np.stack(global_pos, axis=0).astype(np.float32)
    global_mask = np.asarray(global_mask, dtype=bool)
    return global_pos6, global_mask, unknown


def get_sensor_type_mask(sensor_type: np.ndarray):
    eeg_mask = sensor_type == SENSOR_TYPE_DICT["EEG"]
    mag_mask = sensor_type == SENSOR_TYPE_DICT["MAG"]
    grad_mask = sensor_type == SENSOR_TYPE_DICT["GRAD"]
    meg_mask = mag_mask | grad_mask
    return eeg_mask, mag_mask, grad_mask, meg_mask

def normalize_pos(pos: np.ndarray, eeg_mask, meg_mask):
    if eeg_mask.any():
        eeg_mean = np.mean(pos[eeg_mask, :3], axis=0, keepdims=True)
        pos[eeg_mask, :3] -= eeg_mean
        eeg_scale = np.sqrt(3 * np.mean(np.sum(pos[eeg_mask, :3] ** 2, axis=1)))
        if eeg_scale > 1e-8:
            pos[eeg_mask, :3] /= eeg_scale
    if meg_mask.any():
        meg_mean = np.mean(pos[meg_mask, :3], axis=0, keepdims=True)
        pos[meg_mask, :3] -= meg_mean
        meg_scale = np.sqrt(3 * np.mean(np.sum(pos[meg_mask, :3] ** 2, axis=1)))
        if meg_scale > 1e-8:
            pos[meg_mask, :3] /= meg_scale
    return pos


class BrainOminiDataset(Dataset):
    def __init__(self, args, x, y):
        # x: (seq_num, ch_num, N)
        # y: (seq_num, )
        ch_pos = None
        if isinstance(x, dict):
            ch_pos = x.get("pos", None)
            x = x['x']
        # sensor_type
        self.sensor_type = torch.full(
            (x.shape[1],),
            fill_value=SENSOR_TYPE_DICT["EEG"],  # iEEG -> EEG
            dtype=torch.int32,
        )

        channel_path = os.path.join(args.full_data_path, 'channels_lst.json')
        if os.path.exists(channel_path):
            with open(channel_path, 'r') as f:
                channel_names = json.load(f)
            global_pos6, global_valid_mask, unknown = _build_global_pos_table(channel_names, drop_unknown=args.drop_unknown)
            keep_indices = [i for i, b in enumerate(global_valid_mask) if b]
            if len(keep_indices) == 0:
                raise ValueError("No channels matched standard_1020 after canonicalization.")

            if ch_pos is not None:
                N, C = ch_pos.shape
                pos_batch = np.zeros((N, C, 6), dtype=np.float32)
                channel_mask = np.ones((N, C), dtype=bool)
                sensor_type = np.full((N, C), SENSOR_TYPE_DICT["EEG"], dtype=np.int32)

                for i in range(N):
                    for c in range(C):
                        idx = int(ch_pos[i, c])
                        if idx < 0 or idx >= global_pos6.shape[0] or (not global_valid_mask[idx]):
                            pos_batch[i, c] = 0.0
                            channel_mask[i, c] = False
                        else:
                            pos_batch[i, c] = global_pos6[idx]
                            
                    eeg_mask, mag_mask, grad_mask, meg_mask = get_sensor_type_mask(sensor_type[i])
                    eeg_mask = eeg_mask & channel_mask[i]
                    pos_batch[i] = normalize_pos(pos_batch[i], eeg_mask, meg_mask)

                self.pos = torch.from_numpy(pos_batch)          # (N, C, 6)
                self.sensor_type = torch.from_numpy(sensor_type) # (N, C)
                self.channel_mask = torch.from_numpy(channel_mask)  # (N, C)
                self.unknown_channels = unknown
                self.keep_indices = list(range(x.shape[1]))
                
            else:
                self.keep_indices = keep_indices
                x = x[:, keep_indices, ...]
                pos = np.stack(global_pos6, axis=0).astype(np.float32)  # (C,6)
                                
                self.unknown_channels = unknown
                eeg_mask, mag_mask, grad_mask, meg_mask = get_sensor_type_mask(self.sensor_type.numpy())
                eeg_mask = eeg_mask & global_valid_mask
                pos = normalize_pos(pos, eeg_mask, meg_mask)
                pos = pos[keep_indices, :]
                self.pos = torch.from_numpy(pos)  # (keep_C,6)
                self.sensor_type = self.sensor_type[keep_indices]   # (keep_C)
        else:
            self.pos = torch.from_numpy(np.zeros((x.shape[0], x.shape[1], 6), dtype=np.float32))
                
        self.seq_num, self.ch_num, N = x.shape
        x = _std_data_segment(x)    # time level normalization

        self.x = x
        self.y = y
            
        self.nProcessLoader = args.n_process_loader
        self.reload_pool = torch.multiprocessing.Pool(self.nProcessLoader)


    def __getitem__(self, idx):
        """
        {
            "x": (ch_num, N)  or  (ch_num, ...)
            "pos": (ch_num, 6) or None
            "y": scalar tensor
            "ch_names": list[str]
            "sensor_type": (ch_num,) or (ch_num,) 
            "channel_mask": (ch_num,) 
        }
        """
        x = self.x[idx]  # shape: (ch_num, N)
        y = self.y[idx]
        x = x.to(torch.float32) if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32)
        
        pos = None
        if self.pos.dim() == 2:
            pos = self.pos  # (C,6)
        elif self.pos.dim() == 3:
            pos = self.pos[idx]  # (C,6)
        else:
            raise ValueError(f"Unexpected self.pos dim={self.pos.dim()}, shape={tuple(self.pos.shape)}")
        pos = pos.to(torch.float32)
        
        st = self.sensor_type
        if st.dim() == 1:
            sensor_type = st  # (C,)
        elif st.dim() == 2:
            sensor_type = st[idx]  # (C,)
        else:
            raise ValueError(f"Unexpected sensor_type dim={st.dim()}, shape={tuple(st.shape)}")

        return x, y, pos, sensor_type


    def __len__(self):
        return self.seq_num

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          drop_last=False, pin_memory=True,
                          shuffle=shuffle)
