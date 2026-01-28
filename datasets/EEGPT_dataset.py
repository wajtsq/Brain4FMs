import numpy as np
import torch
import os
import json
from torch.utils.data import Dataset, DataLoader

from data_preprocess.utils import _std_data_segment
from model.EEGPT.Modules.models.EEGPT_mcae_finetune import CHANNEL_DICT

class EEGPTDataset(Dataset):
    def __init__(self, args, x, y):
        # x: (seq_num, ch_num, N)
        # y: (seq_num, )   
        if isinstance(x, dict):
            x = x['x']
        channel_path = os.path.join(args.full_data_path, 'channels_lst.json')
        if os.path.exists(channel_path):
            with open(channel_path, 'r') as f:
                channels_names = json.load(f)
            channels_names = [name.split(' ')[-1].split('-')[0].upper() for name in channels_names]
            keep_idx = [i for i, n in enumerate(channels_names) if n in CHANNEL_DICT]
            if len(keep_idx) == 0:
                raise ValueError("No matching channels found in CHANNEL_DICT!")
            
            if len(keep_idx) <= args.cnn_in_channels:
                keep_idx = torch.tensor(keep_idx, dtype=torch.long)
                x = x[:, keep_idx, :]
        
        if len(x.shape) < 3:
            x = np.expand_dims(x, axis=1)
        self.seq_num, self.ch_num, N = x.shape
        # args.cnn_in_channels = self.ch_num
        x = _std_data_segment(x)    # time level normalization

        self.x = x
        self.y = y

        self.nProcessLoader = args.n_process_loader
        self.reload_pool = torch.multiprocessing.Pool(self.nProcessLoader)


    def __getitem__(self, index):
        return self.x[index, :, :], \
               self.y[index,]

    def __len__(self):
        return self.seq_num

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          drop_last=False, pin_memory=True,
                          shuffle=shuffle)
