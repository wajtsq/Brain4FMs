import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader

from data_preprocess.utils import _std_data_segment

class NeuroGPTDataset(Dataset):
    def __init__(self, args, x, y):
        # x: (seq_num, ch_num, N)
        # y: (seq_num, )
        if isinstance(x, dict):
            x = x['x']
        self.seq_num, self.ch_num, N = x.shape
        
        x = _std_data_segment(x)    # time level normalization
        self.T = int(args.seq_len)
        self.patch_len = int(args.patch_len)

        self.x = x
        self.y = y

        self.nProcessLoader = args.n_process_loader
        self.reload_pool = torch.multiprocessing.Pool(self.nProcessLoader)
        
    def _slice_to_seq(self, sig_1: torch.Tensor):
        """
        sig_1: [C, N_i]
        output:
          chunks: [T, C, patch_len]
          mask:   [T] pad=0
        """
        C, Ni = sig_1.shape
        need = self.T * self.patch_len

        # chunk number
        real_T = min(self.T, math.ceil(Ni / self.patch_len))

        if Ni >= need:
            sig = torch.tensor(sig_1[:, :need])
        else:
            pad_len = need - Ni
            pad = sig_1.new_zeros((C, pad_len))
            sig = torch.cat([sig_1, pad], dim=1)

        sig = sig.reshape(C, self.T, self.patch_len).permute(1, 0, 2).contiguous()  # [T, C, patch_len]

        # attention mask
        mask = sig.new_zeros((self.T,), dtype=torch.long)
        mask[:real_T] = 1

        return sig, mask  # [T, C, patch_len], [T]


    def __getitem__(self, index):
        sig = self.x[index] # [C, N]
        chunks, mask = self._slice_to_seq(sig)  # [T, C, patch_len], [T]
        T, C, P = chunks.shape
        inputs = chunks.contiguous()
        return inputs, mask, \
               self.y[index,]

    def __len__(self):
        return self.seq_num

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          drop_last=False, pin_memory=True, shuffle=shuffle)
