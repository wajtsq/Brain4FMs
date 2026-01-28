import numpy as np
import torch
from einops import rearrange
from torch import nn
from transformers import AutoModel
from argparse import Namespace
import torch.nn.functional as F
from model.model_config import ModelPathArgs


class REVE_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.final_dim = 128
        patch_size, overlap_size = 200, 20
        H = (args.seq_len*patch_size - patch_size) // (patch_size - overlap_size) + 1
        args.dim = int(H * args.cnn_in_channels * 512)
        return args

    @staticmethod
    def clsf_loss_func(args, model=None):
        return nn.CrossEntropyLoss(torch.tensor(args.weights, dtype=torch.float32, 
                                                device=torch.device(args.gpu_id)))


    @staticmethod
    def optimizer(args, model, clsf=None):
        return torch.optim.AdamW([
            {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.model_lr},
            {'params': list(clsf.parameters()), 'lr': args.clsf_lr}
        ],
            betas=(0.9, 0.95), eps=1e-6,
        )

    @staticmethod
    def scheduler(optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)


class REVE(nn.Module):
    def __init__(self, args): 
        super().__init__()        
        self.model = AutoModel.from_pretrained(ModelPathArgs.REVE_path, trust_remote_code=True, torch_dtype="auto")
        self.model.final_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.RMSNorm(args.dim),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(args.dim, args.n_class),
        )
    
    def forward(self, eeg, pos, return_output=False):
        logit = self.model(eeg, pos, return_output=return_output)
        return logit
    
    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y, pos = data_packet
        B, C, T = x.shape
        T_new = int(T * 200 / args.sfreq)
        x = F.interpolate(
            x,
            size=T_new,
            mode="linear",
            align_corners=False
        )
        logit = model(x, pos)
            
        if args.run_mode == 'test':
            return logit, y
        else:
            loss = loss_func(logit, y)
            return loss, logit, y