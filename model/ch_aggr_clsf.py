import torch
from torch import nn
from argparse import Namespace
import torch.nn.functional as F
from typing import Optional


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class ChannelCNN(torch.nn.Module):
    def __init__(self, args: Namespace):
        super(ChannelCNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=args.cnn_in_channels, out_channels=args.final_dim // 2, kernel_size=args.cnn_kernel_size),
            nn.BatchNorm1d(args.final_dim // 2),
            nn.Conv1d(in_channels=args.final_dim // 2, out_channels=args.final_dim, kernel_size=args.cnn_kernel_size),
            nn.BatchNorm1d(args.final_dim),
            nn.Conv1d(in_channels=args.final_dim, out_channels=args.final_dim, kernel_size=args.cnn_kernel_size),
        )

    def forward(self, x):
        bsz, ch_num, emb_dim = x.shape
        if ch_num != 1:
            x = self.cnn(x)                     # (bsz, final_dim, --)
            emb = torch.mean(x, dim=-1)   # (bsz, final_dim)
        else:
            emb = torch.squeeze(x, 1) # (bsz, emb_dim)
        return emb


class LinearClsf(torch.nn.Module):
    def __init__(self, args: Namespace):
        super(LinearClsf, self).__init__()
        # classifier
        self.mlp = nn.Sequential(
            nn.Linear(args.final_dim, args.final_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(args.final_dim // 2, args.n_class),
        )
        self.apply(_weights_init)

    def forward(self, x):
        logit = self.mlp(x)
        return logit  # (bsz, num_class)
    

class ChannelAggrClsf(torch.nn.Module):
    def __init__(self, args: Namespace):
        super(ChannelAggrClsf, self).__init__()

        # channel aggregation
        self.cnn = ChannelCNN(args)

        # classifier
        self.mlp = nn.Sequential(
            nn.Linear(args.final_dim, args.final_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(args.final_dim // 2, args.n_class),
        )
        self.apply(_weights_init)

    def forward(self, x):
        # x: (bsz, ch_num, seq_len, patch_len)
        x = self.cnn(x)     # x: (bsz, final_dim)

        # logit = self.softmax(self.mlp(x)) # MUST NOT softmax IN THE CLSF
        logit = self.mlp(x)

        return logit  # (bsz, num_class)
