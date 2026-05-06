import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dropout):
        super(ConvBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.maxpool1 = nn.MaxPool1d(4, 2)

        self.net = nn.Sequential(self.conv1, self.relu1, self.maxpool1, self.dropout1)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        return out


class ConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, dropout=0.2):
        super(ConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            kernel_size = 2 ** (i + 2)
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [ConvBlock(in_channels, out_channels, kernel_size, stride=1, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ChannelConverter(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(ChannelConverter, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        # (bsz, ch_num, seq_len, patch_len)
        out = self.conv(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        return out