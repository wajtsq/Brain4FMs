from collections import OrderedDict

import torch
from torch import nn, optim
from argparse import Namespace
from einops import rearrange

from utils.data_info import data_info_dict
from model.BIOT.model.biot import BIOTClassifier
from model.pre_cnn import ConvNet
from model.model_config import ModelPathArgs


class BIOT_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):

        args.final_dim = 1024
        args.tune_a_part = True
        return args

    @staticmethod
    def clsf_loss_func(args, model):
        if args.weights is None:
            if args.n_class != 2:
                ce_weight = [1.0 for _ in range(args.n_class)]
            else:
                ce_weight = [0.3, 1]
        else:
            ce_weight = args.weights
        print(f'CrossEntropy loss weight = {ce_weight} = {ce_weight[1]/ce_weight[0]:.2f}')
        return nn.CrossEntropyLoss(torch.tensor(ce_weight, dtype=torch.float32, device=torch.device(args.gpu_id)))
        # return nn.CrossEntropyLoss()

    @staticmethod
    def optimizer(args, model, clsf):
        return torch.optim.Adam([
                # {'params': list(model.cnn.parameters()), 'lr': args.clsf_lr},
                {'params': list(model.model_clsf.parameters()), 'lr': args.model_lr},
                # {'params': list(clsf.parameters()), 'lr': args.clsf_lr},
        ],
            betas=(0.9, 0.99), eps=1e-8,
        )

    @staticmethod
    def scheduler(optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)



class BIOT(nn.Module):
    def __init__(self, args: Namespace,):
        super(BIOT, self).__init__()

        self.cnn = ConvNet(num_inputs=1, num_channels=[18])
        self.model_clsf = self.load_pretrained_weights(args)
        if args.freeze:
           self.model_clsf = self.freeze_part(args, self.model_clsf)

    def forward(self, x):
        bsz, ch_num, seq_len, patch_len = x.shape

        # Any channel number and any sequence length can be read, but the maximum number of channels can be 18
        if ch_num > 18:
            x = rearrange(x, 'b c s p -> (b s p) 1 c')
            x = self.cnn(x)
            x = torch.mean(x, dim=-1)
            x = x.reshape(bsz, seq_len, patch_len, 18)
            x = rearrange(x, 'b s p c -> b c (s p)')
        else:
            x = rearrange(x, 'b c s p -> b c (s p)')

        emb, logit = self.model_clsf(x)
        return emb, logit

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        bsz, ch_num, N = x.shape
        if N % args.patch_len != 0:
            args.seq_len = int(N // args.patch_len)
            x = x[:, :, :args.seq_len*args.patch_len]
        x = x.reshape(bsz, ch_num, -1, args.patch_len)

        emb, logit = model(x)

        if args.run_mode == 'test':
            return logit, y
        elif args.run_mode == 'prototype':
            return emb, logit, y
        else:
            loss = loss_func(logit, y)
            return loss, logit, y 

    @staticmethod
    def load_pretrained_weights(args):
        pretrained_model_path = ModelPathArgs.BIOT_path
        biot_classifier = BIOTClassifier(
            emb_size=256,
            heads=8,
            depth=4,
            n_classes=args.n_class,
            n_fft=200,
            hop_length=100,
            n_channels=18,  # here is 18
        )
        biot_classifier.biot.load_state_dict(torch.load(pretrained_model_path, map_location=f'cuda:{args.gpu_id}'))
        return biot_classifier

    @staticmethod
    def freeze_part(args, model_clsf):
        if args.tune_a_part:
            for param in model_clsf.parameters():
                param.requires_grad = False
            for param in model_clsf.classifier.parameters():
                param.requires_grad = True

        return model_clsf
