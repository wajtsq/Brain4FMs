import torch
from torch import nn
import pytorch_lightning as pl
from functools import partial
from argparse import Namespace
import os
import numpy as np
import json
from model.pre_cnn import ChannelConverter
from model.EEGPT.utils import temporal_interpolation
from model.EEGPT.Modules.Transformers.pos_embed import create_1d_absolute_sin_cos_embedding
from model.EEGPT.Modules.models.EEGPT_mcae import EEGTransformer
from model.EEGPT.Modules.models.EEGPT_mcae_finetune import EEGPTClassifier, CHANNEL_DICT
from model.EEGPT.Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from model.model_config import ModelPathArgs


class EEGPT_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.final_dim = 64
        args.channel_dict = 62
        return args

    @staticmethod
    def clsf_loss_func(args, model):
        # return torch.nn.CrossEntropyLoss()
        if args.weights is None:
            ce_weight = [1.0 for _ in range(args.n_class)]
        else:
            ce_weight = args.weights
        print(f'CrossEntropy loss weight = {ce_weight} = {ce_weight[1]/ce_weight[0]:.2f}')
        return nn.CrossEntropyLoss(torch.tensor(ce_weight, dtype=torch.float32, device=torch.device(args.gpu_id)))

    @staticmethod
    def optimizer(args, model, clsf):
        return torch.optim.AdamW([
            {'params': list(model.eegpt.parameters()), 'lr': args.model_lr},
            {'params': list(clsf.parameters()), 'lr': args.clsf_lr},
        ],
            betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01
        )

    @staticmethod
    def scheduler(optimizer):
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)


class LitEEGPTCausal(pl.LightningModule):

    def __init__(self, args: Namespace):
        super().__init__()    
        self.chans_num = args._channels
        self.total_len = args.patch_len*args.seq_len
        # init model
        target_encoder = EEGTransformer(
            img_size=[self.chans_num, self.total_len],        # 256*30
            patch_size=32*2,
            # patch_stride = 32,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
            
        self.target_encoder = target_encoder
        self.chans_id = None
        channel_path = os.path.join(args.full_data_path, 'channels_lst.json')
        if os.path.exists(channel_path):
            with open(channel_path, 'r') as f:
                use_channels_names = json.load(f)
            self.chans_id       = target_encoder.prepare_chan_ids(use_channels_names)
        
        # -- load checkpoint
        load_path = ModelPathArgs.EEGPT_path
        pretrain_ckpt = torch.load(load_path, map_location=f'cuda:{args.gpu_id}')
        
        target_encoder_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]]=v
        
                
        self.target_encoder.load_state_dict(target_encoder_stat)

        self.chan_conv       = Conv1dWithConstraint(self.chans_num, self.chans_num, 1, max_norm=1)
        # self.chan_conv       = Conv1dWithConstraint(2, self.chans_num, 1, max_norm=1)
        
        self.linear_probe1   = LinearWithConstraint(2048, 64, max_norm=1)
        self.drop            = torch.nn.Dropout(p=0.50)        
        self.decoder         = torch.nn.TransformerDecoder(
                                    decoder_layer=torch.nn.TransformerDecoderLayer(64, 4, 64*4, activation=torch.nn.functional.gelu, batch_first=False),
                                    num_layers=4
                                )
        self.cls_token =        torch.nn.Parameter(torch.rand(1,1,64)*0.001, requires_grad=True)
    
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity = True
        
        
    def forward(self, x):
        B, C, T = x.shape
        x = temporal_interpolation(x, self.total_len)
        x = self.chan_conv(x)
        self.target_encoder.eval()
        if self.chans_id is not None:
            z = self.target_encoder(x, self.chans_id.to(x))
        else:
            z = self.target_encoder(x)
        h = z.flatten(2)
        
        h = self.linear_probe1(self.drop(h))
        pos = create_1d_absolute_sin_cos_embedding(h.shape[1], dim=64)
        h = h + pos.repeat((h.shape[0], 1, 1)).to(h)
        
        h = torch.cat([self.cls_token.repeat((h.shape[0], 1, 1)).to(h.device), h], dim=1)
        h = h.transpose(0,1)
        h = self.decoder(h, h)[0,:,:]
        
        return x, h


class EEGPT(nn.Module):
    def __init__(self, args: Namespace,):
        super(EEGPT, self).__init__()
        # self.chan_dict = 62
        # if args.cnn_in_channels > self.chan_dict:
        #     self.ch_cnn = ChannelConverter(in_channels=args.cnn_in_channels, out_channels=62)
        #     args.cnn_in_channels = self.chan_dict
        # self.eegpt = LitEEGPTCausal(args)
        channel_path = os.path.join(args.full_data_path, 'channels_lst.json')
        if os.path.exists(channel_path):
            with open(channel_path, 'r') as f:
                channels_names = json.load(f)
            use_channels_names = [name.split(' ')[-1].split('-')[0] for name in channels_names]
            use_channels_names = [name.upper() for name in use_channels_names if name.upper() in CHANNEL_DICT][:args.cnn_in_channels]
            args._channels = len(use_channels_names)
        else:
            use_channels_names = None
            args._channels = args.cnn_in_channels
        self.eegpt = EEGPTClassifier(num_classes=args.n_class, in_channels=args._channels, 
                                     use_channels_names=use_channels_names, img_size=[args._channels,2000],
                                     use_chan_conv=True, use_predictor=True)
        load_path = ModelPathArgs.EEGPT_path
        pretrain_ckpt = torch.load(load_path, map_location=f'cuda:{args.gpu_id}', weights_only=False)['state_dict']        
        
        current_channel = self.eegpt.target_encoder.chan_embed.weight.shape[0]
        if current_channel != args.channel_dict:
            print(f"⚠️ Detected channel mismatch: pretrain {current_channel} vs current {args.channel_dict}")
            chan_weight = pretrain_ckpt['target_encoder.chan_embed.weight']
            
            mean_row = chan_weight.mean(dim=0, keepdim=True)
            new_weight = mean_row.repeat(current_channel-args.channel_dict, 1)
            new_weight = torch.cat((chan_weight, new_weight))
            pretrain_ckpt['target_encoder.chan_embed.weight'] = new_weight

        self.eegpt.load_state_dict(pretrain_ckpt, strict=False)
        # self.cls = LinearWithConstraint(args.final_dim, args.n_class, max_norm=0.25)
        if args.freeze:
            for name, param in self.eegpt.named_parameters():
                if 'head.' in name:
                    continue
                param.requires_grad = False
        
    def forward(self, x):
        bsz, ch_num, N = x.shape 
        # if ch_num > self.chan_dict:
        #     x = self.ch_cnn(x)
        h, logits = self.eegpt(x)
        # h = self.cls(h)
        return h, logits
    
    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        bsz, ch_num, N = x.shape    
        x = x[..., :args.seq_len*args.patch_len]        
        embs, logit = model(x)
        # emb = emb.unsqueeze(1)
        # logit = clsf(emb)

        if args.run_mode == 'test':
            return logit, y
        else:
            loss = loss_func(logit, y)
            return loss, logit, y
    