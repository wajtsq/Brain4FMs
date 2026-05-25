import os
import json
import torch
from torch import nn
from argparse import Namespace

from model.model_config import ModelPathArgs
from model.BrainOmni.brainomni.model import BrainOmni
from model.BrainOmni.braintokenizer.model import BrainTokenizer


# load braintokenizer model
def get_braintokenizer(ckpt_path) -> BrainTokenizer:
    model_config_path = os.path.join(ckpt_path, "model_cfg.json")
    with open(model_config_path) as f:
        model_config = json.load(f)
    model = BrainTokenizer(**model_config)
    checkpoint = torch.load(
        os.path.join(ckpt_path, "BrainTokenizer.pt"), map_location="cpu",weights_only=True
    )
    model.load_state_dict(checkpoint, strict=False)
    return model


# load brainomni model
def get_brainomni(args, ckpt_path) -> BrainOmni:
    model_config_path = os.path.join(ckpt_path, "model_cfg.json")
    with open(model_config_path) as f:
        model_config = json.load(f)
    model = BrainOmni(**model_config)
    checkpoint = torch.load(os.path.join(ckpt_path, "BrainOmni.pt"), map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    # when using omni, freeze tokenizer
    if args.exp_id != '-5':
        for p in model.tokenizer.parameters():
            p.requires_grad = False
    return model, model.lm_dim


class BrainOmni_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.final_dim = 4*512
        args.drop_unknown = False
        return args

    @staticmethod
    def clsf_loss_func(args, model):
        if args.n_class != 2:
            ce_weight = [1.0 for _ in range(args.n_class)]
        else:
            ce_weight = [0.1, 1]
        print(f'CrossEntropy loss weight = {ce_weight} = {args.weights[1]/args.weights[0]:.2f}')
        return nn.CrossEntropyLoss(torch.tensor(args.weights, dtype=torch.float32, device=torch.device(args.gpu_id)))

    @staticmethod
    def optimizer(args, model, clsf):
        return torch.optim.AdamW([
            {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.model_lr},
            {'params': list(clsf.parameters()), 'lr': args.clsf_lr}
        ],
            betas=(0.9, 0.99), eps=1e-8,
        )

    @staticmethod
    def scheduler(optimizer):
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)


class BrainOmni_Main(nn.Module):
    def __init__(self, args: Namespace,):
        super(BrainOmni_Main, self).__init__()
        self.model, n_dim = get_brainomni(args, ModelPathArgs.BrainOmni_path)
        self.class_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.LazyLinear(n_dim),
            nn.SELU(),
            nn.Linear(n_dim, args.n_class),
        )

    def forward(self, x, pos, sensor_type):
        input_dict = {}
        input_dict['x'] = x
        input_dict['pos'] = pos
        input_dict['sensor_type'] = sensor_type
        emb, indices = self.model.encode(**input_dict)  # B C W D
        emb = emb.mean(2)
        emb = emb.contiguous().view(emb.shape[0], -1)
        
        logits = self.class_head(emb)
        return emb, logits, indices

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y, pos, sensor_type = data_packet
        bsz, ch_num, N = x.shape 
        emb, logit, emb_id = model(x, pos=pos, sensor_type=sensor_type)
        
        if args.run_mode == 'test':
            return logit, y
        elif args.run_mode == 'prototype':
            return emb, logit, y
        else:
            loss = loss_func(logit, y)
            return loss, logit, y
        
    staticmethod
    def codebook_weight(self, model):
        weight = model.model.tokenizer.quantizer.rvq.codebooks
        return weight
