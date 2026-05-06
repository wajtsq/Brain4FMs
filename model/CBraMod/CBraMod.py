import torch
import librosa
from torch import nn, optim
from argparse import Namespace
from model.pre_cnn import ConvNet
from einops import rearrange

from model.CBraMod.models.cbramod import CBraModEnc
from model.model_config import ModelPathArgs


class CBraMod_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.final_dim = args.seq_len*200
        args.mask_ratio = 0.5
        args.domain = 'time'
        return args
    
    @staticmethod
    def clsf_loss_func(args, model):
        if args.n_class == 2:
            ce_weight = [0.3, 1.0]
        else: 
            ce_weight = [1.0 for _ in range(args.n_class)]
        print(f'CrossEntropy loss weight = {ce_weight} = {args.weights[1]/args.weights[0]:.2f}')
        return nn.CrossEntropyLoss(torch.tensor(args.weights, dtype=torch.float32, device=torch.device(args.gpu_id)))       


    @staticmethod
    def optimizer(args, model, clsf):
        return torch.optim.AdamW([
                {'params': list(model.parameters()), 'lr': args.model_lr},
                # {'params': list(model.classifier.parameters()), 'lr': args.clsf_lr},
                {'params': list(clsf.parameters()), 'lr': args.clsf_lr},
        ],
            betas=(0.9, 0.99), eps=1e-8,
        )

    @staticmethod
    def scheduler(optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)
    
def generate_mask(bz, ch_num, patch_num, mask_ratio, device):
    mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long, device=device)
    mask = mask.bernoulli_(mask_ratio)
    return mask
    
class CBraMod(nn.Module):
    def __init__(self, args: Namespace):
        super(CBraMod, self).__init__()
        self.backbone = self.load_model(args)
        if args.freeze:
            self.backbone = self.freeze_part(self.backbone)


    def forward(self, x, mask=None):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x, mask)
        out = feats.contiguous().view(bz, ch_num, seq_len*200)
        # out = self.classifier(out)
        return out


    @staticmethod
    def load_model(args):
        model_clsf = CBraModEnc(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
        path = ModelPathArgs.CBraMod_path
        map_location = torch.device(f'cuda:{args.gpu_id}')
        model_clsf.load_state_dict(torch.load(path, map_location=map_location))
        model_clsf.proj_out = nn.Identity()
        return model_clsf
    
    
    @staticmethod
    def freeze_part(model_clsf):
        for param in model_clsf.parameters():
            param.requires_grad = False

        return model_clsf
    
    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        bsz, ch_num, N = x.shape
        x = x[..., : args.seq_len*args.patch_len]
        mask = None
        # resampling
        if args.sfreq != 200:
            x = x.reshape(bsz*ch_num, -1)
            x = x.to('cpu').numpy()
            x = librosa.resample(x, orig_sr=args.sfreq, target_sr=200)
            x = torch.from_numpy(x).to(f'cuda:{args.gpu_id}')            
        args.patch = 200
        x = x.reshape(bsz, ch_num, -1, args.patch)
        bsz, ch_num, patch_num, patch_size = x.shape
        
        emb = model(x, mask)
        emb = emb.mean(dim=1)
        logit = clsf(emb)

        if args.run_mode == 'test':
            return logit, y
        elif args.run_mode == 'prototype':
            return emb, logit, y
        else:
            loss = loss_func(logit, y)
            return loss, logit, y
