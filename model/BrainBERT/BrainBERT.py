import torch
from omegaconf import OmegaConf
from torch import nn
from argparse import Namespace
import torchaudio
from einops import rearrange, reduce

from utils.data_info import data_info_dict
from model.BrainBERT.models.masked_tf_model import MaskedTFModel
from model.pre_cnn import ConvNet
from model.model_config import ModelPathArgs



def _torch_zscore(x, dim: int):
    """
    Z-score normalization along specified dim using torch.
    """
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, unbiased=False, keepdim=True)
    std_clamped = std.clone()
    std_clamped[std_clamped == 0] = 1.0
    return (x - mean) / std_clamped

class STFTTorchaudioPreprocessor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # torchaudio's Spectrogram provides magnitude
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=cfg.nperseg,
            win_length=cfg.nperseg,
            hop_length=cfg.nperseg - cfg.noverlap,
            normalized=False,
            center=True,
            pad_mode='reflect'
        )

    def forward(self, wav: torch.Tensor):
        # wav: (..., time) or (batch, time)
        # Ensure shape (batch, time)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # Compute STFT: returns complex tensor shape (batch, freq, time, complex=2)
        spec = self.spectrogram(wav)
        # Truncate frequency channels if needed
        if self.cfg.freq_channel_cutoff is not None:
            spec = spec[:, : self.cfg.freq_channel_cutoff, :]

        # Normalization
        if self.cfg.normalizing == 'zscore':
            # zscore over time axis
            spec = _torch_zscore(spec, dim=-1)
            # Avoid all-zero bands
            if spec.std() == 0:
                spec = torch.ones_like(spec)
            # remove edge frames
            spec = spec[..., 5:-5]
        elif self.cfg.normalizing == 'db':
            spec = torch.log(spec + 1e-6)
        # Replace any NaNs
        spec = torch.nan_to_num(spec, nan=0.0)
        # Transpose to match original: (batch, time, freq)
        output = spec.transpose(1, 2)
        return output


class BrainBERT_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.final_dim = 768
        args.nperseg = 100
        args.domain = 'freq'
        args.n_fft = 100
        args.noverlap = 80
        args.freq_channel_cutoff = 40
        args.normalizing = 'zscore'
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
            {'params': filter(lambda p: p.requires_grad, model.cnn.parameters()), 'lr': args.clsf_lr},  # a large lr
            {'params': filter(lambda p: p.requires_grad, model.enc.parameters()), 'lr': args.model_lr},
            {'params': list(clsf.parameters()), 'lr': args.clsf_lr}
        ],
            betas=(0.9, 0.99), eps=1e-8,
        )

    @staticmethod
    def scheduler(optimizer):
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)




class BrainBERT(nn.Module):
    def __init__(self, args: Namespace,):
        super(BrainBERT, self).__init__()
        # in_patch_len = 40
        # self.cnn = ConvNet(num_inputs=1, num_channels=[in_patch_len])
        self.cnn = STFTTorchaudioPreprocessor(args)
        self.enc = self.load_pretrained_weights(args)
        # self.linear_out = nn.Linear(in_features=args.final_dim, out_features=args.n_class)
        if args.freeze:
            for name, param in self.enc.named_parameters():
                param.requires_grad = False

    def forward(self, x, run_mode):
        bsz, ch_num, seq_len, patch_len = x.shape
        # x = x.reshape(bsz*ch_num*seq_len, 1, patch_len)
        x = x.reshape(bsz*ch_num, seq_len*patch_len)
        emb = self.cnn(x)
        # emb = torch.mean(emb, dim=-1).reshape(bsz*ch_num, seq_len, -1)
        if run_mode == 'exp2':
            out = self.enc.forward(emb, intermediate_rep=False)
            out = out.transpose(1,2)
            emb = emb.transpose(1,2)
            return emb, out
        else:
            emb = self.enc.forward(emb, intermediate_rep=True)
        emb = rearrange(emb, '(b c) s d -> b c s d', c=ch_num)
        emb = reduce(emb, 'b c s d -> b c d', 'mean')
        # emb = emb.reshape(bsz, ch_num, seq_len, -1)
        # emb = torch.mean(emb, dim=2)
        return emb

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        bsz, ch_num, N = x.shape
        x = x[..., :args.seq_len*args.patch_len]
        x = x.reshape(bsz, ch_num, -1, args.patch_len)

        emb = model(x, args.run_mode)
        
        middle = int(emb.shape[1]/2)
        emb = emb[:,middle-5:middle+5].mean(axis=1)
        logit = clsf(emb)
        
        if args.run_mode == 'test':
            return logit, y
        elif args.run_mode == 'prototype':
            return emb, logit, y
        else:
            loss = loss_func(logit, y)
            return loss, logit, y
        

    @staticmethod
    def load_pretrained_weights(args, ):
        def _build_model(cfg, gpu_id):
            ckpt_path = cfg.upstream_ckpt
            init_state = torch.load(ckpt_path, map_location=f'cuda:{gpu_id}', weights_only=False)
            upstream_cfg = init_state["model_cfg"]

            # model = models.build_model(upstream_cfg)
            model = MaskedTFModel()
            model.build_model(upstream_cfg, )

            model.load_state_dict(init_state['model'])
            return model

        cfg = OmegaConf.create({"upstream_ckpt": ModelPathArgs.BrainBERT_path})
        model = _build_model(cfg, args.gpu_id).to(args.gpu_id)
        return model
