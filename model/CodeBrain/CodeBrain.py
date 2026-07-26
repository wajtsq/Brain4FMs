from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch
from torch import nn, optim
import torch.nn.functional as F

from model.CodeBrain.SSSM import SSSM
from model.ch_aggr_clsf import time_bandpower
from model.model_config import ModelPathArgs


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "model", "module"):
            nested = raw.get(key)
            if isinstance(nested, dict):
                raw = nested
                break
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported CodeBrain checkpoint format at {path}")

    state = {}
    for key, value in raw.items():
        if not torch.is_tensor(value):
            continue
        clean_key = key
        for prefix in ("module.", "backbone.", "model."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        state[clean_key] = value
    return state


class CodeBrain_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.final_dim = 200
        args.tune_a_part = True
        args.codebrain_patch_len = 200
        args.codebrain_n_layer = 8
        args.codebrain_dropout = 0.1
        args.codebrain_codebook_size_t = 4096
        args.codebrain_codebook_size_f = 4096
        return args

    @staticmethod
    def clsf_loss_func(args, model):
        if args.weights is None:
            return nn.CrossEntropyLoss()
        device = getattr(args, "device", None)
        if device is None:
            device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
        return nn.CrossEntropyLoss(
            torch.tensor(args.weights, dtype=torch.float32, device=device)
        )

    @staticmethod
    def optimizer(args, model, clsf):
        return optim.AdamW([
            {"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": args.model_lr},
            {"params": list(clsf.parameters()), "lr": args.clsf_lr},
        ], betas=(0.9, 0.99), eps=1e-8)

    @staticmethod
    def scheduler(optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)


class CodeBrain(nn.Module):
    """CodeBrain EEGSSM backbone adapted to Brain4FMs DefaultDataset batches."""

    def __init__(self, args: Namespace):
        super().__init__()
        self.final_dim = args.final_dim
        self.patch_len = getattr(args, "codebrain_patch_len", 200)
        self.pretrained_path = ModelPathArgs.CodeBrain_path
        self.tokenizer_path = ModelPathArgs.CodeBrain_tokenizer_path
        self.backbone = SSSM(
            in_channels=200,
            res_channels=200,
            skip_channels=200,
            out_channels=200,
            num_res_layers=getattr(args, "codebrain_n_layer", 8),
            diffusion_step_embed_dim_in=200,
            diffusion_step_embed_dim_mid=200,
            diffusion_step_embed_dim_out=200,
            s4_lmax=570,
            s4_d_state=64,
            s4_dropout=getattr(args, "codebrain_dropout", 0.1),
            s4_bidirectional=True,
            s4_layernorm=True,
            codebook_size_t=getattr(args, "codebrain_codebook_size_t", 4096),
            codebook_size_f=getattr(args, "codebrain_codebook_size_f", 4096),
            if_codebook=False,
        )
        self.loaded_pretrained_keys = 0
        self.missing_pretrained_keys: list[str] = []
        self.unexpected_pretrained_keys: list[str] = []
        self._try_load_pretrained()
        if getattr(args, "freeze", False):
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _try_load_pretrained(self) -> None:
        checkpoint_path = Path(self.pretrained_path)
        if not checkpoint_path.exists():
            print(f"CodeBrain checkpoint not found: {checkpoint_path}")
            return
        state = _load_state_dict(str(checkpoint_path))
        current = self.backbone.state_dict()
        matched = {
            key: value
            for key, value in state.items()
            if key in current and current[key].shape == value.shape
        }
        incompatible = self.backbone.load_state_dict(matched, strict=False)
        self.loaded_pretrained_keys = len(matched)
        self.missing_pretrained_keys = list(incompatible.missing_keys)
        self.unexpected_pretrained_keys = list(incompatible.unexpected_keys)
        print(f"CodeBrain loaded {len(matched)} tensors from {checkpoint_path}")

    def _prepare_patches(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        bsz, ch_num, time_len = x.shape
        seq_len = max(1, (time_len + self.patch_len - 1) // self.patch_len)
        target_len = seq_len * self.patch_len
        if time_len < target_len:
            x = F.pad(x, (0, target_len - time_len))
        else:
            x = x[..., :target_len]
        return x.reshape(bsz, ch_num, seq_len, self.patch_len).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_patches(x)
        feats = self.backbone(x)
        if feats.ndim == 2:
            return feats
        if feats.ndim == 3:
            return feats.mean(dim=1)
        if feats.ndim == 4:
            return feats.mean(dim=(1, 2))
        raise ValueError(f"Unexpected CodeBrain feature shape: {tuple(feats.shape)}")

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        emb = model(x)
        logit = clsf(emb)

        if args.run_mode == "test":
            return logit, y
        if args.run_mode in {"exp1", "exp3", "prototype"}:
            if args.run_mode == "exp3":
                y = time_bandpower(args, x, args.sfreq)
            return emb, logit, y
        loss = loss_func(logit, y)
        return loss, logit, y
