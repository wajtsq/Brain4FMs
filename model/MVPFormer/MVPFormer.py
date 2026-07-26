from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim

from model.MVPFormer.runtime import MVPFormerConfig, MVPFormerModel, WaveEncoder
from model.model_config import ModelPathArgs


def _wave_dwt_size(size_input: int, filter_len: int = 8) -> int:
    max_levels = max(1, int(torch.floor(torch.log2(torch.tensor(size_input / (filter_len - 1)))).item()))
    current = size_input
    sizes: list[int] = []
    for _ in range(max_levels):
        current = (current + 1) // 2
        sizes.append(current)
    sizes.append(current)
    return sum(sizes)


def _infer_segment_size_from_dwt_size(target_dwt_size: int, preferred: int | None = None) -> int | None:
    if preferred is not None and _wave_dwt_size(preferred) == target_dwt_size:
        return preferred
    candidates = [size for size in range(1, 10001) if _wave_dwt_size(size) == target_dwt_size]
    if not candidates:
        return None
    even_candidates = [size for size in candidates if size % 2 == 0]
    if even_candidates:
        return even_candidates[-1]
    return candidates[-1]


def _load_checkpoint_state(path: str) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "model", "module"):
            nested = raw.get(key)
            if isinstance(nested, dict):
                raw = nested
                break
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")
    state: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not torch.is_tensor(value):
            continue
        clean_key = key
        for prefix in ("state_dict.", "model.", "module."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        if clean_key.startswith("genie."):
            clean_key = "mvpformer." + clean_key[len("genie.") :]
        state[clean_key] = value
    return state


def _infer_runtime_config(state: dict[str, torch.Tensor], args: Namespace) -> dict[str, int]:
    pos_key = "mvpformer.positional_embedding.weight"
    ch_key = "mvpformer.channel_embedding.weight"
    q_key = "mvpformer.h.0.attn.q_attn.weight"
    kv_key = "mvpformer.h.0.attn.c_attn.weight"
    mlp_key = "mvpformer.h.0.mlp.gate_proj.weight"
    enc_key = "encoder.proj.weight"

    if not all(key in state for key in (pos_key, ch_key, q_key, kv_key, mlp_key, enc_key)):
        return {
            "final_dim": getattr(args, "final_dim", 768),
            "segment_size": getattr(args, "mvpformer_segment_size", 5000),
            "n_positions": getattr(args, "mvpformer_max_segments", 110),
            "n_channels": getattr(args, "mvpformer_max_channels", 128),
            "n_layer": 12,
            "n_head": 12,
            "n_head_kv": 4,
            "n_inner": 1728,
        }

    final_dim = state[q_key].shape[0]
    target_dwt_size = state[enc_key].shape[1]
    segment_size = _infer_segment_size_from_dwt_size(
        target_dwt_size,
        preferred=getattr(args, "mvpformer_segment_size", None),
    )
    if segment_size is None:
        raise ValueError(f"Unable to infer segment_size from checkpoint encoder dwt_size={target_dwt_size}")
    n_positions = state[pos_key].shape[0]
    n_channels = state[ch_key].shape[0]
    n_layer = len(
        {
            int(key.split(".")[2])
            for key in state
            if key.startswith("mvpformer.h.") and ".attn.q_attn.weight" in key
        }
    )
    n_inner = state[mlp_key].shape[0]
    embed_kv_dim = state[kv_key].shape[0] // 2

    head_candidates = (8, 12, 16, 24, 32)
    n_head = None
    n_head_kv = None
    for candidate in head_candidates:
        if final_dim % candidate != 0:
            continue
        head_dim = final_dim // candidate
        if embed_kv_dim % head_dim != 0:
            continue
        kv_heads = embed_kv_dim // head_dim
        if kv_heads > 0 and candidate % kv_heads == 0:
            n_head = candidate
            n_head_kv = kv_heads
            break
    if n_head is None or n_head_kv is None:
        raise ValueError(
            f"Unable to infer attention heads from checkpoint: final_dim={final_dim}, embed_kv_dim={embed_kv_dim}"
        )

    return {
        "final_dim": final_dim,
        "segment_size": segment_size,
        "n_positions": n_positions,
        "n_channels": n_channels,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "n_inner": n_inner,
    }


class MVPFormer_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.final_dim = 2048
        args.tune_a_part = True
        args.mvpformer_sfreq = 500
        args.mvpformer_segment_size = 5000
        args.mvpformer_max_segments = 110
        args.mvpformer_max_channels = 128
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


class MVPFormer(nn.Module):
    """MVPFormer backbone adapted to Brain4FMs DefaultDataset batches."""

    def __init__(self, args: Namespace):
        super().__init__()
        self.pretrained_path = ModelPathArgs.MVPFormer_path
        self.target_sfreq = getattr(args, "mvpformer_sfreq", 500)
        self.source_sfreq = getattr(args, "sfreq", self.target_sfreq)
        self.segment_size = getattr(args, "mvpformer_segment_size", 5000)
        self.max_segments = getattr(args, "mvpformer_max_segments", 110)
        self.max_channels = getattr(args, "mvpformer_max_channels", 128)
        checkpoint_state = {}
        checkpoint_path = Path(self.pretrained_path)
        if checkpoint_path.exists():
            checkpoint_state = _load_checkpoint_state(str(checkpoint_path))
        config = _infer_runtime_config(checkpoint_state, args)
        self.final_dim = config["final_dim"]
        self.segment_size = config["segment_size"]
        self.max_segments = config["n_positions"]
        self.max_channels = config["n_channels"]
        args.final_dim = self.final_dim
        args.mvpformer_segment_size = self.segment_size
        args.mvpformer_max_segments = self.max_segments
        args.mvpformer_max_channels = self.max_channels
        self.encoder = WaveEncoder(size_input=self.segment_size, size_output=self.final_dim)
        self.mvpformer = MVPFormerModel(
            MVPFormerConfig(
                n_positions=self.max_segments,
                n_channels=self.max_channels,
                n_embd=self.final_dim,
                n_layer=config["n_layer"],
                n_head=config["n_head"],
                n_head_kv=config["n_head_kv"],
                n_inner=config["n_inner"],
                global_att=True,
                activation_function="silu",
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
                layer_norm_epsilon=1e-5,
                initializer_range=0.02,
                scale_attn_weights=True,
                use_cache=False,
                scale_attn_by_inverse_layer_idx=False,
                reorder_and_upcast_attn=False,
            )
        )
        self.loaded_pretrained_keys = 0
        self.missing_pretrained_keys: list[str] = []
        self.unexpected_pretrained_keys: list[str] = []
        if getattr(args, "freeze", False):
            self._freeze_backbone()
        self._try_load_pretrained(checkpoint_state)

    def _freeze_backbone(self) -> None:
        for module in (self.encoder, self.mvpformer):
            for param in module.parameters():
                param.requires_grad = False

    def _try_load_pretrained(self, state: dict[str, torch.Tensor] | None = None) -> None:
        if state is None:
            checkpoint_path = Path(self.pretrained_path)
            if not checkpoint_path.exists():
                return
            state = _load_checkpoint_state(str(checkpoint_path))
        else:
            checkpoint_path = Path(self.pretrained_path)
        if not state:
            return
        current = self.state_dict()
        matched = {}
        for key, value in state.items():
            if key in current and current[key].shape == value.shape:
                matched[key] = value
        incompatible = self.load_state_dict(matched, strict=False)
        self.loaded_pretrained_keys = len(matched)
        self.missing_pretrained_keys = list(incompatible.missing_keys)
        self.unexpected_pretrained_keys = list(incompatible.unexpected_keys)
        if matched:
            print(
                "MVPFormer inferred config "
                f"(dim={self.final_dim}, segment={self.segment_size}, layers={len(self.mvpformer.h)}, heads={self.mvpformer.num_heads}, "
                f"kv_heads={self.mvpformer.num_kv_heads}, mlp={self.mvpformer.h[0].mlp.gate_proj.weight.shape[0]})"
            )
            print(
                f"MVPFormer loaded {len(matched)}/{len(current)} tensors from {checkpoint_path}"
            )

    def _prepare_segments(self, x: torch.Tensor, sfreq: int) -> torch.Tensor:
        x = x.float()
        bsz, ch_num, time_len = x.shape
        if sfreq != self.target_sfreq:
            target_len = max(1, round(time_len * self.target_sfreq / sfreq))
            x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)

        if x.shape[1] > self.max_channels:
            x = x[:, : self.max_channels]

        total_len = x.shape[-1]
        seg_n = max(1, (total_len + self.segment_size - 1) // self.segment_size)
        seg_n = min(seg_n, self.max_segments)
        clipped_len = seg_n * self.segment_size

        if total_len < clipped_len:
            x = F.pad(x, (0, clipped_len - total_len))
        else:
            x = x[..., :clipped_len]

        return x.reshape(bsz, x.shape[1], seg_n, self.segment_size).permute(0, 2, 1, 3).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_segments(x, sfreq=self.source_sfreq)
        bsz, seg_n, ch_num, _ = x.shape
        x_flat = x.reshape(bsz * seg_n * ch_num, -1)
        input_embeds = self.encoder(x_flat).reshape(bsz, seg_n, ch_num, self.final_dim)
        out = self.mvpformer(input_ids=None, inputs_embeds=input_embeds).last_hidden_state
        return out[:, -1].mean(dim=1)

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        emb = model(x)
        logit = clsf(emb)

        if args.run_mode == "test":
            return logit, y
        if args.run_mode in {"exp1", "exp3", "prototype"}:
            return emb, logit, y
        loss = loss_func(logit, y)
        return loss, logit, y
