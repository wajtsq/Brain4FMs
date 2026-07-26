from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import GPT2Config, GPT2Model, GPT2PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import (
    BaseModelOutputWithPastAndCrossAttentions,
    GPT2Block,
)
from transformers.models.llama.modeling_llama import LlamaMLP

RMSNorm = nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm

DB4_DEC_LO = [
    -0.010597401785069032,
    0.0328830116668852,
    0.030841381835560764,
    -0.18703481171888114,
    -0.027983769416859854,
    0.6308807679298587,
    0.7148465705529157,
    0.2303778133088964,
]
DB4_DEC_HI = [
    -0.2303778133088964,
    0.7148465705529157,
    -0.6308807679298587,
    -0.027983769416859854,
    0.18703481171888114,
    0.030841381835560764,
    -0.0328830116668852,
    -0.010597401785069032,
]


class WaveEncoder(nn.Module):
    def __init__(
        self,
        size_input: int,
        size_output: int,
        wavelet: str = "db4",
    ) -> None:
        super().__init__()
        if wavelet != "db4":
            raise ValueError(f"Only db4 is supported in the self-contained runtime, got {wavelet!r}")
        self.size_input = size_input
        self.size_output = size_output
        self.wavelet = wavelet
        self.filter_len = len(DB4_DEC_LO)
        self.register_buffer("dec_lo", torch.tensor(DB4_DEC_LO, dtype=torch.float32))
        self.register_buffer("dec_hi", torch.tensor(DB4_DEC_HI, dtype=torch.float32))
        self.sizes: list[int] = []
        self._compute_dwt_size()
        self.ln = RMSNorm(self.dwt_size, eps=1e-5)
        self.proj = nn.Linear(self.dwt_size, size_output, bias=False)

    def _compute_dwt_size(self) -> None:
        max_levels = max(1, int(math.floor(math.log2(self.size_input / (self.filter_len - 1)))))
        input_size = self.size_input
        for _ in range(max_levels):
            input_size = (input_size + 1) // 2
            self.sizes.append(input_size)
        self.sizes.append(input_size)
        self.dwt_size = sum(self.sizes)

    def _dwt_periodic(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] % 2 == 1:
            x = torch.cat([x, x[:, :1]], dim=-1)

        out_len = (x.shape[-1] + 1) // 2
        idx = (
            2 * torch.arange(out_len, device=x.device).unsqueeze(1)
            + torch.arange(self.filter_len, device=x.device).unsqueeze(0)
        ) % x.shape[-1]
        windows = x[:, idx]
        approx = (windows * self.dec_lo.to(device=x.device, dtype=x.dtype)).sum(dim=-1)
        detail = (windows * self.dec_hi.to(device=x.device, dtype=x.dtype)).sum(dim=-1)
        return approx, detail

    @torch.no_grad()
    def block(self, x: torch.Tensor) -> torch.Tensor:
        current = x.float()
        details = []
        for _ in self.sizes[:-1]:
            current, detail = self._dwt_periodic(current)
            details.append(detail)
        return torch.cat([current] + details[::-1], dim=-1).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.block(x)
        outputs = self.ln(outputs)
        return self.proj(outputs.type(x.dtype))


class MVPFormerConfig(GPT2Config):
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "max_channel_embeddings": "n_channels",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
        "intermediate_size": "n_inner",
        "hidden_act": "activation_function",
    }

    def __init__(
        self,
        n_positions: int = 110,
        n_channels: int = 128,
        n_embd: int = 2048,
        n_layer: int = 24,
        n_head: int = 16,
        n_inner: int = 5632,
        n_head_kv: int = 8,
        global_att: bool = True,
        activation_function: str = "silu",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        scale_attn_weights: bool = True,
        use_cache: bool = False,
        scale_attn_by_inverse_layer_idx: bool = False,
        reorder_and_upcast_attn: bool = False,
    ) -> None:
        self.n_channels = n_channels
        self.global_att = global_att
        self.n_head_kv = n_head_kv
        self.pretraining_tp = 1
        self.mlp_bias = False
        self.lora = False
        super().__init__(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            scale_attn_weights=scale_attn_weights,
            use_cache=use_cache,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
            add_cross_attention=False,
        )


class MVPFormerGQAAttention(nn.Module):
    def __init__(self, config: MVPFormerConfig, layer_idx: int | None = None) -> None:
        super().__init__()
        max_timesteps = config.max_position_embeddings
        self.register_buffer(
            "time_bias",
            torch.tril(torch.ones((max_timesteps, max_timesteps), dtype=torch.bool)).view(
                1, 1, max_timesteps, max_timesteps
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.n_head_kv
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.embed_kv_dim = self.head_dim * self.num_kv_heads
        self.global_att = config.global_att
        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx

        self.q_attn = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.c_attn = nn.Linear(self.embed_dim, 2 * self.embed_kv_dim, bias=False)
        self.position_net = nn.Linear(self.embed_dim, self.embed_kv_dim, bias=False)
        self.channel_net = nn.Linear(self.embed_dim, self.embed_kv_dim, bias=False)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.attn_bias = nn.Parameter(torch.empty(3 * self.embed_kv_dim))
        nn.init.normal_(self.attn_bias, std=0.02)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, clen, tlen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :, :].expand(
            batch, num_key_value_heads, n_rep, clen, tlen, head_dim
        )
        return hidden_states.reshape(
            batch, num_key_value_heads * n_rep, clen, tlen, head_dim
        )

    @staticmethod
    def repeat_channel(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, head_dim, clen = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, :, None].expand(
            batch, num_key_value_heads, head_dim, clen, n_rep
        )
        return hidden_states.reshape(batch, num_key_value_heads, head_dim, clen * n_rep)

    @staticmethod
    def repeat_time(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, head_dim, tlen = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(
            batch, num_key_value_heads, head_dim, n_rep, tlen
        )
        return hidden_states.reshape(batch, num_key_value_heads, head_dim, n_rep * tlen)

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        zero_pad_shape = x.size()[:2] + (x.size(3), 1)
        x_review_shape = x.size()[:2] + (x.size(3), x.size(2))
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x.view(x_review_shape)], dim=-1)
        x_padded_shape = x.size()[:2] + (x.size(2) + 1, x.size(3))
        x_padded = x_padded.view(*x_padded_shape)
        return x_padded[..., 1:, :].view_as(x)

    @staticmethod
    def _rel_shift_chan(x: torch.Tensor) -> torch.Tensor:
        chan_size = x.shape[-1]
        if chan_size > 1:
            upper_val = torch.cat(
                [
                    torch.arange(1, chan_size - i, dtype=torch.int32)
                    for i in range(chan_size - 1)
                ]
            )
        else:
            upper_val = torch.tensor([], dtype=torch.int32)
        idxes = torch.triu_indices(chan_size, chan_size, offset=1)
        shifting_idxes = torch.zeros(chan_size, chan_size, dtype=torch.int32)
        shifting_idxes[..., idxes[0], idxes[1]] = upper_val
        shifting_idxes.transpose(-2, -1)[..., idxes[0], idxes[1]] = upper_val
        shifting_idxes = (chan_size - 1 - shifting_idxes).repeat(
            x.shape[-2] // chan_size, 1
        )
        return x[..., torch.arange(x.size(-2)).unsqueeze(1), shifting_idxes]

    def _split_heads(self, tensor: torch.Tensor, num_heads: int, attn_head_size: int) -> torch.Tensor:
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 3, 1, 2, 4)

    def _merge_heads(self, tensor: torch.Tensor, num_heads: int, attn_head_size: int) -> torch.Tensor:
        tensor = tensor.permute(0, 2, 3, 1, 4).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _rel_attn(
        self,
        query: torch.Tensor,
        global_key: torch.Tensor,
        time_key: torch.Tensor,
        channel_key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, pos_len, chan_len, _ = query.size()
        tot_len = pos_len * chan_len
        global_key_bias, time_key_bias, channel_key_bias = self.attn_bias.split(
            self.embed_kv_dim, dim=0
        )
        if self.global_att:
            global_key_bias = self._split_heads(
                global_key_bias[None, None, None, :], self.num_kv_heads, self.head_dim
            )
        time_key_bias = self._split_heads(
            time_key_bias[None, None, None, :], self.num_kv_heads, self.head_dim
        )
        channel_key_bias = self._split_heads(
            channel_key_bias[None, None, None, :], self.num_kv_heads, self.head_dim
        )
        if self.global_att:
            global_key_bias = self.repeat_kv(global_key_bias, self.num_kv_groups)
            global_key = self.repeat_kv(global_key, self.num_kv_groups)
            global_key = global_key.reshape(bsz, self.num_heads, tot_len, self.head_dim).transpose(-1, -2)
        channel_key_bias = self.repeat_kv(channel_key_bias, self.num_kv_groups)
        time_key_bias = self.repeat_kv(time_key_bias, self.num_kv_groups)
        time_key = self.repeat_kv(time_key, self.num_kv_groups).squeeze(-2).transpose(-1, -2)
        channel_key = self.repeat_kv(channel_key, self.num_kv_groups).squeeze(-3).transpose(-1, -2)

        if self.global_att:
            global_query_head = (query + global_key_bias).reshape(bsz, self.num_heads, -1, self.head_dim)
            global_att = torch.matmul(global_query_head, global_key)
        time_query_head = (query + time_key_bias).reshape(bsz, self.num_heads, -1, self.head_dim)
        channel_query_head = (query + channel_key_bias).reshape(bsz, self.num_heads, -1, self.head_dim)
        time_att = torch.matmul(time_query_head, time_key)
        channel_att = torch.matmul(channel_query_head, channel_key)
        time_att = self._rel_shift(time_att)
        channel_att = self._rel_shift_chan(channel_att)
        attn_weights = self.repeat_channel(time_att, chan_len) + self.repeat_time(channel_att, pos_len)

        if self.global_att:
            window_mask = torch.logical_and(
                torch.tril(torch.ones((pos_len, pos_len), device=query.device, dtype=bool), diagonal=10),
                torch.triu(torch.ones((pos_len, pos_len), device=query.device, dtype=bool), diagonal=-10),
            )
            window_mask = window_mask.repeat_interleave(chan_len, 0).repeat_interleave(chan_len, 1)
            window_mask[-chan_len:] = 1
            attn_weights += global_att.masked_fill(~window_mask, 0.0)

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        causal_mask = (
            torch.tril(torch.ones((pos_len, pos_len), device=query.device, dtype=bool))
            .repeat_interleave(chan_len, 0)
            .repeat_interleave(chan_len, 1)
        )
        attn_weights = attn_weights.masked_fill(~causal_mask, torch.finfo(attn_weights.dtype).min)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        if value.dtype == torch.float16:
            attn_weights = torch.nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.bfloat16
            ).to(value.dtype)
        else:
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        value = self.repeat_kv(value, self.num_kv_groups)
        attn_output = torch.matmul(attn_weights, value.flatten(2, 3))
        return attn_output.view_as(value), attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        positional_embedding: torch.Tensor,
        channel_embedding: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        del encoder_hidden_states, encoder_attention_mask
        query = self.q_attn(hidden_states)
        key, value = self.c_attn(hidden_states).split(self.embed_kv_dim, dim=-1)
        time_key = self.position_net(positional_embedding).unsqueeze(2)
        channel_key = self.channel_net(channel_embedding).unsqueeze(1)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_kv_heads, self.head_dim)
        value = self._split_heads(value, self.num_kv_heads, self.head_dim)
        time_key = self._split_heads(time_key, self.num_kv_heads, self.head_dim)
        channel_key = self._split_heads(channel_key, self.num_kv_heads, self.head_dim)
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None
        attn_output, attn_weights = self._rel_attn(
            query, key, time_key, channel_key, value, attention_mask, head_mask
        )
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class MVPFormerBlock(GPT2Block):
    def __init__(self, config: MVPFormerConfig, layer_idx: int | None = None) -> None:
        torch.nn.Module.__init__(self)
        hidden_size = config.hidden_size
        self.ln_1 = RMSNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = MVPFormerGQAAttention(config, layer_idx=layer_idx)
        self.ln_2 = RMSNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positional_embedding: torch.Tensor,
        channel_embedding: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            positional_embedding,
            channel_embedding,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output + attn_output
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs


class MVPFormerModel(GPT2Model):
    def __init__(self, config: MVPFormerConfig) -> None:
        GPT2PreTrainedModel.__init__(self, config)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.n_head_kv
        self.head_dim = self.embed_dim // self.num_heads
        self.embed_kv_dim = self.head_dim * self.num_kv_heads
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [MVPFormerBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.ln_f = RMSNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.positional_embedding = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.channel_embedding = nn.Embedding(config.max_channel_embeddings, self.embed_dim)
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        channel_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        del token_type_ids, encoder_hidden_states, encoder_attention_mask
        if inputs_embeds is None:
            raise NotImplementedError("MVPFormerModel requires inputs_embeds")
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_shape[-2] + past_length, dtype=torch.long, device=device
            ).unsqueeze(0).view(-1, input_shape[-2])
        if channel_ids is None:
            channel_ids = torch.arange(
                past_length, input_shape[-1] + past_length, dtype=torch.long, device=device
            ).unsqueeze(0).view(-1, input_shape[-1])

        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        position_embeds = self.positional_embedding(position_ids)
        channel_embeds = self.channel_embedding(channel_ids)
        hidden_states = self.drop(inputs_embeds)
        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(
                hidden_states,
                position_embeds,
                channel_embeds,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states).view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(
                v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
        )
