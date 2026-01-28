"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

import torch
from torch import nn
import torch.nn.functional as F
import inspect

from model.NeuroLM.model.model_neural_transformer import NeuralTransformer
from model.NeuroLM.model.norm_ema_quantizer import NormEMAVectorQuantizer

from torch.autograd import Function
from transformers import GPT2LMHeadModel


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class VQ(nn.Module):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 n_embed=8192, 
                 embed_dim=128,
                 decay=0.99,
                 quantize_kmeans_init=True,
                 decoder_out_dim=200,
                 smooth_l1_loss = False,
                 **kwargs
                 ):
        super().__init__()
        print(kwargs)
        if decoder_config.in_chans != embed_dim:
            print(f"Rewrite the in_chans in decoder from {decoder_config.in_chans} to {embed_dim}")
            decoder_config.in_chans = embed_dim

        # encoder & decode params
        print('Final encoder config', encoder_config)
        self.encoder = NeuralTransformer(encoder_config)

        print('Final decoder config', decoder_config)
        self.decoder_freq = NeuralTransformer(decoder_config)
        self.decoder_raw = NeuralTransformer(decoder_config)
                
        self.quantize = NormEMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init, decay=decay,
        )

        self.decoder_out_dim = decoder_out_dim

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config.n_embd, encoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(encoder_config.n_embd, embed_dim) # for quantize
        )
        self.decode_task_layer_freq = nn.Sequential(
            nn.Linear(decoder_config.n_embd, decoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(decoder_config.n_embd, self.decoder_out_dim // 2),
        )
        self.decode_task_layer_raw = nn.Sequential(
            nn.Linear(decoder_config.n_embd, decoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(decoder_config.n_embd, self.decoder_out_dim),
        )

        self.kwargs = kwargs
        
        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer_freq.apply(self._init_weights)
        self.decode_task_layer_raw.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'quantize.embedding.weight', 'decoder.pos_embed', 'decoder.time_embed', 
    #             'encoder.pos_embed', 'encoder.time_embed'}

    @property
    def device(self):
        return self.decoder.cls_token.device
    
    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, input_chans=None, input_times=None, mask=None, **kwargs):
        quantize, embed_ind, loss, _ = self.encode(data, input_chans, input_times, mask)
        return embed_ind.view(data.size(0), data.size(1))

    def encode(self, x, input_chans=None, input_time=None, mask=None):
        batch_size, n, t = x.shape
        encoder_features = self.encoder(x, input_chans, input_time, mask, return_all_tokens=True)

        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))

        quantize, loss, embed_ind = self.quantize(to_quantizer_features)

        return quantize, embed_ind, loss, encoder_features
        
    def decode(self, quantize, input_chans=None, input_time=None, mask=None, **kwargs):
        # reshape tokens to feature maps for patch embed in decoder
        decoder_features_freq = self.decoder_freq(quantize, input_chans, input_time, mask, return_all_tokens=True)
        decoder_features_raw = self.decoder_raw(quantize, input_chans, input_time, mask, return_all_tokens=True)
        rec_freq = self.decode_task_layer_freq(decoder_features_freq)
        rec_raw = self.decode_task_layer_raw(decoder_features_raw)
        return rec_freq, rec_raw
    
    def get_codebook_indices(self, x, input_chans=None, input_time=None, input_mask=None, **kwargs):
        if input_mask is None:
            mask = None
        else:
            mask = input_mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        return self.get_tokens(x, input_chans, input_time, mask, **kwargs)
    
    def calculate_rec_loss(self, rec, target):
        rec_loss = self.loss_fn(rec, target)
        return rec_loss

    def forward(self, x, y_freq, y_raw, input_chans=None, input_time=None, input_mask=None, **kwargs):
        """
        x: shape [B, N, T]
        """
        mask = input_mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        quantize, embed_ind, emb_loss, encoder_features = self.encode(x, input_chans, input_time, mask)
        
        xrec_freq, xrec_raw = self.decode(quantize, input_chans, input_time, mask)

        loss_freq_mask = input_mask.unsqueeze(-1).repeat(1, 1, xrec_freq.size(-1))
        loss_raw_mask = input_mask.unsqueeze(-1).repeat(1, 1, xrec_raw.size(-1))
        rec_freq_loss = self.calculate_rec_loss(xrec_freq * loss_freq_mask, y_freq)
        rec_raw_loss = self.calculate_rec_loss(xrec_raw * loss_raw_mask, y_raw)
        loss = emb_loss + rec_freq_loss + rec_raw_loss

        log = {}
        split="train" if self.training else "val"
        log[f'{split}/quant_loss'] = emb_loss.detach().mean()
        log[f'{split}/rec_freq_loss'] = rec_freq_loss.detach().mean()
        log[f'{split}/rec_raw_loss'] = rec_raw_loss.detach().mean()
        log[f'{split}/total_loss'] = loss.detach().mean()

        return loss, encoder_features, log
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


class VQ_Align(nn.Module):
    def __init__(self, 
                 encoder_config,
                 decoder_config,
                 ):
        super(VQ_Align, self).__init__()
        self.VQ = VQ(encoder_config, decoder_config)
        self.domain_classifier = nn.Sequential(
                nn.Linear(decoder_config.n_embd, 256),
                nn.GELU(),
                nn.Linear(256, 2)
            )

        model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = model_hf.state_dict()
        self.wte = nn.Embedding(50257, 768, _freeze=True)
        self.wte.weight.data = sd_hf['transformer.wte.weight']
        
        self.domain_classifier.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, y_freq=None, y_raw=None, input_chans=None, input_time=None, input_mask=None, alpha=0):
        if y_freq is not None:
            loss, encoder_features, log = self.VQ(x, y_freq, y_raw, input_chans, input_time, input_mask)
            reverse_x = ReverseLayerF.apply(encoder_features, alpha)
            domain_out = self.domain_classifier(reverse_x)
            target = torch.full((domain_out.size(0), domain_out.size(1)), fill_value=-1, device=x.device)
            target[input_mask == True] = 0
            domain_loss = F.cross_entropy(domain_out.view(-1, domain_out.size(-1)), target.view(-1), ignore_index=-1)
            split="train" if self.training else "val"
            log[f'{split}/domain_loss'] = domain_loss.detach().item()
            return loss, domain_loss, log
        else:
            x = self.wte(x).detach()
            domain_out = self.domain_classifier(x)
            domain_loss = F.cross_entropy(domain_out.view(-1, domain_out.size(-1)), torch.ones((x.size(0) * x.size(1),), device=x.device).long(), ignore_index=-1)
            return domain_loss
        
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    