from torch import nn
import torch
import inspect
from argparse import Namespace
import tiktoken
import numpy as np

from model.NeuroLM.model.model import GPTConfig
from model.NeuroLM.model.model_neurolm import NeuroLMMain
from model.model_config import ModelPathArgs



class NeuroLM_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.tokenizer_path = ModelPathArgs.NeuroLM_token_path
        args.NeuroLM_path = ModelPathArgs.NeuroLM_path
        args.final_dim = args.block_size = 768
        args.gpt_path = ModelPathArgs.GPT2_folder_path
        args.gradient_accumulation_steps = 1
        # eeg_max_len, text_max_len = 276, 32
        # if args.seq_len*args.cnn_in_channels > eeg_max_len:
        #     eeg_max_len = args.seq_len*args.cnn_in_channels
        # args.cnn_in_channels = 1
        return args

    @staticmethod
    def clsf_loss_func(args, model):
        if args.weights is None:
            ce_weight = [1.0 for _ in range(args.n_class)]
        else:
            ce_weight = args.weights
        print(f'CrossEntropy loss weight = {ce_weight} = {ce_weight[1]/ce_weight[0]:.2f}')
        return nn.CrossEntropyLoss(torch.tensor(ce_weight, dtype=torch.float32, device=torch.device(args.gpu_id)))
        # return nn.CrossEntropyLoss()

    @staticmethod
    def optimizer(args, model, clsf):
        checkpoint = torch.load(args.NeuroLM_path, map_location=f'cuda:{args.gpu_id}', weights_only=False)
        param_dict = {pn: p for pn, p in model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available
        extra_args = dict(fused=True) if use_fused else dict()
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': 1e-1},
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': clsf.parameters(), 'lr': args.clsf_lr, 'weight_decay': 1e-6}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=5e-4, betas=(0.9, 0.95), **extra_args)

        state_dict = checkpoint['optimizer']
        optimizer_state_dict = optimizer.state_dict()
        for i, params in enumerate(optimizer.param_groups):
            if i < len(state_dict['param_groups']):
                optimizer_state_dict['param_groups'][i] = state_dict['param_groups'][i]
        optimizer_state_dict['state'] = state_dict['state']

        optimizer.load_state_dict(optimizer_state_dict)
        return optimizer

        
    @staticmethod
    def scheduler(optimizer):
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)


class NeuroLM(nn.Module):
    def __init__(self, args: Namespace):
        super(NeuroLM, self).__init__()
        checkpoint = torch.load(args.NeuroLM_path, map_location=f'cuda:{args.gpu_id}', weights_only=False)
        checkpoint_model_args = checkpoint['model_args']
        bias = False # do we use bias inside LayerNorm and Linear layers?
        model_args = dict(n_layer=checkpoint_model_args['n_layer'], n_head=checkpoint_model_args['n_head'], 
                          n_embd=checkpoint_model_args['n_embd'], block_size=args.block_size,
                    bias=bias, vocab_size=50257, dropout=checkpoint_model_args['dropout']) # start with model_args from command line

        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = NeuroLMMain(gptconf, init_from='gpt2',)     # tokenizer_ckpt_path=args.tokenizer_path 
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        
        # influence the config of optimizer
        # for name, param in model.named_parameters():
        #     if 'GPT2.transformer' in name:
        #         param.requires_grad = False
        
        self.model = model
        # enc = tiktoken.get_encoding("gpt2")
        # self.decode = lambda l: enc.decode(l)
        
        
    def forward(self, x_eeg=None, y_eeg=None, x_text=None, y_text=None, input_chans=None, 
                input_time=None, input_mask=None, eeg_mask=None, eeg_text_mask=None):
        loss, log, logits = self.model(x_eeg=x_eeg, y_eeg=y_eeg, x_text=x_text, 
                   y_text=y_text, input_chans=input_chans, input_time=input_time, 
                   input_mask=input_mask, eeg_mask=eeg_mask, eeg_text_mask=eeg_text_mask)
        
        # bsz, in_chan, final_dim = logits.shape
        # logits = logits.reshape(bsz, final_dim, in_chan)
        # logits = self.head(logits)
        # logits = logits.reshape(bsz, -1, final_dim)
        return loss, logits

    
    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):   
        device = f'cuda:{args.gpu_id}'     
        if len(data_packet) == 9:
            X_eeg, text, label, Y_text, input_chans, input_time, eeg_mask, gpt_mask, x = data_packet
            Y_text = Y_text.to(device)
        else:
            X_eeg, text, label, input_chans, input_time, eeg_mask, gpt_mask, x = data_packet
            
        X_eeg = X_eeg.float().to(device)
        X_text = text.to(device)
        label = label.to(device)
        input_chans = input_chans.to(device)
        input_time = input_time.to(device)
        gpt_mask = gpt_mask.to(device)
        if eeg_mask is not None:
            input_mask = eeg_mask.to(device)

        if args.is_parallel:
            Y_eeg = torch.full((X_eeg.size(0), X_eeg.size(1)), fill_value=-1-model.module.model.GPT2.config.vocab_size).to(device)
        else:
            Y_eeg = torch.full((X_eeg.size(0), X_eeg.size(1)), fill_value=-1-model.model.GPT2.config.vocab_size).to(device)

        if len(data_packet) == 9:
            loss, logits = model(x_eeg=X_eeg, y_eeg=Y_eeg, x_text=X_text, y_text=Y_text, input_chans=input_chans, 
                    input_time=input_time, input_mask=input_mask, eeg_mask=eeg_mask, eeg_text_mask=gpt_mask)
        else:    
            if args.is_parallel:
                text, logits = model.module.model.generate(
                    x_eeg=X_eeg, x_text=X_text, input_chans=input_chans, input_time=input_time, 
                    eeg_mask=eeg_mask, eeg_text_mask=gpt_mask, max_new_tokens=5, input_mask=input_mask
                )
            else:
                text, logits = model.model.generate(
                    x_eeg=X_eeg, x_text=X_text, input_chans=input_chans, input_time=input_time, 
                    eeg_mask=eeg_mask, eeg_text_mask=gpt_mask, max_new_tokens=5, input_mask=input_mask
                )
            # text = text[:, 1:] # remove [SEP] token

        emb = logits.mean(1)
        logit = clsf(emb)
        
        if args.run_mode != 'test':
            loss_total = loss_func(logit, label)
            if len(data_packet) == 9: 
                if args.is_parallel:
                    loss = sum(loss)
                loss_total += loss
            return loss_total, logit, label
        else:
            return logit, label
        