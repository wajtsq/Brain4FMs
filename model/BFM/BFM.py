import torch
from torch import nn, optim
from argparse import Namespace
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

from model.model_config import ModelPathArgs

class BFM_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.final_dim = 1024
        args.tokenizer_id = "google/t5-efficient-tiny"
        args.tokenizer_type = "seq2seq"
        args.vocab_size = 4096
        args.pad_token_id = 0
        args.eos_token_id = 1
        args.cnn_in_channels = 1
        
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


    @staticmethod
    def optimizer(args, model, clsf):
        return torch.optim.AdamW([
                {'params': list(model.parameters()), 'lr': args.model_lr},
                {'params': list(clsf.parameters()), 'lr': args.clsf_lr},
        ],
            betas=(0.9, 0.99), eps=1e-8,
        )
        
        
    @staticmethod
    def scheduler(optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)

    

class BFM(nn.Module):
    def __init__(self, args: Namespace):
        super(BFM, self).__init__()
        self.model = self.load_chronos_model(args)
        self.cls = nn.Linear(in_features=args.final_dim, out_features=args.n_class)
        

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels = labels,
            output_hidden_states=True,
            return_dict=True
        )
        loss = out.loss.item()
        logits = out.encoder_last_hidden_state      # [batch, seq_len, hidden_size]
        logits = logits.mean(dim=1)                 # [batch, hidden_size]
        return loss, logits
    
    
    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        y, input_ids, attention_mask, labels = data_packet
        device = f'cuda:{args.gpu_id}'
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        if args.is_parallel:
            loss, logits = model.module(input_ids, attention_mask, labels)
        else:
            loss, logits = model(input_ids, attention_mask, labels)
            
        # logits = logits.unsqueeze(1)
        logit = clsf(logits)
        
        if args.run_mode != 'test':
            loss += loss_func(logit, y)
            return loss, logit, y
        else:
            return logit, y
        
    @staticmethod
    def load_chronos_model(args: Namespace):
        AutoModelClass = (
            AutoModelForSeq2SeqLM if args.tokenizer_type == "seq2seq" else AutoModelForCausalLM
        )
        checkpoint_dir=ModelPathArgs.BFM_path
        model = AutoModelClass.from_pretrained(
            checkpoint_dir, cache_dir=checkpoint_dir, local_files_only=True, weights_only=False
        )
        model.resize_token_embeddings(args.vocab_size)
        model.config.pad_token_id = model.generation_config.pad_token_id = args.pad_token_id
        model.config.eos_token_id = model.generation_config.eos_token_id = args.eos_token_id

        if args.freeze:
            for name, param in model.named_parameters():
                # if 'final_layer_norm.' in name or '.lm_head' in name:
                #     continue
                param.requires_grad = False

        return model
    
    
    @staticmethod
    def codebook_weight(model):
        weight = model.model.shared.weight
        return weight