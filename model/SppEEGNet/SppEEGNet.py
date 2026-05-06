import torch
from torch import nn, optim
from argparse import Namespace
from model.SppEEGNet.downstream_models import Supervised_TUH
from model.model_config import ModelPathArgs



class SppEEGNet_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.final_dim = 100
        args.tune_a_part = True
        return args

    @staticmethod
    def clsf_loss_func(args, model):
        if args.weights is None:
            if args.n_class != 2:
                ce_weight = [1.0 for _ in range(args.n_class)]
            else:
                ce_weight = [0.3, 1.0]
        else:
            ce_weight = args.weights
        print(f'CrossEntropy loss weight = {ce_weight} = {ce_weight[1]/ce_weight[0]:.2f}')
        return nn.CrossEntropyLoss(torch.tensor(ce_weight, dtype=torch.float32, device=torch.device(args.gpu_id)))
        # return nn.CrossEntropyLoss()

    @staticmethod
    def optimizer(args, model, clsf):
        return torch.optim.Adam([
                {'params': list(model.model_clsf.parameters()), 'lr': args.model_lr},
                {'params': list(clsf.parameters()), 'lr': args.clsf_lr},],
            betas=(0.9, 0.99), eps=1e-8,)

    @staticmethod
    def scheduler(optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)
    
    
class SppEEGNet(nn.Module):
    def __init__(self, args: Namespace,):
        super(SppEEGNet, self).__init__()

        self.model_clsf = self.load_pretrained_weights(args)
        self.model_clsf = self.freeze_part(args, self.model_clsf)
        

    def forward(self, x):
        bsz, ch_num, seq_len, patch_len = x.shape

        logit = self.model_clsf(x)
        return logit

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        bsz, ch_num, N = x.shape
        x = x.reshape(bsz, 1, ch_num, -1)       # 1*C*N
        
        emb = model(x)
        logit = clsf(emb)

        if args.run_mode == 'test':
            return logit, y
        elif args.run_mode == 'prototype':
            return emb, logit, y
        else:
            loss = loss_func(logit, y)
            return loss, logit, y
            

    @staticmethod
    def load_pretrained_weights(args):
        pretrained_model_path = ModelPathArgs.SppEEGNet_path
        model = Supervised_TUH(in_features=1, 
                               encoder_h=32, num_classes=args.n_class,)
        pretrained_dict = torch.load(pretrained_model_path, map_location=f'cuda:{args.gpu_id}') 
        pretrained_param_names = list(pretrained_dict.keys())
        model_dict = model.state_dict()
        model_param_names = list(model_dict.keys())

        # :4 load the first conv layer  :8 load the first two conv layers :12 thrid :16 forth :20 fifth :24 sixth 
        for i, _ in enumerate(pretrained_param_names[:-2]):
            model_dict[model_param_names[i]] = pretrained_dict[pretrained_param_names[i]]

        model.load_state_dict(model_dict)
        return model.feature_extractor

    @staticmethod
    def freeze_part(args, model_clsf):
        if args.freeze:
            for param in model_clsf.parameters():
                param.requires_grad = False

        return model_clsf
