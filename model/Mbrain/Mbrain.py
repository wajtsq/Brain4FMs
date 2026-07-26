import os
import torch
from torch import nn, optim
from argparse import Namespace

from model.Mbrain.models.ssl_model import MBrain
from model.Mbrain.models.downstream_task_criterion import DownstreamCriterion, LinearClassifier4EEG
from model.model_config import ModelPathArgs
from model.ch_aggr_clsf import time_bandpower

class Mbrain_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.final_dim = 768
        args.hidden_dim = 256
        args.kernel_size = [4, 4, 4]
        args.stride_size = [2, 2, 1]
        args.padding_size = [0, 0, 0]
        args.graph_threshold = 0.5      # The threshold to sample edges in graph construct module
        args.mbrain_build_mean_matrix = False
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
        # return nn.CrossEntropyLoss()

    @staticmethod
    def optimizer(args, model, clsf):
        return torch.optim.Adam([{'params': model.encoder.parameters(), 'lr': 1e-3}, 
                                #  {'params':model.cls.parameters(), 'lr': 5e-4},
                                 {'params': model.att.parameters(), 'lr': 1e-6},
                                 {'params': clsf.parameters(), 'lr': args.clsf_lr}],
                                    betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=1e-6)

    @staticmethod
    def scheduler(optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)



class Mbrain(nn.Module):
    def __init__(self, args: Namespace,):
        super(Mbrain, self).__init__()
        self.hidden_dim = 256
        Mbrain_model = self.load_pretrained_weights(args)
        self.att = self.load_dm_pretrained_weights(args.hidden_dim * 3,)
        if args.freeze:
            Mbrain_model = self.freeze_part(Mbrain_model)
        self.encoder = Mbrain_model
        self.cls = LinearClassifier4EEG(
                    input_dim=args.hidden_dim * 3,
                    hidden_dim=[256, 128, args.n_class], weighted=False)
        
        
    def forward(self, x, y):
        batch_representation = []
        for batch_idx in range(x.size(0)):
            _, after_gAR, _ = self.encoder(x[batch_idx], train_stage=False)
            # after_gAR.size(): time_span * channel_num * seq_size * dim_ar

            r_max = torch.max(after_gAR[:, :, :, :self.hidden_dim], dim=2)[0]
            r_sum = torch.sum(after_gAR[:, :, :, :self.hidden_dim], dim=2)
            r_mean = torch.mean(after_gAR[:, :, :, :self.hidden_dim], dim=2)

            concat_representation = torch.cat((r_max, r_sum, r_mean), dim=-1)
            after_downAR = self.att(concat_representation)
            batch_representation.append(after_downAR)
            
        batch_representation = torch.stack(batch_representation, dim=0)
        all_losses, logit = self.cls(batch_representation, y, True)

        return batch_representation, all_losses, logit


    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        
        bsz, ch_num, N = x.shape
        if N % args.patch_len != 0:
            args.seq_len = int(N // args.patch_len)
            x = x[:, :, :args.seq_len*args.patch_len]
        x = x.reshape(bsz, -1, ch_num, args.patch_len)

        emb, all_losses, logit = model(x, y)
        # emb = emb[:, -1]
        # logit = clsf(emb)

        if args.run_mode == 'test':
            return logit, y
        elif args.run_mode == 'exp1' or args.run_mode == 'exp3' or args.run_mode == 'few-shot':
            if args.run_mode == 'exp3':
                inputs = x.reshape(bsz, ch_num, -1)
                y = time_bandpower(args, inputs, args.sfreq)
            emb = emb.mean(dim=1)
            emb = emb.mean(dim=1)
            return emb, logit, y
        else:
            loss = loss_func(logit, y)
            if all_losses.ndim > 0:
                all_losses = all_losses.mean()
            loss += all_losses
            return loss, logit, y 
        

    @staticmethod
    def load_pretrained_weights(args):
        pretrained_model_path = Mbrain._select_pretrained_path(args)
        Mbrain_model = MBrain(args=args,
                            hidden_dim=args.hidden_dim,
                            gcn_dim=[256],
                            n_predicts=8,        # Number of time steps in prediction task
                            graph_construct='sample_from_distribution',        # The method for graph construction, including ['sample_from_distribution', 'predefined_distance']
                            direction='single',         # The direction for prediction task, including ['single', 'bi', 'no']
                            replace_ratio=0.15,         # The ratio for replacing timestamps in replacement task.
                            ar_mode='LSTM',             # AR model: 'RNN', 'LSTM', 'GRU', 'TRANSFORMER'
        )
        checkpoint_path = Mbrain._resolve_checkpoint_path(pretrained_model_path)
        map_location = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(state_dict, dict):
            for key in ("BestModel", "Model", "model", "state_dict"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
        incompatible = Mbrain_model.load_state_dict(state_dict, strict=False)
        print(f'Mbrain loaded checkpoint: {checkpoint_path}')
        if incompatible.missing_keys:
            print(f'Mbrain missing keys: {len(incompatible.missing_keys)}')
        if incompatible.unexpected_keys:
            print(f'Mbrain unexpected keys: {len(incompatible.unexpected_keys)}')
        return Mbrain_model

    @staticmethod
    def _select_pretrained_path(args):
        seeg_path = getattr(ModelPathArgs, 'Mbrain_SEEG_path', None)
        if getattr(args, 'dataset', '') in {'HUP-SEEG', 'HUP-ECoG', 'SWEC', 'Cogitate', 'MAYO', 'FNUSA'} and seeg_path and os.path.exists(seeg_path):
            return seeg_path
        return ModelPathArgs.Mbrain_path

    @staticmethod
    def _resolve_checkpoint_path(pretrained_model_path):
        if os.path.isfile(pretrained_model_path):
            return pretrained_model_path
        final_epoch = None
        final_path = None
        for file in os.listdir(pretrained_model_path):
            if not file.endswith('.pt'):
                continue
            stem = file[:-3]
            try:
                epoch = int(stem.split('_')[-1])
            except ValueError:
                epoch = -1
            if final_epoch is None or epoch > final_epoch:
                final_epoch = epoch
                final_path = os.path.join(pretrained_model_path, file)
        if final_path is None:
            raise FileNotFoundError(f'No .pt checkpoint found in {pretrained_model_path}')
        return final_path


    @staticmethod
    def load_dm_pretrained_weights(input_dim):
        downstream_model = DownstreamCriterion(
            input_dim=input_dim,
            bi_direction=False,
        )
        return downstream_model
    

    @staticmethod
    def freeze_part(model_clsf):
        for param in model_clsf.parameters():
            param.requires_grad = False

        return model_clsf
