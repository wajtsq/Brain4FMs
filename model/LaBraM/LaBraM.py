from collections import OrderedDict
import librosa
import torch
from torch import nn, optim
from argparse import Namespace
from einops import rearrange
from timm.models import create_model        

from utils.data_info import data_info_dict
from model.pre_cnn import ConvNet
from model.LaBraM import utils
from model.model_config import ModelPathArgs


class LaBraM_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args._model = 'labram_base_patch200_200'
        args.nb_classes = 2
        args.drop = 0.0
        args.drop_path = 0.1
        args.attn_drop_rate = 0.0
        args.input_size = 200
        args.domain = 'freq'

        args.use_mean_pooling = True
        args.init_scale = 0.001
        args.rel_pos_bias = True
        args.abs_pos_emb = False
        args.layer_scale_init_value = 0.1
        args.qkv_bias = True

        args.finetune = ModelPathArgs.LaBraM_path

        args.model_key = 'model|module'
        args.model_filter_name = 'gzp'
        args.model_prefix = ''

        args.final_dim = 1024
        args.tune_a_part = True
     
        return args


    @staticmethod
    def clsf_loss_func(args, model):
        if args.weights is None:
            ce_weight = [0.4 for _ in range(args.n_class - 1)]
            ce_weight.append(1.0)
        else:
            ce_weight = args.weights        
        print(f'CrossEntropy loss weight = {ce_weight} = {ce_weight[1]/ce_weight[0]:.2f}')
        return nn.CrossEntropyLoss(torch.tensor(ce_weight, dtype=torch.float32, device=torch.device(args.gpu_id)))
        # return nn.CrossEntropyLoss()

    @staticmethod
    def optimizer(args, model, clsf):
        return optim.AdamW(
            [   
                {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.model_lr},
            ],
            betas=(0.9, 0.95), eps=1e-5,
        )

    @staticmethod
    def scheduler(optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)


class LaBraM(nn.Module):
    def __init__(self, args: Namespace,):
        super(LaBraM, self).__init__()
        self.model_clsf = self.load_models(args)
        if args.freeze:
            self.model_clsf = self.freeze_part(args, self.model_clsf)

    def forward(self, x):
        emb, logit = self.model_clsf(x)
        return emb, logit
    
    def vq_forward(self, x):
        b, n, a, t = x.shape
        encoder_features = self.model_clsf(x, input_chans=None, return_patch_tokens=True)[0]

        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))

        N = to_quantizer_features.shape[1]
        h, w = n, N // n

        to_quantizer_features = rearrange(to_quantizer_features, 'b (h w) c -> b c h w', h=h, w=w) # reshape for quantizer
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)    # codebook embs, emb ids
        q = quantize.mean(1)
        q = self.clsf(q)

        return q, embed_ind, loss
    
    def freq_forward(self, x):
        b, n, a, t = x.shape
        x_fft = torch.fft.fft(x, dim=-1)
        amplitude = torch.abs(x_fft)
        amplitude = self.std_norm(amplitude)
        amplitude = amplitude.reshape(b, -1, t)
        angle = torch.angle(x_fft)
        angle = self.std_norm(angle)     

        encoder_features = self.model_clsf(x, input_chans=None, return_patch_tokens=True)[0]
        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))

        N = to_quantizer_features.shape[1]
        h, w = n, N // n

        to_quantizer_features = rearrange(to_quantizer_features, 'b (h w) c -> b c h w', h=h, w=w) # reshape for quantizer
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)    # codebook embs, emb ids
        quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=h, w=w)
        decoder_features, decoder_features_head = self.decoder(quantize, input_chans=None, return_patch_tokens=True)
        rec = self.decode_task_layer(decoder_features)
        rec_angle = self.decode_task_layer_angle(decoder_features)
        Fpos = t // 2 + 1
        return amplitude[..., :Fpos], rec[..., :Fpos]

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        bsz, ch_num, a = x.shape
        if a % 200 != 0:
            a = (int)(a / 200) * 200
            x = x.narrow(2, 0, a)
        x = rearrange(x, 'B N (A T) -> B N A T', T=200)  
        b, n, a, t = x.shape
        if a > 16:
            x = x.narrow(2, 0, 16)
            a = 16
        b, n, a, t = x.shape

        emb, logit = model(x)

        if args.run_mode == 'test':
            return logit, y
        else:
            loss = loss_func(logit, y)
            return loss, logit, y
        
    
    @staticmethod
    def load_models(args):
        import model.LaBraM.modeling_finetune
        seq_len = max(args.seq_len, 16)
        model_ = create_model(
            'labram_base_patch200_200',
            pretrained=False,
            num_classes=args.n_class,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            qkv_bias=args.qkv_bias,
            seq_len=seq_len,
        )
        patch_size = model_.patch_size
        print("Patch size = %s" % str(patch_size))
        args.window_size = (1, args.input_size // patch_size)
        args.patch_size = patch_size
        
        checkpoint = torch.load(args.finetune, map_location='cpu')
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = model_.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.load_state_dict(model_, checkpoint_model, prefix='')
        
        return model_
    
    
    @staticmethod
    def codebook_weight(model):
        weight = model.quantize.embedding.weight.data
        return weight


    @staticmethod
    def freeze_part(args, model_clsf):
        if args.tune_a_part:
            for name, param in model_clsf.named_parameters():
                if 'head.' in name :
                    # or 'fc_norm.' in name or  \
                    # '.attn.v_bias' in name or '.attn.q_bias' in name:
                    continue
                param.requires_grad = False

        return model_clsf