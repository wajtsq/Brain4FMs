import random
import shutil

import torch
import numpy as np
import time
import os
import sys
sys.path.append('/bench-mark')

num_threads = '32'
torch.set_num_threads(int(num_threads))
os.environ['OMP_NUM_THREADS'] = num_threads
os.environ['OPENBLAS_NUM_THREADS'] = num_threads
os.environ['MKL_NUM_THREADS'] = num_threads
os.environ['VECLIB_MAXIMUM_THREADS'] = num_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_threads

import sys
import argparse
import json
from copy import deepcopy
from torch import nn
from sklearn.utils.class_weight import compute_class_weight
from utils.misc import set_seed, load_checkpoint, save_logs, save_checkpoint, update_main_logs, show_logs, process_init
process_init()

from model.ch_aggr_clsf import ChannelAggrClsf, LinearClsf
from pipeline.eval_epoch import evaluate_epoch
from pipeline.train_epoch import train_epoch
from pipeline.prototype_epoch import get_prototype, contrast
from utils.meta_info import get_data_dict, trainer_dict, model_dict
from utils.data_info import data_info_dict
from utils.get_data import sample_training_data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', 'yes', '1'):
        return True
    elif v.lower() in ('false', 'f', 'no', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BrainBenchmark')

    group_train = parser.add_argument_group('Train')
    group_train.add_argument('--exp_id', type=str, default='-1',
                             help='The experimental id.')
    group_train.add_argument('--gpu_id', type=int, default=0,
                             help='The gpu id.')
    group_train.add_argument('--gpu_ids', type=eval, default=[0,1],
                             help='The gpu ids.')
    group_train.add_argument('--is_parallel', type=str2bool, default=True,
                             help='Use more than one gpu')
    group_train.add_argument('--cv_id', type=int, default=4, help='The cross validation id.')
    group_train.add_argument('--run_mode', type=str, default='finetune',   # finetune, test, few-shot, prototype
                             help='To perform finetuning, or testing.')
    group_train.add_argument('--shot', type=int, default=8,
                             help='The sample number of prototype few shot.')
    group_train.add_argument('--batch_size', type=int, default=2,
                             help='Number of batches.')
    group_train.add_argument('--accumulation_steps', type=int, default=1,
                             help='Gradient accumulation parameter.')
    group_train.add_argument('--save_epochs', type=int, default=1,
                             help='The epoch number to save checkpoint.')
    group_train.add_argument('--warmup_epoch', type=int, default=3,
                             help='The epoch number to save checkpoint.')
    group_train.add_argument('--epoch_num', type=int, default=50,
                             help='Epoch number for total iterations.')
    group_train.add_argument('--early_stop', action='store_false',
                             help='Whether to use early stopping technique during training.')
    group_train.add_argument('--patience', type=int, default=5,
                             help='The waiting epoch number for early stopping.')
    group_train.add_argument('--from_pretrained', action='store_false', # when not --from_pretrained, must set the load_ckpt_path
                             help='Whether to finetune from pretrained weights')
    group_train.add_argument('--load_ckpt_path', type=str, default=f'/ckpt/',
                             help='The path to load checkpoint (.pt file or the upper path).')
    group_train.add_argument('--load_best', action='store_false',
                             help='Whether to load the best state in the checkpoints (to continue unsupervised training or begin finetuning).')
    group_train.add_argument('--save_ckpt_path', type=str, default=f'/ckpt/',
                             help='The path to save checkpoint')
    group_train.add_argument('--model_lr', type=float, default=1e-5,
                             help='The learning rate of the pretrained model.')
    group_train.add_argument('--clsf_lr', type=float, default=1e-4,
                             help='The learning rate of the classifier.')
    group_train.add_argument('--tqdm_dis', action='store_true', # add --tqdm_dis to disable
                             help='Whether to use tqdm_dis')

    group_data = parser.add_argument_group('Data')
    # ISRUC FNUSA CHBMIT UCSD_ON UCSD_OFF ADHD_Adult ADHD_Child Schizophrenia_28 MPHCE_mdd(MDD-64) MPHCE_state SEED_IV
    # SD_71 EEGMat DEAP EEGMMIDB_R EEGMMIDB_I Depression_122_STAI Depression_122_BDI MAYO SleepEDFx ADFD(AD-65) BCI-2a 
    # Chisco_read Chisco_imagine
    group_data.add_argument('--dataset', type=str, default='MAYO',
                            help='The dataset to perform training.')
    group_data.add_argument('--downstream', type=str, default='disorder',  # disorder, emotion, Motor imagine
                            help='The dataset to perform training.')
    group_data.add_argument('--sample_seq_num', type=int, default=None,
                            help='The number of sequence sampled from each dataset.')
    group_data.add_argument('--seq_len', type=int, default=10,
                            help='The number of patches in a sequence.')
    group_data.add_argument('--patch_len', type=int, default=256,
                            help='The number of points in a patch.')
    group_data.add_argument('--n_process_loader', type=int, default=5,
                            help='Number of processes to call to load the dataset.')

    group_arch = parser.add_argument_group('Architecture')
    group_arch.add_argument('--random_seed', type=int, default=1,
                            help="Set a specific random seed.")
    # BrainBERT Brant1 BIOT LaBraM Bendr SppEEGNet Mbrain BrainWave EEGPT CBraMod
    # NeuroLM NeuroGPT BFM BrainOmni REVE
    # EEGNet SPaRCNet DeprNet
    group_arch.add_argument('--model', type=str, default='LaBraM', 
                            help='The model to run.')
    group_arch.add_argument('--head', type=str, default='linear',  
                            help='The head of model.')
    group_arch.add_argument('--cnn_in_channels', type=int, default=64,
                            help="The number of input channels of the dataset.")
    group_arch.add_argument('--cnn_kernel_size', type=int, default=8,
                            help="The kernel size of the CNN to aggregate the channels.")
    group_arch.add_argument('--freeze', type=str2bool, default=False,
                            help='Whether to freeze the model.')

    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    
    args.n_class = data_info_dict[args.dataset]['n_class']
    args.sfreq = data_info_dict[args.dataset]['sfreq']
    args.full_data_path = data_info_dict[args.dataset]['data_path']
    args.patch_len = args.sfreq*1
    args.cnn_in_channels = data_info_dict[args.dataset]['channel']
    args.seq_len = data_info_dict[args.dataset]['seq_len']
    args.downstream = data_info_dict[args.dataset]['downstream']
    args.accumulation_steps = max(1, 128 // args.batch_size)
    if 'Chisco' in args.dataset:
        args.label_path = data_info_dict[args.dataset]['label_path']

    trainer = trainer_dict[args.model](args)
    args = trainer.set_config(args)

    if args.random_seed is None:
        args.random_seed = random.randint(0, 2 ** 31)
    set_seed(args.random_seed)

    print(f'CONFIG:\n{json.dumps(vars(args), indent=4, sort_keys=False)}')
    print('-' * 50)

    if args.is_parallel:
        args.gpu_id = args.gpu_ids[0]
    args.device = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() else torch.device('cpu')

    main_logs = {"epoch": []}
    args.weights = [1.0 for i in range(args.n_class)]

    if args.run_mode != 'test':
        tr_x_list, tr_y_list = get_data_dict[args.dataset](args, step='train')
        vl_x_list, vl_y_list = get_data_dict[args.dataset](args, step='valid')
        # args.weights = [0.1, 1]
        args.weights = compute_class_weight('balanced', classes=np.unique(tr_y_list[0]), y=tr_y_list[0])

    args.data_id = '{}_ssn{}_sl{}_pl{}'.format(
        args.dataset,
        args.sample_seq_num,
        args.seq_len,
        args.patch_len,
    )
    
    model = model_dict[args.model](args).to(args.device)
    if args.head == 'linear':
        clsf = LinearClsf(args).to(args.device)
    elif args.head == 'cnn':
        clsf = ChannelAggrClsf(args).to(args.device)

    loss_func = trainer.clsf_loss_func(args, model)
    optimizer = trainer.optimizer(args, model, clsf)
    # scheduler = trainer.scheduler(optimizer)
    scheduler = None

    if args.is_parallel:
        from torch.nn import DataParallel
        gpu_id = args.gpu_id
        model = DataParallel(model, device_ids=args.gpu_ids)
        clsf = DataParallel(clsf, device_ids=args.gpu_ids)

    best_vl_loss = np.inf
    state_dict = None

    # if run_mode == finetune and from_pretrained == True, will load the pretrained weights
    # if run_mode == finetune and from_pretrained == False, will load the ckpt to continue finetuning
    # if run_mode == test, will load the finetuned ckpt to infer

    assert (not args.from_pretrained and args.load_ckpt_path is not None) or \
           args.from_pretrained or \
           args.run_mode == 'test'      # must set the load_ckpt_path if continue finetuning

    # host_name = os.getcwd().split('/')[2]
    # args.path_checkpoint = utils.paths[host_name]  # 根据用户设置path_checkpoint

    # Load ckpt to continue finetuning
    if (not args.from_pretrained and args.load_ckpt_path is not None) or \
       args.run_mode == 'test':
        args.load_ckpt_path = f'{args.load_ckpt_path}/{args.model}_exp{args.exp_id}_cv{args.cv_id}_{args.data_id}/'

        load_path, main_logs = load_checkpoint(args, ckpt_type='finetune')
        print('Load checkpoint:', load_path)

        state_dict = torch.load(load_path, 'cpu', weights_only=False)
        if args.load_best or args.run_mode == 'test':
            model.load_state_dict(state_dict["BestModel"], strict=False)
            clsf .load_state_dict(state_dict["BestClsf"], strict=False)
        else:
            model.load_state_dict(state_dict["Model"], strict=False)
            clsf .load_state_dict(state_dict["Clsf"], strict=False)

        best_vl_loss = state_dict["BestValLoss"]
        best_model_state = state_dict["BestModel"]
        best_clsf_state  = state_dict["BestClsf"]
        optimizer.load_state_dict(state_dict["Optimizer"])
        print(f'Checkpoint loaded: best_vl_loss = {best_vl_loss:.4f}')
        if args.run_mode == 'unsupervised':
            print('Now continue the unsupervised training from the ckpt.')
        elif args.run_mode == 'finetune':
            print('Now continue the finetuning from the ckpt.')
        elif args.run_mode == 'test':
            print('Now do inferring using the ckpt.')
        print('-'*50)
    else:
        # args.load_ckpt_path = f'{args.load_ckpt_path}/{args.model}_exp{args.exp_id}_cv{args.cv_id}_{args.data_id}/'
        # if os.path.exists(os.path.join(args.load_ckpt_path, 'finetune_ckpt', '0.pt')):
        #     load_path, main_logs = load_checkpoint(args, ckpt_type='finetune')
        #     args.from_pretrained = False
        #     print('Load checkpoint:', load_path)
        # else:
        #     print('Not load checkpoints, begin finetuning from pretrained weights.')
        best_model_state = deepcopy(model.state_dict())
        best_clsf_state  = deepcopy(clsf .state_dict())

    # Settings to save ckpt
    if args.save_ckpt_path is not None:
        args.save_ckpt_path = f'{args.save_ckpt_path}/{args.model}_exp{args.exp_id}_cv{args.cv_id}_{args.data_id}/{args.run_mode}_ckpt/'
        # create if not exist
        if not os.path.exists(args.save_ckpt_path):
            os.makedirs(args.save_ckpt_path)
        elif args.run_mode == 'finetune' and args.from_pretrained:  # not to continue finetuning, but begin fintuning
            shutil.rmtree(args.save_ckpt_path)
            os.mkdir(args.save_ckpt_path)
            print(f'To begin the finetuning, have deleted the existing save_ckpt_path in {args.save_ckpt_path}')

    if args.run_mode == 'test':
        ts_x_list, ts_y_list = get_data_dict[args.dataset](args, step='test')

        ts_logs, _ = evaluate_epoch(args, ts_x_list, ts_y_list, model, clsf, loss_func, step='test')
        save_logs(ts_logs, f"{args.save_ckpt_path}/test_logs.json")
        exit(0)
    elif args.run_mode == 'prototype':
        ts_x_list, ts_y_list = get_data_dict[args.dataset](args, step='test')
        tr_x_list, tr_y_list = sample_training_data(tr_x_list, tr_y_list, vl_x_list, vl_y_list, shot=args.shot)
        prototype = get_prototype(args, model, tr_x_list, tr_y_list, clsf, loss_func)
        ts_logs = contrast(args, model, ts_x_list[0], ts_y_list[0], prototype, clsf, loss_func)
        save_logs(ts_logs, f"{args.save_ckpt_path}/{args.shot}-shot_logs.json")
        exit(0)
    elif args.run_mode == 'few-shot':
        ts_x_list, ts_y_list = get_data_dict[args.dataset](args, step='test')
        tr_x_list, tr_y_list = sample_training_data(tr_x_list, tr_y_list, vl_x_list, vl_y_list, shot=args.shot)
        tr_x_list, tr_y_list = [tr_x_list], [tr_y_list]
    print('-' * 50)

    print(f"Running at most {args.epoch_num} epochs")
    start_epoch = len(main_logs["epoch"]) if not args.from_pretrained else 0   # continue when restart
    last_tr_loss = 0
    wait_epoch = 0

    start_time = time.time()
    for epoch in range(start_epoch, args.epoch_num):
        print('-' * 50)
        print(f"Epoch {epoch}")
        # cpu_stats()

        tr_logs, tr_loss = train_epoch(args, tr_x_list, tr_y_list, model, clsf, loss_func, optimizer, scheduler,)
        if epoch >= args.warmup_epoch and args.run_mode != 'few-shot':
            vl_logs, vl_loss = evaluate_epoch(args, vl_x_list, vl_y_list, model, clsf, loss_func, step='valid')

            # print(f'Ran {epoch - start_epoch + 1} epochs in {time.time() - start_time:.2f} seconds')

            # process validation loss
            if vl_loss < best_vl_loss:
                print(f'Best model state updated. Best valid loss {best_vl_loss:.4f} => {vl_loss:.4f}')
                best_vl_loss = deepcopy(vl_loss)
                best_model_state = deepcopy(model.state_dict())
                best_clsf_state  = deepcopy(clsf .state_dict())
                wait_epoch = 0
            elif args.early_stop:
                wait_epoch += 1
                print(f'Early stop: wait epoch {wait_epoch}/{args.patience}')

            # update main logs
            main_logs = update_main_logs(main_logs, tr_logs, vl_logs, epoch)

            # save checkpoint at every epoch
            if args.save_ckpt_path is not None and (
                    epoch % args.save_epochs == 0 or epoch == args.epoch_num - 1 or
                    wait_epoch >= args.patience):
                model_state = model.state_dict()
                clsf_state  = clsf .state_dict()
                optimizer_state = optimizer.state_dict()

                save_checkpoint(
                    model_state,
                    clsf_state,
                    optimizer_state,
                    best_model_state,
                    best_clsf_state,
                    best_vl_loss,
                    f"{args.save_ckpt_path}/0.pt",
                    # f"{args.save_ckpt_path}/{epoch}.pt",
                )
                save_logs(main_logs, f"{args.save_ckpt_path}/logs.json")
                print('Checkpoint and main logs saved.')

            if wait_epoch >= args.patience:
                break

    if args.run_mode == 'few-shot':
        ts_logs, ts_loss = evaluate_epoch(args, ts_x_list, ts_y_list, model, clsf, loss_func, step='test')
        main_logs = update_main_logs(main_logs, tr_logs, ts_logs, epoch)
        save_logs(main_logs, f"{args.save_ckpt_path}/{args.shot}-shot_logs.json")
        print('Main logs saved.')
    print('-' * 10, 'Training finished', '-' * 10)
