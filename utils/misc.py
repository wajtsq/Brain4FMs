import random
import torch
import numpy as np
from copy import deepcopy
import sys
import time
import os
import psutil
import json

from data_process.data_info import data_info_dict


def process_init():
    sys.path.append('model/BrainBERT/')  # for the files in .model.BrainBERT.models


def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def update_logs(args, logs, epo_loss, metrics=None):
    if metrics is None:     # unsupervised
        logs[f"Loss"] = epo_loss
    else:                   # finetune or test
        n_class = data_info_dict[args.dataset]['n_class']
        if n_class == 2:
            logs[f"Loss"] = epo_loss
            logs[f"Acc"] = metrics.acc * 100.0
            logs[f"Prec"] = metrics.prec * 100.0
            logs[f"Rec"] = metrics.rec * 100.0
            logs[f"F2"] = metrics.f_doub * 100.0
            logs[f"AUPRC"] = metrics.auprc * 100.0
            logs[f"AUROC"] = metrics.auroc * 100.0
            logs[f'BACC'] = metrics.bacc * 100.0
            logs[f"MF1"] = metrics.f_one_macro * 100.0
            logs[f"MF2"] = metrics.f_doub_macro * 100.0
        elif n_class > 2:
            logs[f"Acc"] = metrics.accuracy * 100.0
            logs[f"Prec"] = metrics.prec * 100.0
            logs[f"Rec"] = metrics.rec * 100.0
            logs[f"Loss"] = epo_loss
            logs[f"TopKAcc"] = metrics.acc * 100.0
            logs[f"Sens"] = metrics.spec_mean * 100.0
            logs[f"Spec"] = metrics.spec_mean * 100.0
            logs[f"MF1"] = metrics.f_one_macro * 100.0
            logs[f"Kappa"] = metrics.kappa * 100.0
            logs[f"AUROC"] = metrics.auc_roc_macro * 100.0
            logs[f'BACC'] = metrics.bacc * 100.0
            logs[f"MF2"] = metrics.f_doub_macro * 100.0
        else: raise NotImplementedError(f'Illegal number of classes.')

    return logs


def show_logs(prefix, logs, time_info):

    out = prefix + '  '
    for key in logs:
        out += f"{key}:{logs[key]:8.4f}  "
    out += f'({time_info})' if time_info is not None else ''
    print(out)


def update_main_logs(logs, tr_logs, vl_logs, epoch):
    for key, value in dict(tr_logs, **vl_logs).items():
        if key not in logs:
            logs[key] = []
        if isinstance(value, np.ndarray):
            value = value.tolist()
        logs[key].append(value)
    logs["epoch"].append(epoch)
    return logs


def save_logs(data, path_logs):
    with open(path_logs, 'w') as file:
        json.dump(data, file, indent=2)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cpu_stats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())


def load_checkpoint(args, ckpt_type):
    load_path = f'{args.load_ckpt_path}/{ckpt_type}_ckpt/'

    # Load specific checkpoint
    if not os.path.isdir(load_path):
        if load_path.split('.')[-1] == 'pt':
            checkpoint_file = load_path
            load_path = os.path.split(load_path)[0]
        else:
            raise RuntimeError("Invalid checkpoints path: " + load_path)

    # If no specific checkpoint is assigned, load the newest checkpoint
    else:
        checkpoints = [x for x in os.listdir(load_path)
                       if os.path.splitext(x)[1] == '.pt'
                       and os.path.splitext(x)[0].isdigit()]
        if len(checkpoints) == 0:
            raise RuntimeError("No checkpoints found at " + load_path)

        checkpoints.sort(key=lambda x: int(os.path.splitext(x)[0]))
        checkpoint_file = os.path.join(load_path, checkpoints[-1])

    with open(os.path.join(load_path, f'logs.json'), 'rb') as file:
        logs = json.load(file)

    return os.path.abspath(checkpoint_file), logs


def save_checkpoint(model_state, clsf_state, optimizer_state, best_model_state, best_clsf_state, best_vl_loss, ckpt_path):
    state_dict = {"Model": model_state,
                  "Clsf":  clsf_state,
                  "Optimizer": optimizer_state,
                  "BestModel": best_model_state,
                  "BestClsf":  best_clsf_state,
                  "BestValLoss": best_vl_loss}

    torch.save(state_dict, ckpt_path)
