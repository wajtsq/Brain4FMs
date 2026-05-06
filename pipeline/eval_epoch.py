import time

import numpy as np
import torch
from tqdm import tqdm

from utils.data_info import data_info_dict
from utils.meta_info import dataset_class_dict, metrics_dict
from utils.misc import update_logs, show_logs, make_dir_if_not_exist


def evaluate_epoch(args, x_list, y_list, model, clsf, loss_func, step):
    assert step == 'valid' or step == 'test'
    if args.run_mode == 'few-shot' and step == 'test':
        args.run_mode = 'test'

    model.eval()
    clsf.eval()
    device = next(model.parameters()).device

    start_time = time.perf_counter()
    epo_logs = {}
    batch_cnt = 0
    epo_loss = 0

    if args.run_mode == 'finetune' or args.run_mode == 'test':
        epo_y = torch.tensor([], dtype=torch.long)
        epo_pred = torch.tensor([], dtype=torch.long)
        epo_logit = torch.tensor([], dtype=torch.float32)

    file_num = len(x_list)
    for file_idx in range(file_num):
        x = x_list[file_idx]
        y = y_list[file_idx]
        
        if args.model == 'NeuroLM' or args.model == 'BFM':
            valid_dataset = dataset_class_dict[args.model](args, x, y, is_train=False)
        else:
            valid_dataset = dataset_class_dict[args.model](args, x, y)
        valid_loader = valid_dataset.get_data_loader(args.batch_size, shuffle=False, num_workers=0)

        if args.run_mode == 'finetune' or args.run_mode == 'test':
            file_y = torch.tensor([], dtype=torch.long)
            file_pred = torch.tensor([], dtype=torch.long)
            file_logit = torch.tensor([], dtype=torch.float32)
            
        with torch.no_grad():
            for batch_id, data_packet in enumerate(tqdm(valid_loader, disable=args.tqdm_dis, desc=f'file{file_idx}/{file_num}')):
                # x: (bsz, ch_num, seq_len, patch_len)
                # y: (bsz, )
                data_packet = [d.to(device) for d in data_packet]

                if args.run_mode == 'finetune':
                    if args.is_parallel:
                        batch_loss, logit, y = model.module.forward_propagate(args, data_packet, 
                                                                        model, clsf, loss_func)
                    else:
                        batch_loss, logit, y = model.forward_propagate(args, data_packet,
                                                                   model, clsf, loss_func)
                    epo_loss += batch_loss.detach().cpu().numpy()

                elif args.run_mode == 'test':
                    if args.is_parallel:
                        logit, y = model.module.forward_propagate(args, data_packet,
                                                       model, clsf)
                    else:
                        logit, y = model.forward_propagate(args, data_packet,
                                                       model, clsf)

                if args.run_mode == 'finetune' or args.run_mode == 'test':
                    pred = torch.argmax(logit, dim=-1)
                    file_y     = torch.cat([file_y,     y.cpu()], dim=0)
                    file_pred  = torch.cat([file_pred,  pred.detach().cpu()], dim=0)
                    file_logit = torch.cat([file_logit, logit.detach().cpu()], dim=0)

                    batch_cnt += 1

        valid_dataset.reload_pool.close()
        epo_y     = torch.cat([epo_y,     file_y.cpu()], dim=0)
        epo_pred  = torch.cat([epo_pred,  file_pred.detach().cpu()], dim=0)
        epo_logit = torch.cat([epo_logit, file_logit.detach().cpu()], dim=0)

    if args.run_mode == 'finetune' or args.run_mode == 'test':
        metrics = metrics_dict[args.dataset](args, epo_pred.cpu(), epo_logit.cpu(), epo_y.cpu())
    else:
        metrics = None
    epo_loss /= batch_cnt
    epo_logs = update_logs(args, epo_logs, epo_loss, metrics)

    if step == 'valid':
        show_logs('[Valid]', epo_logs, f"{(time.perf_counter()-start_time):.1f}s")
        if args.run_mode == 'finetune':
            print(metrics.conf_matrix)
    elif step == 'test':
        show_logs('[Test]', epo_logs, None)
        epo_logs[f"matrix"] = str(metrics.conf_matrix)
        print(metrics.conf_matrix)

    return epo_logs, epo_loss
