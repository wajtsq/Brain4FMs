import time

import torch
from tqdm import tqdm

from utils.meta_info import dataset_class_dict, metrics_dict
from utils.misc import update_logs, show_logs


def train_epoch(args, tr_x_list, tr_y_list, model, clsf, loss_func, optimizer, scheduler=None):
    model.train()
    clsf.train()
    device = next(model.parameters()).device

    start_time = time.perf_counter()
    epo_logs = {}
    batch_cnt = 0
    epo_loss = 0

    epo_y = torch.tensor([], dtype=torch.long)
    epo_pred = torch.tensor([], dtype=torch.long)
    epo_logit = torch.tensor([], dtype=torch.float32)

    bat_cnt = 0
    file_num = len(tr_x_list)
    for file_idx in range(file_num):
        tr_x = tr_x_list[file_idx]
        tr_y = tr_y_list[file_idx]

        train_dataset = dataset_class_dict[args.model](args, tr_x, tr_y)
        train_loader = train_dataset.get_data_loader(args.batch_size, shuffle=True, num_workers=0)

        for batch_id, data_packet in enumerate(tqdm(train_loader, disable=args.tqdm_dis, desc=f'file{file_idx}/{file_num}')):
            # x: (bsz, ch_num, seq_len, patch_len)
            # y: (bsz, )
            data_packet = [d.to(device) for d in data_packet]

            if args.is_parallel and args.run_mode != 'test':
                ret = model.module.forward_propagate(args, data_packet, model, clsf, loss_func)
            else:
                ret = model.forward_propagate(args, data_packet, model, clsf, loss_func)

            batch_loss, logit, y = ret

            pred = torch.argmax(logit, dim=-1)
            epo_y = torch.cat([epo_y, y.cpu()])
            epo_pred = torch.cat([epo_pred, pred.detach().cpu()], dim=0)
            epo_logit = torch.cat([epo_logit, logit.detach().cpu()], dim=0)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            batch_cnt += 1
            epo_loss += batch_loss.detach().cpu().numpy()

        if scheduler is not None:
            scheduler.step()
        train_dataset.reload_pool.close()

    if args.run_mode == 'finetune' or args.run_mode == 'test':
        metrics = metrics_dict[args.dataset](args, epo_pred.cpu(), epo_logit.cpu(), epo_y.cpu())
    else:
        metrics = None

    epo_loss /= batch_cnt
    epo_logs = update_logs(args, epo_logs, epo_loss, metrics)

    show_logs('[Train]', epo_logs, f"{(time.perf_counter()-start_time):.1f}s")
    if args.run_mode == 'finetune':
        print(metrics.conf_matrix)

    return epo_logs, epo_loss
