import time

import torch
import torch.nn.functional as F

from utils.meta_info import dataset_class_dict, metrics_dict
from utils.misc import update_logs, show_logs


def get_prototype(args, model, tr_x, tr_y, clsf, loss_func):
    if args.model == 'NeuroLM' or args.model == 'BFM':
        x_dataset = dataset_class_dict[args.model](args, tr_x, tr_y, is_train=False)
    else:
        x_dataset = dataset_class_dict[args.model](args, tr_x, tr_y)
    sample_loader = x_dataset.get_data_loader(args.batch_size, shuffle=False, num_workers=0)
    model.eval()
    labels = torch.tensor([], dtype=torch.int)
    representations = torch.tensor([], dtype=torch.float)
    with torch.no_grad():
        for data_packet in sample_loader:
            data_packet = [d.to(args.device) for d in data_packet]
            if args.is_parallel:
                z, logit, y = model.module.forward_propagate(args, data_packet, model, clsf, loss_func)
            else:
                z, logit, y = model.forward_propagate(args, data_packet, model, clsf, loss_func)
            representations = torch.cat((representations, z.cpu()), dim=0)
            labels = torch.cat((labels, y.cpu()), dim=0)

    prototypes = []
    for label in range(args.n_class):
        idx = torch.where(labels == label)[0]
        prototype = torch.mean(representations[idx], dim=0, keepdim=True)
        prototypes.append(prototype)

    return torch.cat(prototypes, dim=0)


def contrast(args, model, x, y, prototype, clsf, loss_func):
    epo_logs = {}
    start_time = time.perf_counter()
    if args.model == 'NeuroLM' or args.model == 'BFM':
        x_dataset = dataset_class_dict[args.model](args, x, y, is_train=False)
    else:
        x_dataset = dataset_class_dict[args.model](args, x, y)
    infer_loader = x_dataset.get_data_loader(args.batch_size, shuffle=False, num_workers=0)
    model.eval()
    true_label = torch.tensor([], dtype=torch.int)
    pred_score = torch.tensor([], dtype=torch.float)
    if args.device.type == 'cuda':
        torch.cuda.set_device(args.gpu_id)
    prototype = prototype.unsqueeze(0).to(args.device)

    with torch.no_grad():
        for data_packet in infer_loader:
            data_packet = [d.to(args.device) for d in data_packet]
            if args.is_parallel:
                z, logit, y = model.module.forward_propagate(args, data_packet, model, clsf, loss_func)
            else:
                z, logit, y = model.forward_propagate(args, data_packet, model, clsf, loss_func)
            true_label = torch.cat((true_label, y.cpu()), dim=0)
            rep = z.detach().unsqueeze(1)
            sim = F.cosine_similarity(rep, prototype, dim=-1)
            pred_score = torch.cat((pred_score, sim.cpu()), dim=0)

    pred = torch.argmax(pred_score, dim=-1)
    metrics = metrics_dict[args.dataset](args, pred.cpu(), pred_score.cpu(), true_label.cpu())
    epo_logs = update_logs(args, epo_logs, 0, metrics)
    show_logs('[Test]', epo_logs, f"{(time.perf_counter() - start_time):.1f}s")
    epo_logs["matrix"] = str(metrics.conf_matrix)
    print(metrics.conf_matrix)

    return epo_logs
