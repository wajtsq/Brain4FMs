import torch
from argparse import Namespace


def draw_ten_sec_all(all_data):
    all_data = torch.from_numpy(all_data).float()
    cosSimilarity = torch.nn.CosineSimilarity(dim=-1)
    batch_size = len(all_data)
    all_sim_matrix = 0

    for i in range(batch_size):
        data = all_data[i]
        # data.size(): time_span * channel_num * dim
        data = data.permute(1,0,2)
        data = data.reshape(data.size(0), data.size(1)*data.size(2))

        source = torch.repeat_interleave(data, data.size(0), dim=0)
        target = data.repeat(data.size(0), 1)
        dot_product = cosSimilarity(source, target)
        sim_matrix = dot_product.reshape(data.size(0), data.size(0))
        all_sim_matrix += sim_matrix

    average_sim_matrix = all_sim_matrix / batch_size
    return average_sim_matrix

def similarity_mean_eeg(args: Namespace):
    from utils.meta_info import get_data_dict
    print('\n', '-'*20,'build coarse-grained correlation matrix','-'*20)
    tr_x_list, _ = get_data_dict[args.dataset](args, step='train')
    x = tr_x_list[0]
    if isinstance(x, dict):
        x = x['x']
    bsz, ch_num, N = x.shape
    if N % args.patch_len != 0:
        args.seq_len = int(N // args.patch_len)
        x = x[:, :, :args.seq_len*args.patch_len]
    x = x.reshape(bsz, ch_num, -1, args.patch_len)
    sim_matrix_all = draw_ten_sec_all(x)
    del tr_x_list
    return sim_matrix_all