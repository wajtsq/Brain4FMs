import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
import pickle
from argparse import Namespace

from model.Mbrain.models.transformer_layer import BertModel
from model.Mbrain.models.weighted_gcn import GCNConv
from model.Mbrain.models.utils import similarity_mean_eeg
# from .plot_api import draw_heatmap


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num=1):
        super(GCN, self).__init__()
        self.layer_num = layer_num
        assert(len(hidden_dim) >= layer_num)
        for i in range(layer_num):
            if i == 0:
                setattr(self, 'conv' + str(i), GCNConv(input_dim, hidden_dim[i], add_self_loops=False))
            else:
                setattr(self, 'conv' + str(i), GCNConv(hidden_dim[i-1], hidden_dim[i], add_self_loops=False))

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        all_hidden_representation = []
        for i in range(self.layer_num):
            # delete relu function in the last layer
            x = getattr(self, 'conv' + str(i))(x, edge_index, edge_weight)
            if i != self.layer_num - 1:
                x = F.relu(x)
                # x = F.dropout(x, p=0.5)
            all_hidden_representation.append(x)
            # x = F.relu(getattr(self, 'conv' + str(i))(x, edge_index, edge_weight))
            # all_hidden_representation.append(x)
            # if i != self.layer_num - 1:
            #     x = F.dropout(x, p=0.5)
        return all_hidden_representation


class ChannelNorm(nn.Module):
    def __init__(self, num_features, epsilon=1e-05, affine=True):
        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1, num_features, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        cum_mean = x.mean(dim=1, keepdim=True)
        cum_var = x.var(dim=1, keepdim=True)
        x = (x - cum_mean) * torch.rsqrt(cum_var + self.epsilon)

        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class Encoder(nn.Module):
    def __init__(self,
                 hidden_size,
                 kernel_size,
                 stride_size,
                 padding_size):
        super(Encoder, self).__init__()
        norm_layer = ChannelNorm

        # 250 HZ
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size[0], stride=stride_size[0], padding=padding_size[0])
        self.batchNorm1 = norm_layer(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size[1], stride=stride_size[1], padding=padding_size[1])
        self.batchNorm2 = norm_layer(hidden_size)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size[2], stride=stride_size[2], padding=padding_size[2])
        self.batchNorm3 = norm_layer(hidden_size)
        self.DOWNSAMPLING = 4

    def get_dim_output(self):
        return self.conv3.out_channels

    def forward(self, x):
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        return x


class ARModel(nn.Module):
    def __init__(self,
                 dim_encoded,  # the dimension of input
                 dim_output,   # the dimension of output (hidden layer)
                 n_predicts,   # total steps for forecasting
                 keep_hidden,  # keep hidden layer's value during backend
                 nLevelsAR,    # stacking depth
                 mode='GRU'):
        super(ARModel, self).__init__()
        self.RESIDUAL_STD = 0.1
        if mode == "LSTM":
            self.baseNet = nn.LSTM(dim_encoded, dim_output,
                                   num_layers=nLevelsAR, batch_first=True, bidirectional=False)
        elif mode == "RNN":
            self.baseNet = nn.RNN(dim_encoded, dim_output,
                                  num_layers=nLevelsAR, batch_first=True, bidirectional=True)
        elif mode == "GRU":
            self.baseNet = nn.GRU(dim_encoded, dim_output,
                                  num_layers=nLevelsAR, batch_first=True, bidirectional=True)
        else:
            # seq_len based on the sequence length after encoder
            self.baseNet = BertModel(
                input_size=dim_encoded,
                n_predicts=n_predicts,
                seq_len=45,
                hidden_size=dim_output,
                position_embedding_type='static'
            )
        self.ar_mode = mode
        self.hidden = None
        self.keepHidden = keep_hidden

    def get_dim_output(self):
        return self.baseNet.hidden_size

    def forward(self, x, mask_state):
        if self.ar_mode == 'TRANSFORMER':
            x = self.baseNet(
                x,
                mask_state=mask_state
            )
        else:
            try:
                self.baseNet.flatten_parameters()
            except RuntimeError:
                pass
            x, h = self.baseNet(x, self.hidden)  # (h_0, c_0) is None
            if self.keepHidden:
                if isinstance(h, tuple):
                    self.hidden = tuple(x.detach() for x in h)
                else:
                    self.hidden = h.detach()
        return x


class MBrain(nn.Module):
    def __init__(self,
                 hidden_dim,            #
                 channel_num,           #
                 gcn_dim,               #
                 n_predicts,            # total steps for forecasting
                 graph_construct,
                 direction,
                 replace_ratio,    # replace ratio in replacement task
                 ar_mode,         #
                 args: Namespace):
        super(MBrain, self).__init__()
        self.hidden_dim = hidden_dim
        self.channel_num = channel_num
        self.gcn_dim = gcn_dim
        self.nPredicts = n_predicts
        self.graph_construct = graph_construct
        self.direction = direction
        self.replace_ratio = replace_ratio
        self.ar_mode = ar_mode

        # one Encoder and one Gar new_model for all channel
        self.gEncoder = Encoder(hidden_size=hidden_dim,
                                kernel_size=args.kernel_size,
                                stride_size=args.stride_size,
                                padding_size=args.padding_size)
        self.gAR = ARModel(hidden_dim, hidden_dim, n_predicts, False, 1, mode=ar_mode)

        self.fGCN = GCN(hidden_dim, gcn_dim, len(self.gcn_dim))
        if self.direction == 'bi':
            self.bGCN = GCN(hidden_dim, gcn_dim, len(self.gcn_dim))

        if graph_construct == 'sample_from_distribution':
            self.linear_out = nn.Linear(hidden_dim * 2, hidden_dim)
            self.linear_cat = nn.Linear(hidden_dim, 1)

            # predefined mean matrix of similarity
            self.mean_matrix = similarity_mean_eeg(args)
            self.threshold = args.graph_threshold

            self.mean_matrix = self.mean_matrix.cuda(args.gpu_id)
            self.Softplus = torch.nn.Softplus()

        elif graph_construct == 'predefined_distance':
            # Predefined distance graph
            with open('utils/adj_mx_3d.pkl', 'rb') as pf:
                adj_mx_data = pickle.load(pf)
            adj_mx = adj_mx_data[2]
            adj_mx = torch.from_numpy(adj_mx)
            mask_matrix = torch.ones((adj_mx.size(0), adj_mx.size(0))) - torch.eye(adj_mx.size(0))
            adj_mx = adj_mx * mask_matrix

            self.distance_graph = adj_mx.cuda()


    def sample_replace_data(self, encoded_data):
        batch_size, channel_num, seq_size, dim_en = encoded_data.size()
        tot_num = batch_size * channel_num * seq_size
        replace_num = int(tot_num * self.replace_ratio)
        source_idx = np.random.choice(tot_num, replace_num, replace=False)
        target_idx = np.random.choice(tot_num, replace_num, replace=False)

        neg_ext = encoded_data.contiguous().view(-1, dim_en)
        idx = np.arange(tot_num)
        idx[target_idx] = source_idx
        replace_data = neg_ext[idx].view(batch_size, channel_num, seq_size, dim_en)

        # 0 : not replaced || 1 : replaced
        # if_replace = (source_idx // seq_size) != (target_idx // seq_size)
        if_replace = (source_idx // seq_size % channel_num) != (target_idx // seq_size % channel_num)
        replace_label = torch.zeros((tot_num), dtype=torch.long, device=encoded_data.device)
        replace_label[target_idx] = torch.tensor(if_replace, dtype=torch.long, device=encoded_data.device)

        return replace_data, replace_label

    def diffusion_module(self, after_gAR_batch, forward=True):
        batch_size, node_num = after_gAR_batch.size()[:2]
        # after_gAR_batch.size(): (batch_size * time_span) * channel_num * dim_ar

        if self.graph_construct == 'sample_from_distribution':
            source_nodes = after_gAR_batch.view(batch_size, -1, 1, after_gAR_batch.size(-1)).expand(-1, -1, node_num, -1)
            target_nodes = after_gAR_batch.view(batch_size, 1, -1, after_gAR_batch.size(-1)).expand(-1, node_num, -1, -1)
            x = torch.cat([source_nodes, target_nodes], dim=-1)
            x = F.relu(self.linear_out(x))
            x = self.linear_cat(x)
            var_matrix = x.view(batch_size, node_num, node_num)

            normal_distribution = torch.randn_like(var_matrix, device=var_matrix.device)
            sample_weight = self.mean_matrix + self.Softplus(var_matrix) * normal_distribution

            mask_matrix = torch.ones((node_num, node_num)) - torch.eye(node_num)
            mask_matrix = mask_matrix.cuda()
            sample_weight = sample_weight * mask_matrix

            i, j, k = torch.where(sample_weight >= self.threshold)

            idx_shift = torch.tensor([batch_num * node_num for batch_num in range(batch_size)],
                                      dtype=torch.long, device=sample_weight.device)

            weights = sample_weight[i, j, k]
            j = j + idx_shift[i]
            k = k + idx_shift[i]

            edge_index = torch.stack([j, k], dim=0)
            graph = Data(x=after_gAR_batch.contiguous().view(-1, after_gAR_batch.size(-1)), edge_index=edge_index,
                         edge_attr=weights)

            if forward:
                n_predicts_representation = self.fGCN(graph)
            else:
                n_predicts_representation = self.bGCN(graph)

            # one-layer GNN use
            after_gnn_batch = torch.stack(n_predicts_representation, dim=0)
            after_gnn_batch = after_gnn_batch.view(batch_size, 1, node_num, after_gnn_batch.size(-1))


            edge_num = len(i)
            edge_weight = torch.sum(weights).item()

            random_number = np.random.randint(0,300)
            if random_number == 0:
                if edge_num:
                    print("Average number of edges: %d; Average weight of edges: %.2f" % (
                    (edge_num / batch_size), (edge_weight / edge_num)))
                elif not edge_num:
                    print("None edge!")

            return after_gnn_batch

        elif self.graph_construct == 'predefined_distance':
            j, k = torch.where(self.distance_graph > 0)
            weights = self.distance_graph[j, k]
            edge_num = len(j)

            j = j.repeat(batch_size)
            k = k.repeat(batch_size)
            idx_shift = torch.tensor([batch_num * node_num for batch_num in range(batch_size)],
                                     dtype=torch.long, device=weights.device)
            idx_shift = torch.repeat_interleave(idx_shift, edge_num)
            j = j + idx_shift
            k = k + idx_shift

            edge_index = torch.stack([j, k], dim=0)
            weights = weights.repeat(batch_size)
            graph = Data(x=after_gAR_batch.contiguous().view(-1, after_gAR_batch.size(-1)), edge_index=edge_index,
                         edge_attr=weights)

            if forward:
                n_predicts_representation = self.fGCN(graph)
            else:
                n_predicts_representation = self.bGCN(graph)

            after_gnn_batch = torch.stack(n_predicts_representation, dim=0)
            after_gnn_batch = after_gnn_batch.view(batch_size, 1, node_num, after_gnn_batch.size(-1))

            return after_gnn_batch
        else:
            raise Exception("Other graph learning modules code have not been optimized!")


    def forward(self, batch_data, train_stage=True):
        # batch_data.size(): BatchSize * ChannelNum * SeqLength
        # direction of AR new_model: ['single', 'bi']

        batch_size, channel_num, seq_len = batch_data.size()
        batch_data = batch_data.view(-1, 1, seq_len)
        after_encoder = self.gEncoder(batch_data).permute(0, 2, 1)
        after_gAR = self.gAR(after_encoder, mask_state=self.direction)

        after_encoder_batch = after_encoder.view(batch_size, channel_num, after_encoder.size(-2), after_encoder.size(-1))
        after_gAR_batch = after_gAR.view(batch_size, channel_num, after_gAR.size(-2), after_gAR.size(-1))

        ##################################################################
        ########################### GCN Module ###########################
        ##################################################################
        after_gnn_batch = []
        batch_size, channel_num, seq_size, dim_ar = after_gAR_batch.size()
        # if ar_mode != 'TRANSFORMER': dim_ar = hidden_dim * 2

        if self.graph_construct != 'NoGraph':
            if train_stage:
                if self.direction == 'single':
                    window_size = seq_size - self.nPredicts
                    after_gnn_batch.append(self.diffusion_module(after_gAR_batch[:, :, window_size - 1, :self.hidden_dim], True))
                elif self.direction == 'bi':
                    # consider the forward direction
                    window_size = seq_size - (self.nPredicts // 2)
                    after_gnn_batch.append(self.diffusion_module(after_gAR_batch[:, :, window_size - 1, :self.hidden_dim], True))
                    # consider the backward direction
                    after_gnn_batch.append(self.diffusion_module(after_gAR_batch[:, :, self.nPredicts//2, -self.hidden_dim:], False))
            else:
                if self.direction == 'single':
                    pass
                    # after_gnn_batch.append(self.diffusion_module(after_gAR_batch[:, :, -1, :self.hidden_dim], True))
                elif self.direction == 'bi':
                    after_gnn_batch.append(self.diffusion_module(after_gAR_batch[:, :, -1, :self.hidden_dim], True))
                    after_gnn_batch.append(self.diffusion_module(after_gAR_batch[:, :, 0, -self.hidden_dim:], False))

        ##################################################################
        ################### Whether Replaced Prediction ##################
        ##################################################################
        if train_stage:
            replace_data, replace_label = self.sample_replace_data(after_encoder_batch)

            batch_data, channel_num, seq_len, enc_dim = replace_data.size()
            replace_data = replace_data.view(-1, seq_len, enc_dim)
            reaplce_after_gAR = self.gAR(replace_data, mask_state='single')
            reaplce_after_gAR = reaplce_after_gAR.contiguous().view(batch_size, channel_num, after_gAR.size(-2), after_gAR.size(-1))

            return after_encoder_batch, after_gAR_batch, after_gnn_batch, reaplce_after_gAR, replace_label
        else:
            return after_encoder_batch, after_gAR_batch, after_gnn_batch