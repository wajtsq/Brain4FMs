import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, weighted=False):
        super(LinearClassifier, self).__init__()
        self.input_dim = input_dim
        self.layer_num = len(hidden_dim)
        self.predictors = nn.ModuleList()
        if weighted:
            self.lossCriterion = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 1]))
        else:
            self.lossCriterion = nn.CrossEntropyLoss()

        for i in range(self.layer_num):
            if i == 0:
                self.predictors.append(nn.Linear(input_dim, hidden_dim[0]))
            else:
                self.predictors.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))

    def forward(self, data, label, pred_out=False):
        # data.size(): batch_size * time_span * channel_num * dim
        # label.size(): batch_size * time_span * channel_num
        data = data.view(-1, data.size(-1))
        label = label.view(-1)
        for i in range(self.layer_num):
            data = self.predictors[i](data)
            if i != self.layer_num - 1:
                data = F.relu(data)
        loss = self.lossCriterion(data, label).view(1, -1)
        acc = torch.eq(data.argmax(dim=1), label).sum() / label.size(0)
        acc = acc.view(1, -1)

        if pred_out:
            true_label = label
            pred_label = torch.argmax(data, dim=1).view(-1)
            return loss, acc, true_label, pred_label
        else:
            return loss, acc


class SimpleLSTM(nn.Module):
    def __init__(self,
                 dim_encoded,  # the dimension of input
                 dim_output,   # the dimension of output (hidden layer)
                 bi_direction,
                 ):
        super(SimpleLSTM, self).__init__()
        self.baseNet = nn.LSTM(dim_encoded, dim_output,
                               num_layers=1, batch_first=False, bidirectional=bi_direction)
        self.hidden = None

    def forward(self, x):
        try:
            self.baseNet.flatten_parameters()
        except RuntimeError:
            pass
        x, h = self.baseNet(x, self.hidden)  # (h_0, c_0) is None
        return x


class BertSelfAttention(nn.Module):
    def __init__(
            self,
            hidden_size=256,
            num_attention_heads=8,
            attention_probs_dropout_prob=0.1,
    ):
        super(BertSelfAttention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs.detach()) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(
            self,
            hidden_size=256,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
    ):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(
            self,
            hidden_size=256,
            num_attention_heads=8,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
    ):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )
        self.output = BertSelfOutput(
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob
        )
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class DownstreamCriterion(nn.Module):
    def __init__(
            self,
            input_dim,
            bi_direction=False,
    ):
        super(DownstreamCriterion, self).__init__()
        self.downstream_ar = SimpleLSTM(
            dim_encoded=input_dim,
            dim_output=input_dim,
            bi_direction=bi_direction,
        )
        self.attention = BertAttention(
            hidden_size=input_dim,
            num_attention_heads=8,
        )

    def forward(self, x):
        # x.size(): time_span * channel_num * input_dim
        after_ar = self.downstream_ar(x)
        after_attention = self.attention(
            hidden_states=after_ar,
            attention_mask=None,
            output_attentions=False,
        )
        return after_attention[0]


class LinearClassifier4EEG(nn.Module):
    def __init__(self, input_dim, hidden_dim, weighted=False):
        super(LinearClassifier4EEG, self).__init__()
        self.input_dim = input_dim
        self.layer_num = len(hidden_dim)
        self.predictors = nn.ModuleList()
        if weighted:
            self.lossCriterion = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 1]))
        else:
            self.lossCriterion = nn.CrossEntropyLoss()

        for i in range(self.layer_num):
            if i == 0:
                self.predictors.append(nn.Linear(input_dim, hidden_dim[0]))
            else:
                self.predictors.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))

        self.attention = nn.Linear(input_dim, 1)
        self.softmax_act = nn.Softmax(dim=1)

    def forward(self, data, label, pred_out=False, cal_auc=False):

        # only max-pooling
        last_rep = data[:, -1]
        patient_in_clip = torch.max(last_rep, dim=-2)[0]

        label = label.view(-1, )

        for i in range(self.layer_num):
            patient_in_clip = self.predictors[i](patient_in_clip)
            if i != self.layer_num - 1:
                patient_in_clip = F.relu(patient_in_clip)

        loss = self.lossCriterion(patient_in_clip, label)

        return loss, patient_in_clip
