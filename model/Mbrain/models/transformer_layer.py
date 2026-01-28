import torch
import torch.nn as nn
import math

# from transformers.modeling_utils import apply_chunking_to_forward
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.activations import ACT2FN


class StaticPositionEmbedding(nn.Module):
    def __init__(
            self,
            seq_len,
            d_model,
    ):
        super(StaticPositionEmbedding, self).__init__()
        pos = torch.arange(0., seq_len).unsqueeze(1).repeat(1, d_model)
        dim = torch.arange(0., d_model).unsqueeze(0).repeat(seq_len, 1)
        div = torch.exp(- math.log(10000) * (2 * (dim // 2) / d_model))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        self.register_buffer('pe', pos.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class AbsolutePositionEmbedding(nn.Module):
    def __init__(
            self,
            seq_len,
            input_size,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
    ):
        super(AbsolutePositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(input_size, eps=layer_norm_eps)
        self.position_embeddings = nn.Embedding(seq_len, input_size)

        self.register_buffer("position_ids", torch.arange(seq_len).expand((1, -1)))

    def forward(self, x):
        position_embeddings = self.position_embeddings(self.position_ids)
        embeddings = x + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertEmbeddings(nn.Module):
    def __init__(
            self,
            input_size=256,
            seq_len=62,
            position_embedding_type='absolute',
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
    ):
        super(BertEmbeddings, self).__init__()
        if position_embedding_type == 'absolute':
            self.position_embeddings = AbsolutePositionEmbedding(seq_len, input_size, layer_norm_eps, hidden_dropout_prob)
        else:
            self.position_embeddings = StaticPositionEmbedding(seq_len, input_size)
        self.seq_len = seq_len

    def forward(
            self,
            inputs_embeds,
    ):
        # inputs_embeds.size(): BatchSize x SeqLen x InputSize
        assert inputs_embeds.size(1) == self.seq_len
        embeddings = self.position_embeddings(inputs_embeds)

        return embeddings


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


class BertIntermediate(nn.Module):
    def __init__(
            self,
            hidden_size=256,
            intermediate_size=256,
            hidden_act='gelu',
    ):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(
            self,
            intermediate_size=256,
            hidden_size=256,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
    ):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(
            self,
            hidden_size=256,
            num_attention_heads=8,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            intermediate_size=256,
            hidden_act='gelu',
            chunk_size_feed_forward=0,
    ):
        super(BertLayer, self).__init__()
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob
        )
        self.intermediate = BertIntermediate(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act
        )
        self.output = BertOutput(
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob
        )

    def forward(self,
                hidden_states,
                attention_mask,
                output_attentions=False,
                ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertModel(nn.Module):
    def __init__(
            self,
            input_size=256,
            n_predicts=12,
            seq_len=62,
            hidden_size=256,
            position_embedding_type='static',
            num_attention_heads=8,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            intermediate_size=512,
            hidden_act='gelu',
            chunk_size_feed_forward=0,
    ):
        super(BertModel, self).__init__()
        self.embeddings = BertEmbeddings(
            input_size=input_size,
            seq_len=seq_len,
            position_embedding_type=position_embedding_type,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
        )
        self.encoder = BertLayer(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            chunk_size_feed_forward=chunk_size_feed_forward,
        )
        self.nPredicts = n_predicts
        self.seqLen = seq_len
        self.position_embedding_type = position_embedding_type

    @staticmethod
    def get_extended_attention_mask(batch_size, seq_len, attention_mask, x):
        # Construct the top triangle matrix
        seq_ids = torch.arange(seq_len, device=x.device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_len, 1) <= seq_ids[None, :, None]
        causal_mask = causal_mask.to(attention_mask.dtype)
        # The second is the dimension of attention heads in the self-attention module
        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=x.dtype)
        # Make the masked positions very low
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e4
        # extended_attention_mask.size(): BatchSize x 1 x SeqLen x SeqLen

        return extended_attention_mask

    def forward(self, x, mask_state='single', output_attentions=False):
        # x.size(): BatchSize x SeqLen x InputSize
        embeddings_output = self.embeddings(x)

        batch_size, seq_len = x.size()[:-1]
        assert(mask_state in ['single', 'bi'])

        if mask_state == 'single':
            attention_mask = torch.ones(size=(batch_size, seq_len), device=x.device)
            extended_attention_mask = self.get_extended_attention_mask(batch_size, seq_len, attention_mask, x)
        else:
            attention_mask = torch.ones(size=(batch_size, seq_len // 2), device=x.device)
            half_extended_attention_mask = self.get_extended_attention_mask(batch_size, seq_len//2, attention_mask, x)
            # Construct the full four-corner masked matrix
            bottom_right = half_extended_attention_mask
            bottom_left = torch.flip(half_extended_attention_mask, dims=[-1])
            bottom_matrix = torch.cat((bottom_left, bottom_right), dim=-1)
            top_matrix = torch.flip(bottom_matrix, dims=[-2])
            extended_attention_mask = torch.cat((top_matrix, bottom_matrix), dim=-2)

        encoder_output = self.encoder(embeddings_output, extended_attention_mask, output_attentions)

        return encoder_output if output_attentions else encoder_output[0]


if __name__ == '__main__':
    dim_encoded = dim_output = 256
    n_predicts = 12
    model = BertModel(
        input_size=dim_encoded,
        n_predicts=n_predicts,
        seq_len=50,
        hidden_size=dim_output
    )
    data = torch.randn(size=(32, 50, 256))
    x = model(data, mask_state='bi', output_attentions=False)
    print(x.shape)

