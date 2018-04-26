from __future__ import division
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class PositionEncoding(nn.Module):
    def __init__(self, max_positions, hidden_dim, pad_token=0):
        super(PositionEncoding, self).__init__()
        embedding_length = max_positions + 1
        self.position_encoding = nn.Embedding(embedding_length, hidden_dim, padding_idx=pad_token)

        position_weights = np.array([[pos / np.power(10000, 2 * (j // 2) / hidden_dim) for j in range(hidden_dim)]
        if pos != 0 else np.zeros(hidden_dim) for pos in range(embedding_length)])

        position_weights[1:, 0::2] = np.sin(position_weights[1:, 0::2]) # dim 2i
        position_weights[1:, 1::2] = np.cos(position_weights[1:, 1::2]) # dim 2i+1
        position_weights = torch.from_numpy(position_weights).type(torch.FloatTensor)
        self.position_encoding.weight.data = position_weights
        self.position_encoding.weight.requires_grad = False

    def forward(self, batch):
        batch_positions = torch.arange(batch.size(1)).long()
        if torch.cuda.is_available():
            batch_positions = batch_positions.cuda()
        output = self.position_encoding(batch_positions).expand(batch.size(0), -1, -1)
        return output

class LayerNormalization(nn.Module):
    def __init__(self, hidden_dim, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)

    def forward(self, input_batch):
        mu = torch.mean(input_batch, keepdim=True, dim=-1)
        sigma = torch.std(input_batch, keepdim=True, dim=-1)
        ln_out = (input_batch - mu.expand_as(input_batch)) / (sigma.expand_as(input_batch) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

class SeparableConvolution(nn.Module):
    def __init__(self, hidden_dim, kernel_size):
        super(SeparableConvolution, self).__init__()
        self.depthwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, padding=(kernel_size - 1)//2)
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.layer_norm = LayerNormalization(hidden_dim)

    def forward(self, input_batch, mask):
        input_batch = input_batch * mask
        transposed_input = input_batch.transpose(1,2)
        depth_output = self.depthwise(transposed_input)
        pointwise_output = self.pointwise(depth_output).transpose(1,2)
        highway_output = pointwise_output + input_batch
        norm_output = self.layer_norm(highway_output)
        return norm_output

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, hidden_dim, number_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(hidden_dim, 0.5)
        self.softmax = nn.Softmax(dim=2)
        self.one_scalar = nn.Parameter(torch.ones(1))
        self.number_heads = number_heads

    def forward(self, queries, keys, values, mask):

        attn = torch.bmm(queries, keys.transpose(1, 2)) / self.temper
        size = attn.size()
        repeated_mask = mask.repeat(self.number_heads,  1, 1)
        expanded_mask = repeated_mask.expand(size)
        inverted_mask = (- expanded_mask.data + self.one_scalar.data).byte()

        attn.data.masked_fill_(inverted_mask, -10**10)
        attn = self.softmax(attn)
        output = torch.bmm(attn, values)

        return output

class SelfAttention(nn.Module):
    def __init__(self, number_heads, hidden_dim):
        super(SelfAttention, self).__init__()
        self.number_heads = number_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // number_heads

        self.query_projection = nn.Parameter(torch.FloatTensor(number_heads, hidden_dim, self.head_dim))
        self.key_projection = nn.Parameter(torch.FloatTensor(number_heads, hidden_dim, self.head_dim))
        self.value_projection = nn.Parameter(torch.FloatTensor(number_heads, hidden_dim, self.head_dim))

        self.attention = ScaledDotProductAttention(hidden_dim, number_heads)
        self.linear_projection = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = LayerNormalization(hidden_dim)

        nn.init.xavier_normal(self.query_projection)
        nn.init.xavier_normal(self.key_projection)
        nn.init.xavier_normal(self.value_projection)

    def forward(self, input_batch, mask):

        head_dim = self.head_dim
        number_heads = self.number_heads
        batch_size, seq_len, hidden_dim = input_batch.size()

        # treat as a (number_heads) size batch
        transformed_batch = input_batch.repeat(number_heads, 1, 1).view(number_heads, -1, hidden_dim) # number_heads x (b_size*len_q) x hidden_dim
        
        # treat the result as a (number_heads * mb_size) size batch
        queries = torch.bmm(transformed_batch, self.query_projection).view(-1, seq_len, head_dim)
        keys = torch.bmm(transformed_batch, self.key_projection).view(-1, seq_len, head_dim)   
        values = torch.bmm(transformed_batch, self.value_projection).view(-1, seq_len, head_dim)  

        # perform attention, result size = (number_heads * b_size) x hidden_dim
        outputs = self.attention(queries, keys, values, mask)


        # back to original size batch, result size = b_size x seq len x hidden_dim
        outputs = torch.cat(torch.split(outputs, batch_size, dim=0), dim=-1) 

        # project back to residual size
        outputs = self.linear_projection(outputs)
        masked_outputs = outputs * mask
        highway_outputs = masked_outputs + input_batch
        norm_outputs = self.layer_norm(highway_outputs)
        
        return norm_outputs


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, inner_dim):
        super(FeedForward, self).__init__()
        self.inner_layer = nn.Conv1d(hidden_dim, inner_dim, 1) # position-wise
        self.outer_layer = nn.Conv1d(inner_dim, hidden_dim, 1) # position-wise
        self.layer_norm = LayerNormalization(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, input_batch):
        inner_output = self.inner_layer(input_batch.transpose(1,2))
        activation_output = self.relu(inner_output)
        outer_output = self.outer_layer(activation_output).transpose(2, 1)
        highway_output = outer_output + input_batch
        norm_output = self.layer_norm(highway_output)
        return norm_output

class EncoderBlock(nn.Module):
    def __init__(self, max_positions, hidden_dim, kernel_size, n_conv, attention_heads):
        super(EncoderBlock, self).__init__()
        self.position_encoding = PositionEncoding(max_positions, hidden_dim)
        self.convolution_layers = nn.ModuleList([SeparableConvolution(hidden_dim, kernel_size) for _ in range(n_conv)])
        self.attention_layer = SelfAttention(attention_heads, hidden_dim)
        self.feedforward_layer = FeedForward(hidden_dim, 2 * hidden_dim)

    def forward(self, input_batch, mask):
        
        pos_encoded = self.position_encoding(input_batch)
        conv_output = input_batch + pos_encoded        
        for convolution_layer in self.convolution_layers:
            conv_output = convolution_layer(conv_output, mask)
        attended_output = self.attention_layer(conv_output, mask)
        output = self.feedforward_layer(attended_output)

        return output

class EncoderBlocks(nn.Module):
    def __init__(self, n_blocks, max_positions, hidden_dim, kernel_size, n_conv, attention_heads):
        super(EncoderBlocks, self).__init__()
        self.encoder_blocks = nn.ModuleList([EncoderBlock(max_positions, hidden_dim, kernel_size, n_conv, attention_heads) for _ in range(n_blocks)])
    def forward(self, input_batch, mask):
        block_output = input_batch
        for encoder_block in self.encoder_blocks:
            block_output = encoder_block(block_output, mask)
        return block_output
