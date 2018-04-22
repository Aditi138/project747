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

    def forward(self, input_batch):
        output = self.position_encoding(input_batch)
        return output

class LayerNormalization(nn.Module):
    def __init__(self, input_dim, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(input_dim), requires_grad=True)

    def forward(self, input_batch):
        mu = torch.mean(input_batch, keepdim=True, dim=-1)
        sigma = torch.std(input_batch, keepdim=True, dim=-1)
        ln_out = (input_batch - mu.expand_as(input_batch)) / (sigma.expand_as(input_batch) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

class SeparableConvolution(nn.Module):
    def __init__(self, input_dim, kernel_size):
        super(SeparableConvolution, self).__init__()
        self.depthwise = nn.Conv1d(input_dim, input_dim, kernel_size, groups=input_dim, padding=(kernel_size - 1)/2)
        self.pointwise = nn.Conv1d(input_dim, input_dim, 1)
        self.layer_norm = LayerNormalization(input_dim)

    def forward(self, input_batch, mask):
        depth_output = self.depthwise(input_batch)
        pointwise_output = self.pointwise(depth_output)
        highway_output = pointwise_output + input_batch
        norm_output = self.layer_norm(highway_output)
        masked_output = norm_output * mask
        return masked_output

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, input_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(input_dim, 0.5)
        self.softmax = nn.Softmax

    def forward(self, queries, keys, values, attn_mask=None):

        attn = torch.bmm(queries, keys.transpose(1, 2)) / self.temper

        if attn_mask is not None:
            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        output = torch.bmm(attn, values)

        return output, attn

class SelfAttention(nn.Module):
    def __init__(self, number_heads, input_dim):
        super(SelfAttention, self).__init__()
        self.number_heads = number_heads
        self.input_dim = input_dim
        self.head_dim = input_dim / number_heads

        self.query_projection = nn.Parameter(torch.FloatTensor(number_heads, input_dim, self.head_dim))
        self.key_projection = nn.Parameter(torch.FloatTensor(number_heads, input_dim, self.head_dim))
        self.value_projection = nn.Parameter(torch.FloatTensor(number_heads, input_dim, self.head_dim))

        self.attention = ScaledDotProductAttention(input_dim)
        self.linear_projection = nn.Linear(input_dim, input_dim)

        self.layer_norm = LayerNormalization(input_dim)

        nn.init.xavier_normal(self.query_projection)
        nn.init.xavier_normal(self.key_projection)
        nn.init.xavier_normal(self.value_projection)

    def forward(self, input_batch, attn_mask=None):

        head_dim = self.head_dim
        number_heads = self.number_heads
        batch_size, seq_len, input_dim = input_batch.size()

        # treat as a (number_heads) size batch
        transformed_batch = input_batch.repeat(number_heads, 1, 1).view(number_heads, -1, input_dim) # number_heads x (b_size*len_q) x input_dim
        
        # treat the result as a (number_heads * mb_size) size batch
        queries = torch.bmm(transformed_batch, self.query_projection).view(-1, input_dim, head_dim)
        keys = torch.bmm(transformed_batch, self.key_projection).view(-1, input_dim, head_dim)   
        values = torch.bmm(transformed_batch, self.value_projection).view(-1, input_dim, head_dim)  

        # perform attention, result size = (number_heads * b_size) x input_dim
        outputs, attns = self.attention(queries, keys, values, attn_mask=attn_mask.repeat(number_heads, 1, 1))

        # back to original size batch, result size = b_size x seq len x input_dim
        outputs = torch.cat(torch.split(outputs, batch_size, dim=0), dim=-1) 

        # project back to residual size
        outputs = self.linear_projection(outputs)
        highway_outputs = outputs + input_batch
        norm_outputs = self.layer_norm(highway_outputs)

        return norm_outputs


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.inner_layer = nn.Conv1d(input_dim, hidden_dim, 1) # position-wise
        self.outer_layer = nn.Conv1d(hidden_dim, input_dim, 1) # position-wise
        self.layer_norm = LayerNormalization(input_dim)
        self.relu = nn.ReLU()

    def forward(self, input_batch):
        inner_output = self.inner_layer(input_batch.transpose(1,2))
        activation_output = self.relu(inner_output)
        outer_output = self.outer_layer(activation_output).transpose(2, 1)
        highway_output = outer_output + input_batch
        norm_output = self.layer_norm(highway_output)
        return norm_output

class EncoderBlock(nn.Module):
    def __init__(self, max_positions, input_dim, kernel_size, n_conv, attention_heads):
        super(EncoderBlock, self).__init__()
        self.position_encoding = PositionEncoding(max_positions, input_dim)
        self.convolution_layers = nn.ModuleList([SeparableConvolution(input_dim, kernel_size) for _ in range(n_conv)])
        self.attention_layer = SelfAttention(attention_heads, input_dim)

    def forward(self, input):
        pos_encoded = self.position_encoding(input)
        conv_output = pos_encoded
        for convolution_layer in self.convolution_layers:
            conv_output = convolution_layer(conv_output)
        
        attended_output = self.attention_layer(conv_output)
        output = self.feedforward_layer(attended_output)

        return output
        
