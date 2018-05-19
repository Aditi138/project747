import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from bidaf import BiDAF, LinearSeqAttn,weighted_avg,BiLinearAttn, log_sum_exp
import codecs
import numpy as np

class SentenceScorer(nn.Module):
	def __init__(self, args, loader):
		super(SentenceScorer, self).__init__()
		word_vocab_size = loader.vocab.get_length()
		embed_size = args.embed_size
		hidden_size = args.hidden_size

		##  Embedding Layer
		self.word_embedding_layer = LookupEncoder(word_vocab_size, embedding_dim=embed_size,
												  pretrain_embedding=loader.pretrain_embedding)
		## Encoder Layer
		self.question_encoder = EncoderBlock(embed_size, hidden_size, 3)
		self.chunk_encoder = EncoderBlock(embed_size, hidden_size, 3)
		self.answer_encoder = EncoderBlock(embed_size, hidden_size, 3)

		## Output Layer
		self.modeling_layer = MLP(args.hidden_size)

	# def forward(self, question, chunks, chunks_mask, answer, answer_mask, gold_index):
	def forward(self, question, chunks, chunks_mask, gold_index):
		chunks_encoded = self.chunk_encoder(chunks, chunks_mask)
		question_encoded = self.question_encoder(question)
		question_expanded = question_encoded.expand(chunks_encoded.size())
		combined_representation = torch.cat((chunks_encoded, question_expanded), 2).squeeze(dim=1)
		scores = self.modeling_layer(combined_representation).squeeze().unsqueeze(0)
		return scores

	def eval(self, question, chunks, chunks_mask):
		chunks_encoded = self.chunk_encoder(chunks, chunks_mask)
		question_encoded = self.question_encoder(question)
		question_expanded = question_encoded.expand(chunks_encoded.size())
		combined_representation = torch.cat((chunks_encoded, question_expanded), 2).squeeze(dim=1)
		scores = self.modeling_layer(combined_representation).squeeze().unsqueeze(0)
		return scores


class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(4 * hidden_size, 4 * hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(4 * hidden_size, 4 * hidden_size)
        self.activation = nn.ReLU()
        self.linear3 = nn.Linear(4 * hidden_size, 1)

    def forward(self, input):
        input = self.linear1(input)
        input = self.activation(input)
        input = self.linear2(input)
        input = self.activation(input)
        input = self.linear3(input)
        return input

class EncoderBlock(nn.Module):
	def __init__(self, embed_size, hidden_size, kernel_size):
		super(EncoderBlock, self).__init__()
		self.convolution_layer1 = nn.Conv1d(embed_size, hidden_size, kernel_size, padding=(kernel_size - 1) / 2)
		self.convolution_layer2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=(kernel_size - 1) / 2)
		self.convolution_layer3 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=(kernel_size - 1) / 2)
		self.convolution_layer4 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=(kernel_size - 1) / 2)

		self.activation = nn.ReLU()

	def forward(self, input, mask=None):
		input = input.transpose(1, 2)
		input = self.convolution_layer1(input)
		input = self.activation(input)
		input = self.convolution_layer2(input)
		input = self.activation(input)
		input = self.convolution_layer3(input)
		input = self.activation(input)

		if mask is not None:
			input = input * mask.unsqueeze(1)
		input1 = F.max_pool1d(input, kernel_size=input.size()[2])
		input2 = F.avg_pool1d(input, kernel_size=input.size()[2])
		input = torch.cat((input1, input2), 1)
		input = input.transpose(1, 2)
		return input


class LookupEncoder(nn.Module):
	def __init__(self, vocab_size, embedding_dim, pretrain_embedding=None):
		super(LookupEncoder, self).__init__()
		self.embedding_dim = embedding_dim
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

		if pretrain_embedding is not None:
			self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrain_embedding))

		self.word_embeddings.weight.requires_grad = False

	def forward(self, batch):
		return self.word_embeddings(batch)