import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from bidaf import BiDAF

class ContextMRR(nn.Module):
	def __init__(self, vocab_size, input_size, hidden_size, embeddings = None):
		self.word_embedding_layer = LookupEncoder(vocab_size, input_size)
		self.contextual_embedding_layer = RecurrentContext(input_size, hidden_size, num_layers=1)
		self.attention_flow_layer1 = BiDAF(hidden_size)
		self.attention_flow_layer2 = BiDAF(hidden_size)


class RecurrentContext(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers):
		# format of input output
		self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
								  bidirectional=True, batch_first=True)

	def forward(self, batch):
		## batch :
		output, hidden = self.lstm_layer(batch)


class LookupEncoder(nn.Module):
	def __init__(self, vocab_size, embedding_dim, pretrain_embedding=None):
		super(LookupEncoder, self).__init__()
		self.embedding_dim = embedding_dim
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrain_embedding))

	def lookup(self, batch):
		return self.word_embeddings(batch)