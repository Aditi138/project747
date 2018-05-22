import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from bidaf import BiDAF, LinearSeqAttn,weighted_avg,BiLinearAttn, log_sum_exp
import codecs
import numpy as np

class TriAttn(nn.Module):
	def __init__(self, args, loader):
		super(TriAttn, self).__init__()
		hidden_size = args.hidden_size
		embed_size = args.embed_size
		word_vocab_size = loader.vocab.get_length()

		self.dropout_emb = args.dropout_emb

		## word embedding layer
		self.word_embedding_layer = LookupEncoder(word_vocab_size, embedding_dim=embed_size,pretrain_embedding=loader.pretrain_embedding)

		## contextual embedding layer
		# self.contextual_embedding_layer = RecurrentContext(input_size=embed_size, hidden_size=embed_size // 2, num_layers=args.num_layers)

		## bidirectional attention flow between question and context

		self.attention_flow_a2q = BiDAF(embed_size)


		## modelling layer for question and context : this layer also converts the 8 dimensional input intp two dimensioanl output
		self.modeling_layer_q = RecurrentContext(embed_size, hidden_size)

		self.modeling_layer_a = RecurrentContext(embed_size, hidden_size)

		self.self_attn_q = LinearSeqAttn(2 * hidden_size)

		self.self_attn_a = LinearSeqAttn(2 * hidden_size)


		## output layer
		output_layer_inputdim = 2 * hidden_size
		self.output_layer = OutputLayer(output_layer_inputdim, hidden_size)
		self.answer_context_bilinear = nn.Linear(2 * hidden_size, 2 * hidden_size)
		self.query_answer_bilinear = nn.Linear(2 * hidden_size, 2 * hidden_size)
		self.loss = torch.nn.CrossEntropyLoss()

		self.dropout = torch.nn.Dropout(args.dropout)



	def forward(self, query_embedded, batch_query_length,batch_query_mask,
				context_embedded, batch_context_length,batch_context_mask,batch_context_scores,
				batch_candidates_embedded, batch_candidate_lengths_sorted, batch_candidate_masks_sorted,batch_candidate_unsort,
				gold_index, gold_chunk):
		query_embedded = self.word_embedding_layer(query_embedded)
		batch_candidates_embedded = self.word_embedding_layer(batch_candidates_embedded)

		# dropout emb
		query_embedded = nn.functional.dropout(query_embedded, p=self.dropout_emb, training=True).unsqueeze(0)
		batch_candidates_embedded = nn.functional.dropout(batch_candidates_embedded, p=self.dropout_emb, training=True)

		batch_query_mask = batch_query_mask.unsqueeze(0)

		batch_size = batch_candidates_embedded.size(0)

		query_input_modeled, _ = self.modeling_layer_q(query_embedded, batch_query_length)
		query_input_modeled = self.dropout(query_input_modeled)

		# answer_modeled = batch_candidates_encoded
		answer_input_modelling = batch_candidates_embedded
		answer_modeled, _ = self.modeling_layer_a(answer_input_modelling,batch_candidate_lengths_sorted)  # (N, |A|, 2d)

		query_self_attention = self.self_attn_q(query_input_modeled, batch_query_mask)
		q_hidden = weighted_avg(query_input_modeled, query_self_attention)

		answer_self_attention = self.self_attn_a(answer_modeled, batch_candidate_masks_sorted)
		a_hidden = weighted_avg(answer_modeled, answer_self_attention)

		logits_qa = self.query_answer_bilinear(q_hidden) * a_hidden  # (N, 2d)
		answer_scores = self.output_layer(logits_qa)  # (N,K,4d) ==>#(N,K,1)

		## unsort the answer scores
		answer_scores = torch.index_select(answer_scores, 0, batch_candidate_unsort)
		loss = self.loss(answer_scores.transpose(0, 1), gold_index)
		sorted, indices = torch.sort(answer_scores, dim=0, descending=False)
		return loss, indices

	def eval(self,query_embedded, batch_query_length,batch_query_mask,
				context_embedded, batch_context_length,batch_context_mask,batch_context_scores,
				batch_candidates_embedded, batch_candidate_lengths_sorted, batch_candidate_masks_sorted,batch_candidate_unsort
				):
		query_embedded = self.word_embedding_layer(query_embedded)
		batch_candidates_embedded = self.word_embedding_layer(batch_candidates_embedded)

		# dropout emb
		query_embedded = nn.functional.dropout(query_embedded, p=self.dropout_emb, training=False).unsqueeze(0)
		batch_candidates_embedded = nn.functional.dropout(batch_candidates_embedded, p=self.dropout_emb, training=False)

		batch_query_mask = batch_query_mask.unsqueeze(0)

		batch_size = batch_candidates_embedded.size(0)

		query_input_modeled, _ = self.modeling_layer_q(query_embedded, batch_query_length)
		query_input_modeled = self.dropout(query_input_modeled)

		# answer_modeled = batch_candidates_encoded
		answer_input_modelling = batch_candidates_embedded
		answer_modeled, _ = self.modeling_layer_a(answer_input_modelling,
												  batch_candidate_lengths_sorted)  # (N, |A|, 2d)

		query_self_attention = self.self_attn_q(query_input_modeled, batch_query_mask)
		q_hidden = weighted_avg(query_input_modeled, query_self_attention)

		answer_self_attention = self.self_attn_a(answer_modeled, batch_candidate_masks_sorted)
		a_hidden = weighted_avg(answer_modeled, answer_self_attention)

		logits_qa = self.query_answer_bilinear(q_hidden) * a_hidden  # (N, 2d)
		answer_scores = self.output_layer(logits_qa)  # (N,K,4d) ==>#(N,K,1)

		## unsort the answer scores
		answer_scores = torch.index_select(answer_scores, 0, batch_candidate_unsort)
		sorted, indices = torch.sort(answer_scores, dim=0, descending=True)
		return indices,indices


class OutputLayer(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(OutputLayer, self).__init__()
		self.mlp = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(hidden_size, 1),
			#nn.Softmax(), ## since loss is being replaced by cross entropy the exoected input into loss function
		)

	def forward(self, batch):
		return self.mlp(batch)

class RecurrentContext(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers=1):
		# format of input output
		super(RecurrentContext, self).__init__()
		self.lstm_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
								  bidirectional=True, batch_first=True)

	def forward(self, batch, batch_length):
		packed = torch.nn.utils.rnn.pack_padded_sequence(batch, batch_length, batch_first=True)
		self.lstm_layer.flatten_parameters()
		outputs, hidden = self.lstm_layer(packed)  # output: concatenated hidden dimension
		outputs_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
		return outputs_unpacked, hidden


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
