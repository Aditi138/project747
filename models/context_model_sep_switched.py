import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from bidaf import BiDAF

class ContextMRR_Sep_Switched(nn.Module):
	def __init__(self, args, loader):
		super(ContextMRR_Sep_Switched, self).__init__()
		hidden_size = args.hidden_size
		embed_size = args.embed_size

		## dropout layer
		if args.dropout > 0:
			self._dropout = torch.nn.Dropout(p=args.dropout)
		else:
			self._dropout = lambda x: x

		## contextual embedding layer
		self.contextual_embedding_layer = RecurrentContext(input_size=embed_size, hidden_size=hidden_size, num_layers=1)

		## bidirectional attention flow between question and context
		self.attention_flow_layer1 = BiDAF(2*hidden_size)

		modeling_layer_inputdim = 2 * hidden_size
		self.modeling_layer1 = RecurrentContext(modeling_layer_inputdim, hidden_size, num_layers=1)

		self.linearrelu = ffnLayer(4*hidden_size, 4*hidden_size)

		output_layer_inputdim = 6*hidden_size
		self.output_layer = OutputLayer(output_layer_inputdim, hidden_size)

		self.loss = torch.nn.CrossEntropyLoss()


	def forward(self, batch_query, batch_query_length,batch_query_mask,
				batch_context, batch_context_length,batch_context_mask,
				batch_candidates_sorted, batch_candidate_lengths_sorted, batch_candidate_masks_sorted,batch_candidate_unsort,
				gold_index, negative_indices):


		query_embedded = batch_query.unsqueeze(0)
		context_embedded = batch_context.unsqueeze(0)
		## Encode query and context
		# (N, J, 2d)
		query_encoded,query_encoded_hidden = self.contextual_embedding_layer(query_embedded, batch_query_length)
		query_encoded = self._dropout(query_encoded)
		# (N, T, 2d)
		context_encoded,_ = self.contextual_embedding_layer(context_embedded, batch_context_length)
		context_encoded = self._dropout(context_encoded)

		## required to support single element batch of question
		batch_query_mask = batch_query_mask.unsqueeze(0)
		batch_context_mask = batch_context_mask.unsqueeze(0)


		context_attention_encoded, context_aware_query_encoded, query_aware_context_encoded = self.attention_flow_layer1(
			query_encoded, context_encoded, batch_query_mask, batch_context_mask)

		# (N,T,2d) => (N,1,4d)
		#context_modelled = self.modeling_layer1
		context_avg_pool = torch.cat([torch.max(context_aware_query_encoded, dim=1)[0], torch.mean(context_aware_query_encoded, dim=1)], dim=1)

		#(N, 1, 4d) => (N,1,4d)
		query_aware_context_modeled = self.linearrelu(context_avg_pool)
		query_aware_context_modeled = self._dropout(query_aware_context_modeled)


		batch_size = batch_candidates_sorted.size(0)
		batch_context_modeled = query_aware_context_modeled.expand(batch_size,query_aware_context_modeled.size(1))

		batch_candidates_embedded = batch_candidates_sorted
		# (N1, K, 2d)
		batch_candidates_encoded,batch_candidates_hidden = self.contextual_embedding_layer(batch_candidates_embedded, batch_candidate_lengths_sorted)
		batch_candidates_hidden = torch.cat([batch_candidates_hidden[-2], batch_candidates_hidden[-1]], dim=1)
		batch_candidates_hidden = self._dropout(batch_candidates_hidden)

		context_answer_hidden_state = torch.cat([batch_candidates_hidden,batch_context_modeled], dim=1)
		answer_scores = self.output_layer(context_answer_hidden_state)
		answer_modeled = self._dropout(answer_scores)
		answer_modeled = F.softmax(answer_modeled, dim=0)

		## unsort the answer scores
		answer_modeled = torch.index_select(answer_modeled, 0, batch_candidate_unsort)
		loss = self.loss(answer_modeled.transpose(0,1), gold_index)
		sorted, indices = torch.sort(answer_modeled, dim=0, descending=True)
		return loss, indices


	def eval(self,batch_query, batch_query_length,batch_query_mask,
				batch_context, batch_context_length,batch_context_mask,
			 batch_candidates_sorted, batch_candidate_lengths_sorted,batch_candidate_masks_sorted, batch_candidate_unsort):

		query_embedded = batch_query.unsqueeze(0)
		context_embedded = batch_context.unsqueeze(0)
		## Encode query and context
		# (N, J, 2d)
		query_encoded, query_encoded_hidden = self.contextual_embedding_layer(query_embedded, batch_query_length)
		query_encoded = self._dropout(query_encoded)
		# (N, T, 2d)
		context_encoded, _ = self.contextual_embedding_layer(context_embedded, batch_context_length)
		context_encoded = self._dropout(context_encoded)

		## required to support single element batch of question
		batch_query_mask = batch_query_mask.unsqueeze(0)
		batch_context_mask = batch_context_mask.unsqueeze(0)

		context_attention_encoded, context_aware_query_encoded, query_aware_context_encoded = self.attention_flow_layer1(
			query_encoded, context_encoded, batch_query_mask, batch_context_mask)

		# (N,T,2d) => (N,1,4d)

		context_avg_pool = torch.cat(
			[torch.max(context_aware_query_encoded, dim=1)[0], torch.mean(context_aware_query_encoded, dim=1)], dim=1)

		# (N, 1, 4d) => (N,1,4d)
		query_aware_context_modeled = self.linearrelu(context_avg_pool)
		query_aware_context_modeled = self._dropout(query_aware_context_modeled)

		batch_size = batch_candidates_sorted.size(0)
		batch_context_modeled = query_aware_context_modeled.expand(batch_size, query_aware_context_modeled.size(1))

		batch_candidates_embedded = batch_candidates_sorted
		# (N1, K, 2d)
		batch_candidates_encoded, batch_candidates_hidden = self.contextual_embedding_layer(batch_candidates_embedded,
																							batch_candidate_lengths_sorted)
		batch_candidates_hidden = torch.cat([batch_candidates_hidden[-2], batch_candidates_hidden[-1]], dim=1)
		batch_candidates_hidden = self._dropout(batch_candidates_hidden)

		context_answer_hidden_state = torch.cat([batch_candidates_hidden, batch_context_modeled], dim=1)
		answer_scores = self.output_layer(context_answer_hidden_state)
		answer_modeled = self._dropout(answer_scores)
		answer_modeled = F.log_softmax(answer_modeled, dim=0)

		## unsort the answer scores
		answer_modeled = torch.index_select(answer_modeled, 0, batch_candidate_unsort)
		sorted, indices = torch.sort(F.log_softmax(answer_modeled.squeeze(0), dim=0), dim=0, descending=True)
		return indices





class OutputLayer(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(OutputLayer, self).__init__()
		self.mlp = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 1),
			# nn.Sigmoid(), ## since loss is being replaced by cross entropy the exoected input into loss function
		)

	def forward(self, batch):
		return self.mlp(batch)


class ffnLayer(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(ffnLayer, self).__init__()
		self.mlp = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU()
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
    def forward(self, batch):
        return self.word_embeddings(batch)
