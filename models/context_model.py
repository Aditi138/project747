import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from bidaf import BiDAF

class ContextMRR(nn.Module):
	def __init__(self, args, loader):
		super(ContextMRR, self).__init__()
		hidden_size = args.hidden_size
		embed_size = args.embed_size
		word_vocab_size = loader.vocab.get_length()

		## word embedding layer
		#self.word_embedding_layer = LookupEncoder(word_vocab_size, embedding_dim=embed_size) #, pretrain_embedding=loader.pretrain_embedding)

		## dropout layer
		if args.dropout > 0:
			self._dropout = torch.nn.Dropout(p=args.dropout)
		else:
			self._dropout = lambda x: x

		## contextual embedding layer
		self.contextual_embedding_layer = RecurrentContext(input_size=embed_size, hidden_size=hidden_size, num_layers=1)

		## bidirectional attention flow between question and context
		self.attention_flow_layer1 = BiDAF(2*hidden_size)

		## modelling layer for question and context : this layer also converts the 8 dimensional input intp two dimensioanl output
		modeling_layer_inputdim = 8 * hidden_size
		self.modeling_layer1 = RecurrentContext(modeling_layer_inputdim, hidden_size)

		'''BIDAF 2'''
		self.contextual_embedding_layer_2 = RecurrentContext(input_size=embed_size, hidden_size=hidden_size, num_layers=1)

		## bidirectional attention flow between [q+c] and answer
		self.attention_flow_layer2 = BiDAF(2*hidden_size)

		## modeling layer
		modeling_layer_inputdim = 6*hidden_size
		self.modeling_layer2 = RecurrentContext(modeling_layer_inputdim, hidden_size)

		## output layer
		## current implementation: run an mlp on the concatenated hidden states of the answer modeling layer
		output_layer_inputdim = 4*hidden_size
		self.output_layer = OutputLayer(output_layer_inputdim, hidden_size)

		self.loss = torch.nn.CrossEntropyLoss()


	def forward(self, batch_query, batch_query_length,batch_query_mask,
				batch_context, batch_context_length,batch_context_mask,
				batch_candidates_sorted, batch_candidate_lengths_sorted, batch_candidate_masks_sorted,batch_candidate_unsort,
				gold_index, negative_indices):

		## Embed query and context
		# (N, J, d)
		#query_embedded = self.word_embedding_layer(batch_query.unsqueeze(0))
		# (N, T, d)
		#context_embedded = self.word_embedding_layer(batch_context.unsqueeze(0))

		query_embedded = batch_query.unsqueeze(0)
		context_embedded = batch_context.unsqueeze(0)
		## Encode query and context
		# (N, J, 2d)
		query_encoded,_ = self.contextual_embedding_layer(query_embedded, batch_query_length)
		query_encoded = self._dropout(query_encoded)
		# (N, T, 2d)
		context_encoded,_ = self.contextual_embedding_layer(context_embedded, batch_context_length)
		context_encoded = self._dropout(context_encoded)

		## required to support single element batch of question
		batch_query_mask = batch_query_mask.unsqueeze(0)
		batch_context_mask = batch_context_mask.unsqueeze(0)

		## BiDAF 1 to get ~U, ~h and G (8d) between context and query
		# (N, T, 8d) , (N, T ,2d) , (N, 1, 2d)
		context_attention_encoded, query_aware_context_encoded, context_aware_query_encoded = self.attention_flow_layer1(query_encoded, context_encoded,batch_query_mask,batch_context_mask)

		## modelling layer 1
		# (N, T, 8d) => (N, T, 2d)
		context_modeled,_ = self.modeling_layer1(context_attention_encoded, batch_context_length)
		context_modeled = self._dropout(context_modeled)

		'''
		BIDAF 2
		'''
		## BiDAF for answers
		batch_size = batch_candidates_sorted.size(0)
		# N=1 so (N, T, 2d) => (N1, T, 2d)
		batch_context_modeled = context_modeled.repeat(batch_size,1,1)
		# (N1, K, d)
		#batch_candidates_embedded = self.word_embedding_layer(batch_candidates_sorted)
		batch_candidates_embedded = batch_candidates_sorted
		# (N1, K, 2d)
		batch_candidates_encoded,_ = self.contextual_embedding_layer(batch_candidates_embedded, batch_candidate_lengths_sorted)
		batch_candidates_encoded = self._dropout(batch_candidates_encoded)

		answer_attention_encoded, context_aware_answer_encoded, answer_aware_context_encoded = self.attention_flow_layer2(batch_context_modeled, batch_candidates_encoded, batch_context_mask,batch_candidate_masks_sorted)

		## concatenate original answer and context aware answer
		input_to_answer_model = torch.cat([batch_candidates_encoded,context_aware_answer_encoded,batch_candidates_encoded * context_aware_answer_encoded],dim=-1)

		## modelling layer 2
		# (N1, K, 8d) => (N1, K, 2d)
		answer_modeled, (answer_hidden_state, answer_cell_state) = self.modeling_layer2(input_to_answer_model, batch_candidate_lengths_sorted)
		answer_modeled = self._dropout(answer_modeled)

		answer_modeled_replaced = self.attention_flow_layer2.replace_masked_values(answer_modeled.transpose(1, 2),
																				   batch_candidate_masks_sorted.unsqueeze(
																					   1), 1e-7)
		answer_modeled_mask = answer_modeled.transpose(1, 2) * batch_candidate_masks_sorted.unsqueeze(1)
		answer_concat_hidden = torch.cat(
			(torch.max(answer_modeled_replaced, dim=2)[0], torch.mean(answer_modeled_mask, dim=2)), dim=1)

		## output layer : concatenate hidden dimension of the final answer model layer and run through an MLP : (N1, 2d) => (N1, d)
		# (N1, 2d) => (N1, 1)
		# answer_concat_hidden = torch.cat([answer_hidden_state[-2], answer_hidden_state[-1]], dim=1)

		# (N1, 4d) => (N1, 1)
		# take maxmimum and average of hidden embeddings and concatenate them


		answer_scores = self.output_layer(answer_concat_hidden)


		## unsort the answer scores
		answer_scores_unsorted = torch.index_select(answer_scores, 0, batch_candidate_unsort)

		## Hinge Loss
		# gold_features = torch.index_select(answer_scores_unsorted, 0, index=gold_index)
		# negative_features = torch.index_select(answer_scores_unsorted, 0, index=negative_indices)
		# #negative_metrics = torch.index_select(batch_metrics, 0, index=negative_indices)
		# #negative_features = negative_features + negative_metrics.unsqueeze(1)
		# max_negative_feature, max_negative_index = torch.max(negative_features, 0)
		# loss = torch.clamp(1 - gold_features + max_negative_feature, 0)

		loss = self.loss(answer_scores_unsorted.transpose(0,1), gold_index)
		sorted, indices = torch.sort(F.log_softmax(answer_scores_unsorted.squeeze(0),dim=0), dim=0, descending=True)
		return loss, indices


	def eval(self,batch_query, batch_query_length,batch_query_mask,
				batch_context, batch_context_length,batch_context_mask,
			 batch_candidates_sorted, batch_candidate_lengths_sorted,batch_candidate_masks_sorted, batch_candidate_unsort):
		## Embed query and context
		# (N, J, d)
		#query_embedded = self.word_embedding_layer(batch_query.unsqueeze(0))
		# (N, T, d)
		#context_embedded = self.word_embedding_layer(batch_context.unsqueeze(0))

		query_embedded = batch_query.unsqueeze(0)
		context_embedded = batch_context.unsqueeze(0)

		## Encode query and context
		# (N, J, 2d)
		query_encoded, _ = self.contextual_embedding_layer(query_embedded, batch_query_length)
		# (N, T, 2d)
		context_encoded, _ = self.contextual_embedding_layer(context_embedded, batch_context_length)

		## BiDAF 1 to get ~U, ~h and G (8d) between context and query
		# (N, T, 8d) , (N, T ,2d) , (N, 1, 2d)
		batch_query_mask = batch_query_mask.unsqueeze(0)
		batch_context_mask = batch_context_mask.unsqueeze(0)

		context_attention_encoded, query_aware_context_encoded, context_aware_query_encoded = self.attention_flow_layer1(
			query_encoded, context_encoded,batch_query_mask,batch_context_mask)

		## modelling layer 1
		# (N, T, 8d) => (N, T, 2d)
		context_modeled, _ = self.modeling_layer1(context_attention_encoded, batch_context_length)

		## BiDAF for answers
		batch_size = batch_candidates_sorted.size(0)
		# N=1 so (N, T, 2d) => (N1, T, 2d)
		batch_context_modeled = context_modeled.repeat(batch_size, 1, 1)
		# (N1, K, d)
		#batch_candidates_embedded = self.word_embedding_layer(batch_candidates_sorted)
		batch_candidates_embedded = batch_candidates_sorted
		# (N1, K, 2d)
		batch_candidates_encoded, _ = self.contextual_embedding_layer(batch_candidates_embedded,
																	  batch_candidate_lengths_sorted)
		answer_attention_encoded, context_aware_answer_encoded, answer_aware_context_encoded = self.attention_flow_layer2(
			batch_context_modeled, batch_candidates_encoded,batch_context_mask,batch_candidate_masks_sorted)

		input_to_answer_model = torch.cat([batch_candidates_encoded, context_aware_answer_encoded,
										   batch_candidates_encoded * context_aware_answer_encoded], dim=-1)

		## modelling layer 2
		# (N1, K, 8d) => (N1, K, 2d)
		answer_modeled, (answer_hidden_state, answer_cell_state) = self.modeling_layer2(input_to_answer_model,
																						batch_candidate_lengths_sorted,)

		answer_modeled_replaced = self.attention_flow_layer2.replace_masked_values(answer_modeled.transpose(1, 2),
																				   batch_candidate_masks_sorted.unsqueeze(
																					   1), 1e-7)
		answer_modeled_mask = answer_modeled.transpose(1, 2) * batch_candidate_masks_sorted.unsqueeze(1)
		answer_concat_hidden = torch.cat(
			(torch.max(answer_modeled_replaced, dim=2)[0], torch.mean(answer_modeled_mask, dim=2)), dim=1)

		## output layer : concatenate hidden dimension of the final answer model layer and run through an MLP : (N1, 2d) => (N1, d)
		# (N1, 2d) => (N1, 1)
		# answer_concat_hidden = torch.cat([answer_hidden_state[-2], answer_hidden_state[-1]], dim=1)

		# (N1, K, 4d) => (N1, 1, 4d)
		# take maxmimum and average of hidden embeddings and concatenate them

		answer_scores = self.output_layer(answer_concat_hidden)

		## unsort the answer scores
		answer_scores_unsorted = torch.index_select(answer_scores, 0, batch_candidate_unsort)
		sorted, indices = torch.sort(	F.log_softmax(answer_scores_unsorted, dim=0), dim=0, descending=True)
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

class RecurrentContext(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers=1):
		# format of input output
		super(RecurrentContext, self).__init__()
		self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
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