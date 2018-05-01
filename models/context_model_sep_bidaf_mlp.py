import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from bidaf import BiDAF
import numpy as np

class ContextMRR_Sep_Bidaf_MLP(nn.Module):
	def __init__(self, args, loader):
		super(ContextMRR_Sep_Bidaf_MLP, self).__init__()
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

		c2q_linearLayer_dim = 2 * hidden_size
		self.c2q_linearLayer = TimeDistributed(nn.Sequential(
			torch.nn.Linear(c2q_linearLayer_dim, c2q_linearLayer_dim),
			torch.nn.ReLU()))

		modeling_layer_inputdim = 2 * hidden_size
		self.modeling_layer1 = RecurrentContext(modeling_layer_inputdim, hidden_size, num_layers=1)


		self.contextual_embedding_layer_2 = RecurrentContext(input_size=embed_size, hidden_size=hidden_size, num_layers=1)
		self.attention_flow_layer2 = BiDAF(2 * hidden_size)

		self.c2a_linearLayer = TimeDistributed(nn.Sequential(
			torch.nn.Linear(c2q_linearLayer_dim, c2q_linearLayer_dim),
			torch.nn.ReLU()))



		self.modeling_layer2 = RecurrentContext(modeling_layer_inputdim, hidden_size, num_layers=1)

		output_layer_inputdim = 12 *hidden_size
		self.output_layer = OutputLayer(output_layer_inputdim, 2*hidden_size)

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
		query_encoded_hidden = torch.cat([query_encoded_hidden[-2], query_encoded_hidden[-1]], dim=1)
		query_encoded = self._dropout(query_encoded)
		# (N, T, 2d)
		context_encoded,_ = self.contextual_embedding_layer(context_embedded, batch_context_length)
		context_encoded = self._dropout(context_encoded)

		## required to support single element batch of question
		batch_query_mask = batch_query_mask.unsqueeze(0)
		batch_context_mask = batch_context_mask.unsqueeze(0)


		context_attention_encoded, query_aware_context_encoded, context_aware_query_encoded = self.attention_flow_layer1(
			query_encoded, context_encoded, batch_query_mask.unsqueeze(1), batch_context_mask)

		# (N,T,2d) => (N,1,2d)
		query_aware_context_encoded = self.c2q_linearLayer(query_aware_context_encoded)
		query_aware_context_encoded = self._dropout(query_aware_context_encoded)

		# (N,T,2d) => (N,1,2d)
		context_modeled, context_modeled_hidden = self.modeling_layer1(query_aware_context_encoded, batch_context_length)
		context_modeled = self._dropout(context_modeled)
		# (N,T,2d) => (N,1,4d)
		context_avg_pool = torch.cat([torch.max(context_modeled, dim=1)[0], torch.mean(context_modeled, dim=1)], dim=1)
		#(N,1,8d)
		context_rich_rep_using_question = torch.cat([context_avg_pool,query_encoded_hidden ], dim  =1)

		'''BIDAF 2'''
		batch_size = batch_candidates_sorted.size(0)
		# (N1, K, 2d)
		context_encoded_2, _ = self.contextual_embedding_layer_2(context_embedded, batch_context_length)
		context_encoded_2 = self._dropout(context_encoded_2)
		batch_context_modeled_2 = context_encoded_2.expand(batch_size, context_encoded_2.size(1),
														   context_encoded_2.size(2))
		batch_context_mask_expanded = batch_context_mask.expand(batch_size, batch_context_mask.size(1))
		# (N1, K, d)

		batch_candidates_embedded = batch_candidates_sorted
		# (N1, K, 2d)
		batch_candidates_encoded, batch_candidates_hidden = self.contextual_embedding_layer_2(batch_candidates_embedded,
																							  batch_candidate_lengths_sorted)
		batch_candidates_hidden = torch.cat([batch_candidates_hidden[-2], batch_candidates_hidden[-1]], dim=1)
		batch_candidates_hidden = self._dropout(batch_candidates_hidden)

		context_over_answer_attention_encoded, answer_aware_context_encoded, context_aware_answer_encoded = self.attention_flow_layer2(
			batch_candidates_encoded, batch_context_modeled_2, batch_candidate_masks_sorted.unsqueeze(1),batch_context_mask_expanded)

		answer_aware_context_encoded = self.c2a_linearLayer(answer_aware_context_encoded)
		answer_aware_context_encoded = self._dropout(answer_aware_context_encoded)

		# (N,T,2d) => (N,1,2d)
		answer_aware_context_encoded, _ = self.modeling_layer2(answer_aware_context_encoded,
															   np.array([batch_context_length[0]] * batch_size))
		answer_aware_context_encoded = self._dropout(answer_aware_context_encoded)
		# (N,T,2d) => (N,1,4d)
		answer_aware_context_encoded = torch.cat([torch.max(answer_aware_context_encoded, dim=1)[0], torch.mean(answer_aware_context_encoded, dim=1)], dim=1)
		# (N,1,8d)
		context_rich_rep_using_answer = torch.cat([answer_aware_context_encoded, batch_candidates_hidden], dim=1)

		scores = torch.cat([context_rich_rep_using_question.expand(batch_size,context_rich_rep_using_question.size(1)),context_rich_rep_using_answer], dim=1)
		answer_scores = self.output_layer(scores)


		## unsort the answer scores
		answer_modeled = torch.index_select(answer_scores, 0, batch_candidate_unsort)
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
		query_encoded_hidden = torch.cat([query_encoded_hidden[-2], query_encoded_hidden[-1]], dim=1)
		query_encoded = self._dropout(query_encoded)
		# (N, T, 2d)
		context_encoded, _ = self.contextual_embedding_layer(context_embedded, batch_context_length)
		context_encoded = self._dropout(context_encoded)

		## required to support single element batch of question
		batch_query_mask = batch_query_mask.unsqueeze(0)
		batch_context_mask = batch_context_mask.unsqueeze(0)

		context_attention_encoded, query_aware_context_encoded, context_aware_query_encoded = self.attention_flow_layer1(
			query_encoded, context_encoded, batch_query_mask.unsqueeze(1), batch_context_mask)

		# (N,T,2d) => (N,1,2d)
		query_aware_context_encoded = self.c2q_linearLayer(query_aware_context_encoded)
		query_aware_context_encoded = self._dropout(query_aware_context_encoded)

		# (N,T,2d) => (N,1,2d)
		context_modeled, context_modeled_hidden = self.modeling_layer1(query_aware_context_encoded,
																	   batch_context_length)
		context_modeled = self._dropout(context_modeled)
		# (N,T,2d) => (N,1,4d)
		context_avg_pool = torch.cat([torch.max(context_modeled, dim=1)[0], torch.mean(context_modeled, dim=1)], dim=1)
		# (N,1,8d)
		context_rich_rep_using_question = torch.cat(
			[context_avg_pool, query_encoded_hidden], dim=1)

		'''BIDAF 2'''
		batch_size = batch_candidates_sorted.size(0)
		# (N1, K, 2d)
		context_encoded_2, _ = self.contextual_embedding_layer_2(context_embedded, batch_context_length)
		context_encoded_2 = self._dropout(context_encoded_2)
		batch_context_modeled_2 = context_encoded_2.expand(batch_size, context_encoded_2.size(1),
														   context_encoded_2.size(2))
		batch_context_mask_expanded = batch_context_mask.expand(batch_size, batch_context_mask.size(1))
		# (N1, K, d)

		batch_candidates_embedded = batch_candidates_sorted
		# (N1, K, 2d)
		batch_candidates_encoded, batch_candidates_hidden = self.contextual_embedding_layer_2(batch_candidates_embedded,
																							  batch_candidate_lengths_sorted)
		batch_candidates_hidden = torch.cat([batch_candidates_hidden[-2], batch_candidates_hidden[-1]], dim=1)
		batch_candidates_hidden = self._dropout(batch_candidates_hidden)

		context_over_answer_attention_encoded, answer_aware_context_encoded, context_aware_answer_encoded = self.attention_flow_layer2(
			batch_candidates_encoded, batch_context_modeled_2, batch_candidate_masks_sorted.unsqueeze(1),
			batch_context_mask_expanded)

		answer_aware_context_encoded = self.c2a_linearLayer(answer_aware_context_encoded)
		answer_aware_context_encoded = self._dropout(answer_aware_context_encoded)

		# (N,T,2d) => (N,1,2d)
		answer_aware_context_encoded, _ = self.modeling_layer2(answer_aware_context_encoded,
															   np.array([batch_context_length[0]] * batch_size))
		answer_aware_context_encoded = self._dropout(answer_aware_context_encoded)
		# (N,T,2d) => (N,1,4d)
		answer_aware_context_encoded = torch.cat(
			[torch.max(answer_aware_context_encoded, dim=1)[0], torch.mean(answer_aware_context_encoded, dim=1)], dim=1)
		# (N,1,8d)
		context_rich_rep_using_answer = torch.cat(
			[answer_aware_context_encoded, batch_candidates_hidden], dim=1)

		scores = torch.cat([context_rich_rep_using_question.expand(batch_size, context_rich_rep_using_question.size(1)),
							context_rich_rep_using_answer], dim=1)
		answer_scores = self.output_layer(scores)

		## unsort the answer scores
		answer_modeled = torch.index_select(answer_scores, 0, batch_candidate_unsort)
		sorted, indices = torch.sort(answer_modeled, dim=0, descending=True)
		return indices





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
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size)
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

class TimeDistributed(torch.nn.Module):
	"""
    Given an input shaped like ``(batch_size, time_steps, [rest])`` and a ``Module`` that takes
    inputs like ``(batch_size, [rest])``, ``TimeDistributed`` reshapes the input to be
    ``(batch_size * time_steps, [rest])``, applies the contained ``Module``, then reshapes it back.
    Note that while the above gives shapes with ``batch_size`` first, this ``Module`` also works if
    ``batch_size`` is second - we always just combine the first two dimensions, then split them.
    """
	def __init__(self, module):
		super(TimeDistributed, self).__init__()
		self._module = module

	def forward(self, *inputs):  # pylint: disable=arguments-differ
		reshaped_inputs = []
		for input_tensor in inputs:
			input_size = input_tensor.size()
			if len(input_size) <= 2:
				raise RuntimeError("No dimension to distribute: " + str(input_size))

			# Squash batch_size and time_steps into a single axis; result has shape
			#  (batch_size * time_steps, input_size).
			squashed_shape = [-1] + [x for x in input_size[2:]]
			reshaped_inputs.append(input_tensor.contiguous().view(*squashed_shape))

		reshaped_outputs = self._module(*reshaped_inputs)

		# Now get the output back into the right shape.
        # (batch_size, time_steps, [hidden_size])
		new_shape = [input_size[0], input_size[1]] + [x for x in reshaped_outputs.size()[1:]]
		outputs = reshaped_outputs.contiguous().view(*new_shape)

		return outputs

