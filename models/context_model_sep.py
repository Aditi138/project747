import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from bidaf import BiDAF

class ContextMRR_Sep(nn.Module):
	def __init__(self, args, loader):
		super(ContextMRR_Sep, self).__init__()
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
		self.modeling_layer1 = RecurrentContext(modeling_layer_inputdim, hidden_size, num_layers=1)

		'''BIDAF 2'''
		self.contextual_embedding_layer_2 = RecurrentContext(input_size=embed_size, hidden_size=hidden_size, num_layers=1)

		## bidirectional attention flow between [q+c] and answer
		self.attention_flow_layer2 = BiDAF(2*hidden_size)

		## modeling layer
		modeling_layer_inputdim = 8*hidden_size
		self.modeling_layer2 = RecurrentContext(modeling_layer_inputdim, hidden_size, num_layers=1)

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
		query_encoded,query_encoded_hidden = self.contextual_embedding_layer(query_embedded, batch_query_length)
		query_encoded_hidden = torch.cat([query_encoded_hidden[-2], query_encoded_hidden[-1]], dim=1)
		query_encoded = self._dropout(query_encoded)
		# (N, T, 2d)
		context_encoded,_ = self.contextual_embedding_layer(context_embedded, batch_context_length)
		context_encoded = self._dropout(context_encoded)

		## required to support single element batch of question
		batch_query_mask = batch_query_mask.unsqueeze(0)
		batch_context_mask = batch_context_mask.unsqueeze(0)

		## BiDAF 1 to get ~U, ~h and G (8d) between context and query
		# (N, T, 8d) , (N, T ,2d) , (N, 1, 2d)
		#context_attention_encoded, query_aware_context_encoded, context_aware_query_encoded = self.attention_flow_layer1(query_encoded, context_encoded,batch_query_mask,batch_context_mask)

		query_attention_encoded, context_aware_query_encoded, query_aware_context_encoded = self.attention_flow_layer1(
			context_encoded, query_encoded, batch_context_mask,batch_query_mask)

		## modelling layer 1
		# (N, T, 8d) => (N, T, 2d)
		query_modeled,query_modeled_hidden = self.modeling_layer1(query_attention_encoded, batch_query_length)
		query_modeled = torch.cat([query_encoded_hidden,query_modeled_hidden[-2], query_modeled_hidden[-1]], dim=1)


		'''
		BIDAF 2
		'''
		## BiDAF for answers
		batch_size = batch_candidates_sorted.size(0)
		# N=1 so (N, T, 2d) => (N1, T, 2d)
		context_encoded_2, _ = self.contextual_embedding_layer_2(context_embedded, batch_context_length)
		context_encoded_2 = self._dropout(context_encoded_2)
		batch_context_modeled = context_encoded_2.expand(batch_size,context_encoded_2.size(1), context_encoded_2.size(2))
		# (N1, K, d)

		batch_candidates_embedded = batch_candidates_sorted
		# (N1, K, 2d)
		batch_candidates_encoded,batch_candidates_hidden = self.contextual_embedding_layer_2(batch_candidates_embedded, batch_candidate_lengths_sorted)
		batch_candidates_hidden = torch.cat([batch_candidates_hidden[-2], batch_candidates_hidden[-1]], dim=1)
		batch_candidates_encoded = self._dropout(batch_candidates_encoded)

		answer_attention_encoded, context_aware_answer_encoded, answer_aware_context_encoded = self.attention_flow_layer2(
			batch_context_modeled, batch_candidates_encoded, batch_context_mask, batch_candidate_masks_sorted)

		## modelling layer 2
		# (N1, K, 8d) => (N1, K, 2d)
		answer_modeled, answer_hidden_state = self.modeling_layer2(answer_attention_encoded, batch_candidate_lengths_sorted)
		answer_hidden_state = torch.cat([batch_candidates_hidden, answer_hidden_state[-2], answer_hidden_state[-1]], dim=1)
		answer_modeled = self._dropout(answer_hidden_state)

		answer_scores = torch.mm(answer_modeled, query_modeled.transpose(0, 1))

		## unsort the answer scores
		answer_scores_unsorted = torch.index_select(answer_scores, 0, batch_candidate_unsort)
		loss = self.loss(answer_scores_unsorted.transpose(0,1), gold_index)
		sorted, indices = torch.sort(F.log_softmax(answer_scores_unsorted.squeeze(0),dim=0), dim=0, descending=True)
		return loss, indices


	def eval(self,batch_query, batch_query_length,batch_query_mask,
				batch_context, batch_context_length,batch_context_mask,
			 batch_candidates_sorted, batch_candidate_lengths_sorted,batch_candidate_masks_sorted, batch_candidate_unsort):
		## Embed query and context
		# (N, J, d)
		# query_embedded = self.word_embedding_layer(batch_query.unsqueeze(0))
		# (N, T, d)
		# context_embedded = self.word_embedding_layer(batch_context.unsqueeze(0))

		query_embedded = batch_query.unsqueeze(0)
		context_embedded = batch_context.unsqueeze(0)
		## Encode query and context
		# (N, J, 2d)
		query_encoded, query_encoded_hidden = self.contextual_embedding_layer(query_embedded,
																									batch_query_length)
		query_encoded_hidden = torch.cat([query_encoded_hidden[-2], query_encoded_hidden[-1]], dim=1)
		query_encoded = self._dropout(query_encoded)
		# (N, T, 2d)
		context_encoded, _ = self.contextual_embedding_layer(context_embedded, batch_context_length)
		context_encoded = self._dropout(context_encoded)

		## required to support single element batch of question
		batch_query_mask = batch_query_mask.unsqueeze(0)
		batch_context_mask = batch_context_mask.unsqueeze(0)

		## BiDAF 1 to get ~U, ~h and G (8d) between context and query
		# (N, T, 8d) , (N, T ,2d) , (N, 1, 2d)
		# context_attention_encoded, query_aware_context_encoded, context_aware_query_encoded = self.attention_flow_layer1(query_encoded, context_encoded,batch_query_mask,batch_context_mask)

		query_attention_encoded, context_aware_query_encoded, query_aware_context_encoded = self.attention_flow_layer1(
			context_encoded, query_encoded, batch_context_mask, batch_query_mask)

		## modelling layer 1
		# (N, T, 8d) => (N, T, 2d)
		query_modeled, query_modeled_hidden = self.modeling_layer1(query_attention_encoded,
																						 batch_query_length)
		query_modeled = torch.cat([query_encoded_hidden, query_modeled_hidden[-2], query_modeled_hidden[-1]], dim=1)

		'''
        BIDAF 2
        '''
		## BiDAF for answers
		batch_size = batch_candidates_sorted.size(0)
		# N=1 so (N, T, 2d) => (N1, T, 2d)
		context_encoded_2, _ = self.contextual_embedding_layer_2(context_embedded, batch_context_length)
		context_encoded_2 = self._dropout(context_encoded_2)
		batch_context_modeled = context_encoded_2.expand(batch_size, context_encoded_2.size(1),
														 context_encoded_2.size(2))
		# (N1, K, d)

		batch_candidates_embedded = batch_candidates_sorted
		# (N1, K, 2d)
		batch_candidates_encoded, batch_candidates_hidden = self.contextual_embedding_layer_2(
			batch_candidates_embedded, batch_candidate_lengths_sorted)
		batch_candidates_hidden = torch.cat([batch_candidates_hidden[-2], batch_candidates_hidden[-1]], dim=1)
		batch_candidates_encoded = self._dropout(batch_candidates_encoded)

		answer_attention_encoded, context_aware_answer_encoded, answer_aware_context_encoded = self.attention_flow_layer2(
			batch_context_modeled, batch_candidates_encoded, batch_context_mask, batch_candidate_masks_sorted)

		## modelling layer 2
		# (N1, K, 8d) => (N1, K, 2d)
		answer_modeled, answer_hidden_state = self.modeling_layer2(answer_attention_encoded,
																						batch_candidate_lengths_sorted)
		answer_hidden_state = torch.cat([batch_candidates_hidden, answer_hidden_state[-2], answer_hidden_state[-1]],
										dim=1)
		answer_modeled = self._dropout(answer_hidden_state)

		answer_scores = torch.mm(answer_modeled, query_modeled.transpose(0, 1))

		## unsort the answer scores
		answer_scores_unsorted = torch.index_select(answer_scores, 0, batch_candidate_unsort)
		sorted, indices = torch.sort(F.log_softmax(answer_scores_unsorted.squeeze(0), dim=0), dim=0, descending=True)
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