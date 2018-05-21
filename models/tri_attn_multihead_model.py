import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from bidaf import BiDAF, LinearSeqAttn,weighted_avg,BiLinearAttn, log_sum_exp, MultiHeadBiLinearAttn, MultiHeadLinearSeqAttn
import codecs
import numpy as np

class TriAttnMultiHead(nn.Module):
	def __init__(self, args, loader):
		super(TriAttnMultiHead, self).__init__()
		hidden_size = args.hidden_size
		embed_size = args.embed_size
		word_vocab_size = loader.vocab.get_length()

		self.dropout_emb = args.dropout_emb
		self.clark_gardener = args.clark_gardener

		## word embedding layer
		self.word_embedding_layer = LookupEncoder(word_vocab_size, embedding_dim=embed_size,pretrain_embedding=loader.pretrain_embedding)

		## contextual embedding layer (will add when pretraining on Squad)
		# self.contextual_embedding_layer = RecurrentContext(input_size=embed_size, hidden_size=embed_size // 2, num_layers=args.num_layers)

		## bidirectional attention flow between question and context
		self.attention_flow_c2q = BiDAF(embed_size)
		self.attention_flow_a2q = BiDAF(embed_size)
		self.attention_flow_a2c = BiDAF(embed_size)

		## modelling layer for question and context : this layer also converts the 8 dimensional input intp two dimensioanl output
		self.modeling_layer_q = RecurrentContext(embed_size, hidden_size)
		self.modeling_layer_c = RecurrentContext(embed_size * 2, hidden_size)
		self.modeling_layer_a = RecurrentContext(embed_size * 3, hidden_size)

		self.self_attn_q = MultiHeadLinearSeqAttn(2 * hidden_size)
		self.self_attn_c = MultiHeadBiLinearAttn(2 * hidden_size,2 * hidden_size)
		self.self_attn_a = MultiHeadLinearSeqAttn(2 * hidden_size)


		## output layer
		output_layer_inputdim = 4 * hidden_size
		self.output_layer = OutputLayer(output_layer_inputdim, hidden_size)
		self.answer_context_bilinear = nn.Linear(2 * hidden_size, 2 * hidden_size)
		self.query_answer_bilinear = nn.Linear(2 * hidden_size, 2 * hidden_size)
		self.loss = torch.nn.CrossEntropyLoss()

		self.dropout = torch.nn.Dropout(args.dropout)

		## sentence scoring module
		self.sentence_scorer = SentenceScorer(args, loader)



	def forward(self, query_embedded, batch_query_length,batch_query_mask,
				context_embedded, batch_context_length,batch_context_mask,batch_context_scores,
				batch_candidates_embedded, batch_candidate_lengths_sorted, batch_candidate_masks_sorted,batch_candidate_unsort,
				gold_index, gold_chunk):
		query_embedded = self.word_embedding_layer(query_embedded)
		context_embedded = self.word_embedding_layer(context_embedded)
		batch_candidates_embedded = self.word_embedding_layer(batch_candidates_embedded)

		# dropout emb
		query_embedded = nn.functional.dropout(query_embedded, p=self.dropout_emb, training=True).unsqueeze(0)
		context_embedded = nn.functional.dropout(context_embedded, p=self.dropout_emb, training=True)
		batch_candidates_embedded = nn.functional.dropout(batch_candidates_embedded, p=self.dropout_emb, training=True)

		batch_query_mask = batch_query_mask.unsqueeze(0)



		num_chunks = context_embedded.size(0)
		query_embedded_chunk_wise = query_embedded.expand(num_chunks, query_embedded.size(1), query_embedded.size(2))
		batch_query_mask_chunk_wise = batch_query_mask.expand(num_chunks, batch_query_mask.size(1))

		batch_size = batch_candidates_embedded.size(0)
		context_encoded = context_embedded
		query_aware_context_encoded, c2q_attention_matrix = self.attention_flow_c2q(query_embedded_chunk_wise,
																					context_encoded,
																					batch_query_mask_chunk_wise,
																					batch_context_mask)
		query_aware_context_encoded = self.dropout(query_aware_context_encoded)
		query_aware_answer_encoded, _ = self.attention_flow_a2q(
			query_embedded.expand(batch_size, query_embedded.size(1), query_embedded.size(2)),
			batch_candidates_embedded,
			batch_query_mask.expand(batch_size, batch_query_mask.size(1)), batch_candidate_masks_sorted)

		context_combined = context_encoded.view(-1, context_encoded.size(2)).unsqueeze(0)

		context_aware_answer_encoded, _ = self.attention_flow_a2c(
			context_combined.expand(batch_size, context_combined.size(1), context_combined.size(2)),
			batch_candidates_embedded,
			batch_context_mask.expand(batch_size, batch_context_mask.size(0), batch_context_mask.size(1)),
			batch_candidate_masks_sorted, split=True, num_chunks=num_chunks)

		query_input_modeled, _ = self.modeling_layer_q(query_embedded, batch_query_length)
		query_input_modeled = self.dropout(query_input_modeled)

		context_input_modelling = torch.cat([query_aware_context_encoded, context_encoded], dim=-1)
		context_modeled, _ = self.modeling_layer_c(context_input_modelling, batch_context_length)  # (N, |C|, 2d)
		context_modeled = self.dropout(context_modeled)

		# answer_modeled = batch_candidates_encoded
		answer_input_modelling = torch.cat(
			[query_aware_answer_encoded.unsqueeze(1).expand(-1,num_chunks,-1,-1),
			 context_aware_answer_encoded,
			 batch_candidates_embedded.unsqueeze(1).expand(-1,num_chunks,-1,-1)], dim=-1)
		answer_modeled, _ = self.modeling_layer_a(answer_input_modelling.view(-1, answer_input_modelling.size(2), answer_input_modelling.size(3)),
												  np.repeat(batch_candidate_lengths_sorted, num_chunks))  # (N, |A|, 2d)


		## self attention (multiheaded self attention)
		query_self_attention, query_input_modelled_transformed = self.self_attn_q(query_input_modeled, batch_query_mask)
		q_hidden = weighted_avg(query_input_modelled_transformed, query_self_attention)

		answer_self_attention, answer_modeled_transformed = self.self_attn_a(answer_modeled,
												 batch_candidate_masks_sorted.unsqueeze(1).expand(-1, num_chunks,-1).contiguous().view(batch_size*num_chunks, -1))
		a_hidden = weighted_avg(answer_modeled_transformed, answer_self_attention)

		context_self_attention, context_modeled_transformed = self.self_attn_c(context_modeled, q_hidden.expand(num_chunks, q_hidden.size(1)),
												  batch_context_mask)
		c_hidden = weighted_avg(context_modeled_transformed, context_self_attention)

		logits_qa = self.query_answer_bilinear(q_hidden) * a_hidden  # (N, 2d)
		logits_qa = logits_qa.view(batch_size, num_chunks, -1)  # (N,k,2d)

		context_chunk_wise = self.answer_context_bilinear(c_hidden)  # (K, 2d)
		context_chunk_wise = context_chunk_wise.expand(batch_size, context_chunk_wise.size(0),
													   context_chunk_wise.size(1))  # (N,K,2d)
		logits_ca = context_chunk_wise * a_hidden.view(batch_size, num_chunks, -1)  # (N,K,2d)

		scores = self.output_layer(torch.cat([logits_qa, logits_ca], dim=-1))  # (N,K,4d) ==>#(N,K,1)

		## call sentence scorer here
		# batch_candidates_embedded_unsorted = torch.index_select(batch_candidates_embedded, 0, batch_candidate_unsort)
		# batch_candidate_masks_unsorted = torch.index_select(batch_candidate_masks_sorted, 0, batch_candidate_unsort)
		batch_context_scores = self.sentence_scorer(query_embedded, context_embedded, batch_context_mask,
													gold_chunk)

		weighted_candidates = scores.squeeze(-1) + batch_context_scores  # (N,K)

		if self.clark_gardener:
			log_weighted_candidates = log_sum_exp(weighted_candidates[:, gold_chunk].unsqueeze(1), dim=-1)
		else:
			log_weighted_candidates = log_sum_exp(weighted_candidates, dim=-1)  # (N)

		log_denominator = log_sum_exp(weighted_candidates.view(-1), dim=0)

		answer_scores =  log_denominator  - log_weighted_candidates# (N)

		## unsort the answer scores
		answer_scores = torch.index_select(answer_scores, 0, batch_candidate_unsort)
		loss = answer_scores[gold_index]
		sorted, indices = torch.sort(answer_scores, dim=0, descending=False)
		return loss, indices

	def eval(self,query_embedded, batch_query_length,batch_query_mask,
				context_embedded, batch_context_length,batch_context_mask,batch_context_scores,
				batch_candidates_embedded, batch_candidate_lengths_sorted, batch_candidate_masks_sorted,batch_candidate_unsort
				):
		query_embedded = self.word_embedding_layer(query_embedded)
		context_embedded = self.word_embedding_layer(context_embedded)
		batch_candidates_embedded = self.word_embedding_layer(batch_candidates_embedded)

		# dropout emb
		query_embedded = nn.functional.dropout(query_embedded, p=self.dropout_emb, training=False).unsqueeze(0)
		context_embedded = nn.functional.dropout(context_embedded, p=self.dropout_emb, training=False)
		batch_candidates_embedded = nn.functional.dropout(batch_candidates_embedded, p=self.dropout_emb, training=False)

		batch_query_mask = batch_query_mask.unsqueeze(0)

		# contextual layer
		context_encoded = context_embedded
		# context_encoded, _ = self.contextual_embedding_layer(context_embedded, batch_context_length)
		# context_encoded = nn.functional.dropout(context_encoded, p=self.dropout_emb, training=True)


		num_chunks = context_embedded.size(0)
		query_embedded_chunk_wise = query_embedded.expand(num_chunks, query_embedded.size(1), query_embedded.size(2))
		batch_query_mask_chunk_wise = batch_query_mask.expand(num_chunks, batch_query_mask.size(1))

		batch_size = batch_candidates_embedded.size(0)
		context_encoded = context_embedded

		query_aware_context_encoded, c2q_attention_matrix = self.attention_flow_c2q(query_embedded_chunk_wise,
																					context_encoded,
																					batch_query_mask_chunk_wise,
																					batch_context_mask)
		query_aware_context_encoded = self.dropout(query_aware_context_encoded)

		query_aware_answer_encoded, _ = self.attention_flow_a2q(
			query_embedded.expand(batch_size, query_embedded.size(1), query_embedded.size(2)),
			batch_candidates_embedded,
			batch_query_mask.expand(batch_size, batch_query_mask.size(1)), batch_candidate_masks_sorted)

		context_combined = context_encoded.view(-1, context_encoded.size(2)).unsqueeze(0)

		context_aware_answer_encoded, _ = self.attention_flow_a2c(
			context_combined.expand(batch_size, context_combined.size(1), context_combined.size(2)),
			batch_candidates_embedded,
			batch_context_mask.expand(batch_size, batch_context_mask.size(0), batch_context_mask.size(1)),
			batch_candidate_masks_sorted, split=True, num_chunks=num_chunks)

		query_input_modeled, _ = self.modeling_layer_q(query_embedded, batch_query_length)
		query_input_modelled = self.dropout(query_input_modeled)

		context_input_modelling = torch.cat([query_aware_context_encoded, context_encoded], dim=-1)
		context_modeled, _ = self.modeling_layer_c(context_input_modelling, batch_context_length)  # (N, |C|, 2d)
		context_modeled = self.dropout(context_modeled)

		# answer_modeled = batch_candidates_encoded
		answer_input_modelling = torch.cat(
			[query_aware_answer_encoded.unsqueeze(1).expand(-1, num_chunks, -1, -1),
			 context_aware_answer_encoded,
			 batch_candidates_embedded.unsqueeze(1).expand(-1, num_chunks, -1, -1)], dim=-1)
		answer_modeled, _ = self.modeling_layer_a(
			answer_input_modelling.view(-1, answer_input_modelling.size(2), answer_input_modelling.size(3)),
			np.repeat(batch_candidate_lengths_sorted, num_chunks))  # (N, |A|, 2d)

		query_self_attention = self.self_attn_q(query_input_modelled, batch_query_mask)
		q_hidden = weighted_avg(query_input_modelled, query_self_attention)

		answer_self_attention = self.self_attn_a(answer_modeled,
												 batch_candidate_masks_sorted.unsqueeze(1).expand(-1, num_chunks,
																								  -1).contiguous().view(
													 batch_size * num_chunks, -1))
		a_hidden = weighted_avg(answer_modeled, answer_self_attention)

		context_self_attention = self.self_attn_c(context_modeled, q_hidden.expand(num_chunks, q_hidden.size(1)),
												  batch_context_mask)
		c_hidden = weighted_avg(context_modeled, context_self_attention)

		logits_qa = self.query_answer_bilinear(q_hidden) * a_hidden  # (N, 2d)
		logits_qa = logits_qa.view(batch_size, num_chunks, -1)  # (N,k,2d)

		context_chunk_wise = self.answer_context_bilinear(c_hidden)  # (K, 2d)
		context_chunk_wise = context_chunk_wise.expand(batch_size, context_chunk_wise.size(0),
													   context_chunk_wise.size(1))  # (N,K,2d)
		logits_ca = context_chunk_wise * a_hidden.view(batch_size, num_chunks, -1)  # (N,K,2d)


		scores = self.output_layer(torch.cat([logits_qa, logits_ca], dim=-1))  # (N,K,4d) ==>#(N,K,1)

		## call sentence scorer here
		# batch_candidates_embedded_unsorted = torch.index_select(batch_candidates_embedded, 0, batch_candidate_unsort)
		# batch_candidate_masks_unsorted = torch.index_select(batch_candidate_masks_sorted, 0, batch_candidate_unsort)
		batch_context_scores = self.sentence_scorer.eval(query_embedded, context_embedded, batch_context_mask)

		weighted_candidates = scores.squeeze(-1) + batch_context_scores  # (N,K)

		log_weighted_candidates = log_sum_exp(weighted_candidates, dim=-1)  # (N)
		log_denominator = log_sum_exp(weighted_candidates.view(-1), dim=0)

		answer_scores = log_weighted_candidates  # (N)

		## unsort the answer scores
		answer_scores = torch.index_select(answer_scores, 0, batch_candidate_unsort)
		sorted, indices = torch.sort(answer_scores, dim=0, descending=True)
		return indices, c2q_attention_matrix


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