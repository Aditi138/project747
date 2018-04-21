import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from bidaf import BiDAF

def masked_log_softmax(vector, mask):
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    We assume that both ``vector`` and ``mask`` (if given) have shape ``(batch_size, vector_dim)``.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=1)

def masked_softmax(self, vector, mask):
	"""
	``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
	masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
	``None`` in for the mask is also acceptable; you'll just get a regular softmax.
	We assume that both ``vector`` and ``mask`` (if given) have shape ``(batch_size, vector_dim)``.
	In the case that the input vector is completely masked, this function returns an array
	of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of a model
	that uses categorical cross-entropy loss.
	"""
	if mask is None:
		result = F.softmax(vector, dim=-1)
	else:
		# To limit numerical errors from large vector elements outside the mask, we zero these out.
		result = torch.nn.functional.softmax(vector * mask, dim=-1)
		result = result * mask
		result = result / (result.sum(dim=1, keepdim=True) + 1e-13)
	return result

def replace_masked_values(self,tensor, mask, replace_with):
	"""
	Replaces all masked values in ``tensor`` with ``replace_with``.  ``mask`` must be broadcastable
	to the same shape as ``tensor``. We require that ``tensor.dim() == mask.dim()``, as otherwise we
	won't know which dimensions of the mask to unsqueeze.
	"""
	# We'll build a tensor of the same shape as `tensor`, zero out masked values, then add back in
	# the `replace_with` value.
	if tensor.dim() != mask.dim():
		raise Exception("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
	one_minus_mask = 1.0 - mask
	values_to_add = replace_with * one_minus_mask
	return tensor * mask + values_to_add

def weighted_sum(matrix, attention):
	if attention.dim() == 2 and matrix.dim() == 3:
		return attention.unsqueeze(1).bmm(matrix).squeeze(1)
	if attention.dim() == 3 and matrix.dim() == 3:
		return attention.bmm(matrix)
	if matrix.dim() - 1 < attention.dim():
		expanded_size = list(matrix.size())
		for i in range(attention.dim() - matrix.dim() + 1):
			matrix = matrix.unsqueeze(1)
			expanded_size.insert(i + 1, attention.size(i + 1))
		matrix = matrix.expand(*expanded_size)
	intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
	return intermediate.sum(dim=-2)

class ContextMRR(nn.Module):
	def __init__(self, args, vocab):
		super(ContextMRR, self).__init__()
		hidden_size = args.hidden_size
		embed_size = args.embed_size
		word_vocab_size = vocab.get_length()

		if args.dropout > 0:
			self._dropout = torch.nn.Dropout(p=args.dropout)
		else:
			self._dropout = lambda x: x

		## word embedding layer
		self.word_embedding_layer = LookupEncoder(word_vocab_size, embedding_dim=embed_size)

		## contextual embedding layer
		self.contextual_embedding_layer = RecurrentContext(input_size=embed_size, hidden_size=hidden_size, num_layers=1)

		## bidirectional attention flow between question and context
		self.attention_flow_layer1 = BiDAF(2*hidden_size)

		## modelling layer for question and context : this layer also converts the 8 dimensional input intp two dimensioanl output
		modeling_layer_inputdim = 8 * hidden_size
		self.modeling_layer1 = RecurrentContext(modeling_layer_inputdim, hidden_size)
		self.modeling_dim = 2 * hidden_size


		span_start_input_dim = modeling_layer_inputdim + (2 * hidden_size)
		self._span_predictor = TimeDistributed(torch.nn.Linear(span_start_input_dim, 1))

		span_end_input_dim = modeling_layer_inputdim + (2 * hidden_size)
		self._span_end_predictor = TimeDistributed(torch.nn.Linear(span_end_input_dim, 1))

		span_end_dim = modeling_layer_inputdim + 3 * self.modeling_dim
		self._span_end_encoder = RecurrentContext(span_end_dim, hidden_size)

		self._span_start_accuracy = Accuracy()
		self._span_end_accuracy = Accuracy()
		self._span_accuracy = Accuracy()


	def forward(self, batch_query, batch_query_length,batch_query_mask,
				batch_context, batch_context_length,batch_context_mask,
				batch_candidates_sorted, batch_candidate_lengths_sorted, batch_candidate_masks_sorted,batch_candidate_unsort,
				gold_index, negative_indices, batch_metrics, span_start, span_end):

		## Embed query and context
		# (N, J, d)
		query_embedded = self.word_embedding_layer(batch_query)
		# (N, T, d)
		context_embedded = self.word_embedding_layer(batch_context)

		passage_length = context_embedded.size(1)

		## Encode query and context
		# (N, J, 2d)
		query_encoded,_ = self.contextual_embedding_layer(query_embedded, batch_query_length)
		query_encoded = self._dropout(query_encoded)
		# (N, T, 2d)
		context_encoded,_ = self.contextual_embedding_layer(context_embedded, batch_context_length)
		context_encoded = self._dropout(context_encoded)

		## BiDAF 1 to get ~U, ~h and G (8d) between context and query
		# (N, T, 8d) , (N, T ,2d) , (N, 1, 2d)

		context_attention_encoded, query_aware_context_encoded, context_aware_query_encoded = self.attention_flow_layer1(query_encoded, context_encoded,batch_query_mask,batch_context_mask)

		## modelling layer 1
		# (N, T, 8d) => (N, T, 2d)
		context_modeled,_ = self._dropout(self.modeling_layer1(context_attention_encoded, batch_context_length))

		# Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim))
		span_start_input = self._dropout(torch.cat([context_attention_encoded, context_modeled], dim=-1))

		# Shape: (batch_size, passage_length)
		span_start_logits = self._span_start_predictor(span_start_input).squeeze(-1)

		# Shape: (batch_size, passage_length)
		span_start_probs = masked_softmax(span_start_logits, batch_context_mask)

		# Shape: (batch_size, modeling_dim)
		span_start_representation = weighted_sum(context_modeled, span_start_probs)

		# Shape: (batch_size, passage_length, modeling_dim)
		tiled_start_representation = span_start_representation.unsqueeze(1).expand(span_start_representation.size(0),
																				   passage_length,
																				   self.modeling_dim)

		# Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim * 3)
		span_end_representation = torch.cat([context_attention_encoded,
											 context_modeled,
											 tiled_start_representation,
											 context_modeled * tiled_start_representation],
											dim=-1)

		# Shape: (batch_size, passage_length, encoding_dim)
		encoded_span_end = self._dropout(self._span_end_encoder(span_end_representation, batch_context_length))

		# Shape: (batch_size, passage_length, encoding_dim * 4 + span_end_encoding_dim)
		span_end_input = self._dropout(torch.cat([context_attention_encoded, encoded_span_end], dim=-1))

		span_end_logits = self._span_end_predictor(span_end_input).squeeze(-1)
		span_end_probs = masked_softmax(span_end_logits, batch_context_mask)

		span_start_logits = replace_masked_values(span_start_logits, batch_context_mask, -1e7)
		span_end_logits = replace_masked_values(span_end_logits, batch_context_mask	, -1e7)

		best_span = self.get_best_span(span_start_logits, span_end_logits)

		# Compute the loss for training.
		if span_start is not None:
			loss = F.nll_loss(masked_log_softmax(span_start_logits, batch_context_mask), span_start.squeeze(-1))
			self._span_start_accuracy.accuracy(span_start_logits, span_start.squeeze(-1))
			loss += F.nll_loss(masked_log_softmax(span_end_logits, batch_context_mask), span_end.squeeze(-1))
			self._span_end_accuracy.accuracy(span_end_logits, span_end.squeeze(-1))
			self._span_accuracy.accuracy(best_span, torch.stack([span_start, span_end], -1))
			return loss

	def get_best_span(self,span_start_logits, span_end_logits):
		if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
			raise ValueError("Input shapes must be (batch_size, passage_length)")

		batch_size, passage_length = span_start_logits.size()
		max_span_log_prob = [-1e20] * batch_size
		span_start_argmax = [0] * batch_size
		best_word_span = Variable(span_start_logits.data.new()
								  .resize_(batch_size, 2).fill_(0)).long()

		span_start_logits = span_start_logits.data.cpu().numpy()
		span_end_logits = span_end_logits.data.cpu().numpy()

		for b in range(batch_size):  # pylint: disable=invalid-name
			for j in range(passage_length):
				val1 = span_start_logits[b, span_start_argmax[b]]
				if val1 < span_start_logits[b, j]:
					span_start_argmax[b] = j
					val1 = span_start_logits[b, j]

				val2 = span_end_logits[b, j]

				if val1 + val2 > max_span_log_prob[b]:
					best_word_span[b, 0] = span_start_argmax[b]
					best_word_span[b, 1] = j
					max_span_log_prob[b] = val1 + val2
		return best_word_span

	def eval(self,batch_query, batch_query_length,batch_query_mask,
				batch_context, batch_context_length,batch_context_mask,
			 batch_candidates_sorted, batch_candidate_lengths_sorted,batch_candidate_masks_sorted, batch_candidate_unsort):
		## Embed query and context
		# (N, J, d)
		query_embedded = self.word_embedding_layer(batch_query.unsqueeze(0))
		# (N, T, d)
		context_embedded = self.word_embedding_layer(batch_context.unsqueeze(0))

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
		batch_candidates_embedded = self.word_embedding_layer(batch_candidates_sorted)
		# (N1, K, 2d)
		batch_candidates_encoded, _ = self.contextual_embedding_layer(batch_candidates_embedded,
																	  batch_candidate_lengths_sorted)
		answer_attention_encoded, context_aware_answer_encoded, answer_aware_context_encoded = self.attention_flow_layer2(
			batch_context_modeled, batch_candidates_encoded,batch_context_mask,batch_candidate_masks_sorted)

		## modelling layer 2
		# (N1, K, 8d) => (N1, K, 2d)
		answer_modeled, (answer_hidden_state, answer_cell_state) = self.modeling_layer2(answer_attention_encoded,
																						batch_candidate_lengths_sorted,)

		## output layer : concatenate hidden dimension of the final answer model layer and run through an MLP : (N1, 2d) => (N1, d)
		# (N1, 2d) => (N1, 1)
		answer_concat_hidden = torch.cat([answer_hidden_state[-2], answer_hidden_state[-1]], dim=1)
		answer_scores = self.output_layer(answer_concat_hidden)

		## unsort the answer scores
		answer_scores_unsorted = torch.index_select(answer_scores, 0, batch_candidate_unsort)
		sorted, indices = torch.sort(answer_scores_unsorted, dim=0, descending=True)
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
			nn.Sigmoid(),
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
		# self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrain_embedding))

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



class Accuracy:
	def __init__(self,top_k=1):
		self.top_k = top_k
		self.correct_count = 0.0
		self.total_count = 0.0

	def unwrap_to_tensors(*tensors):
		"""
	    If you actually passed in Variables to a Metric instead of Tensors, there will be
	    a huge memory leak, because it will prevent garbage collection for the computation
	    graph. This method ensures that you're using tensors directly and that they are on
	    the CPU.
	    """
		return (x.data.cpu() if isinstance(x, torch.autograd.Variable) else x for x in tensors)


	def accuracy(self,predictions, gold_labels,mask ):
		# Get the data from the Variables.
		predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

		# Some sanity checks.
		num_classes = predictions.size(-1)

		if gold_labels.dim() != predictions.dim() - 1:
			raise Exception("gold_labels must have dimension == predictions.size() - 1 but "
									 "found tensor of shape: {}".format(predictions.size()))

		if (gold_labels >= num_classes).any():
			raise Exception("A gold label passed to Categorical Accuracy contains an id >= {}, "
									 "the number of classes.".format(num_classes))

		# Top K indexes of the predictions (or fewer, if there aren't K of them).
		# Special case topk == 1, because it's common and .max() is much faster than .topk().
		if self.top_k == 1:
			top_k = predictions.max(-1)[1].unsqueeze(-1)
		else:
			top_k = predictions.topk(min(self.top_k, predictions.shape[-1]), -1)[1]

		# This is of shape (batch_size, ..., top_k).
		correct = top_k.eq(gold_labels.long().unsqueeze(-1)).float()

		if mask is not None:
			correct *= mask.float().unsqueeze(-1)
			self.total_count += mask.sum()
		else:
			self.total_count += gold_labels.numel()
		self.correct_count += correct.sum()