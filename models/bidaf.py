import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

def masked_softmax(vector, mask):
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

def replace_masked_values(tensor, mask, replace_with):
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

def last_dim_softmax(tensor,mask):
	tensor_shape = tensor.size()
	reshaped_tensor = tensor.view(-1, tensor.size()[-1])
	if mask is not None:
		while mask.dim() < tensor.dim():
			mask = mask.unsqueeze(1)
		mask = mask.expand_as(tensor).contiguous().float()
		mask = mask.view(-1, mask.size()[-1])

	reshaped_result = masked_softmax(reshaped_tensor, mask)
	return reshaped_result.view(*tensor_shape)


class BiDAF(nn.Module):
	def __init__(self, input_size):
		super(BiDAF, self).__init__()
		self.input_size = input_size
		self.similarity_layer = nn.Linear(3*input_size, 1)
		self.similarity_layer.bias.data.fill_(1)

	## TODO: Add capability of sentence mask (scoring) for sentence selection model
	def forward(self, U, H, U_mask, H_mask): #H:context U: query
		T = H.size(1)   #Context Length
		J = U.size(1)  #Quesiton Length

		batch_size = U.size(0)


		## compute S from H and U
		## shape of ctx_C: (N, T, 2d) and ctx_Q : # (N, J, 2d)
		## make both matrices of shape (N, T, J, 2d) to compute S

		expanded_U = U.unsqueeze(1).expand((U.size(0), H.size(1), U.size(1), U.size(2)))  # (N, T, J, 2d)
		expanded_H = H.unsqueeze(2).expand((H.size(0), H.size(1), U.size(1), H.size(2)))  # (N, T, J, 2d)

		HU = torch.mul(expanded_H, expanded_U)  # (N, T, J, 4d)
		cat_data = torch.cat([expanded_H, expanded_U, HU], 3)  # (N, T, J, 6d)
		S = self.similarity_layer(cat_data).view(-1, T, J)  # (N, T, J, 1) => (N, T, J)

		#Query aware context representation.
		c2q = torch.bmm(masked_softmax(S, U_mask), U)

		masked_similarity = replace_masked_values(S, U_mask.unsqueeze(1), -1e7)
		print(masked_similarity.size())
		mb = torch.max(masked_similarity, dim=-1)[0].squeeze(-1)
		print(mb.size())
		print(H_mask.size())
		b = masked_softmax(mb, H_mask)

		## (N, 1, 2d) = (N,1,T) * (N, T, 2d)
		q2c = torch.bmm(b.unsqueeze(1), H).squeeze(1)

		## (N, 1, 2d) => (N, T, 2d)

		tiled_q2c = q2c.unsqueeze(1).expand(batch_size, T, q2c.size(-1))

		G = torch.cat([H, c2q, H * c2q, H * tiled_q2c], dim=-1)

		return G, c2q, q2c
