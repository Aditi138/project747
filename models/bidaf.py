import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class BiDAF(nn.Module):
	def __init__(self, input_size):
		super(BiDAF, self).__init__()
		self.input_size = input_size
		self.similarity_layer = nn.Linear(3*input_size, 1)

	## TODO: Add capability of sentence mask (scoring) for sentence selection model
	def forward(self, U, H):
		T = H.size(1)
		J = U.size(1)

		## compute S from H and U
		## shape of ctx_C: (N, T, 2d) and ctx_Q : # (N, J, 2d)
		## make both matrices of shape (N, T, J, 2d) to compute S
		expanded_H = H.unsqueeze(2).expand((-1,-1, J,-1)) # (N, T, J, 2d)
		expanded_U = U.unsqueeze(1).expand(-1, T, -1,-1) # (N, T, J, 2d)
		HU = torch.mul(expanded_H, expanded_U) # (N, T, J, 4d)
		cat_data = torch.cat((expanded_H, expanded_U, HU), 3) # (N, T, J, 6d)
		S = self.similarity_layer(cat_data).view(-1, T, J) # (N, T, J, 1) => (N, T, J)

		## compute ~U (context 2 query)
		## softmax along J's dimension
		## (N, T, 2d) = (N, T, J) X (N, J, 2d)
		c2q = torch.bmm(F.softmax(S, dim=2), U)

		## compute ~h and expand to ~H
		b = F.softmax(torch.max(S, 2)[0], -1)
		## (N, 1, 2d) = (N,1,T) * (N, T, 2d)
		q2c = torch.bmm(b.unsqueeze(1), H)
		## (N, 1, 2d) => (N, T, 2d)
		tiled_q2c = q2c.repeat(1,T,1)

		## G : concatenate H, ~U, H.~U and H.~H
		# (N,T,2d): (N,T,2d) : (N,T,2d) => (N,T,6d)
		G = torch.cat((H, c2q, torch.mul(H,c2q), torch.mul(H, tiled_q2c)), dim=2)
		return G, c2q, q2c

