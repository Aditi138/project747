import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
def variable(v, arg_use_cuda = True, volatile=False):
    if use_cuda and arg_use_cuda:
        return Variable(v, volatile=volatile).cuda()
    return Variable(v, volatile=volatile)