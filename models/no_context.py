from models.model import Model
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NoContext(Model):
    def __init__(self):
        pass

class BILSTMsim(NoContext):
    def __init__(self, input_dim, hidden_dim):
        self.question_encoding = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.answer_encoding = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)

    def forward(questions, answers):
        pass
        
