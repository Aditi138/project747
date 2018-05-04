import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class ChunkScore(nn.Module):
    def __init__(self, args, vocab_size, pretrain_embedding):
        super(ChunkScore, self).__init__()
        self.embedding_layer = LookupEncoder(vocab_size, args.embed_size, pretrain_embedding=pretrain_embedding)
        self.question_encoder = EncoderBlock(args.embed_size, args.hidden_size, 3)
        self.chunk_encoder = EncoderBlock(args.embed_size, args.hidden_size, 3)
        self.modeling_layer = MLP(args.hidden_size)
        self.loss_function=nn.CrossEntropyLoss()
        
    def forward(self, chunks, question, gold_index):

        chunks_embedded = self.embedding_layer(chunks)
        question_embedded=self.embedding_layer(question)
        chunks_encoded = self.chunk_encoder(chunks_embedded)
        question_encoded=self.question_encoder(question_embedded)
        question_expanded=question_encoded.expand(chunks_encoded.size())
        combined_representation=torch.cat((chunks_encoded, question_expanded), 2).squeeze(dim=1)
        scores=self.modeling_layer(combined_representation).squeeze().unsqueeze(0)
        loss=self.loss_function(scores, gold_index)
        return loss, scores.data.numpy()


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, hidden_size, kernel_size):
        super(EncoderBlock, self).__init__()
        self.convolution_layer=nn.Conv1d(embed_size, hidden_size, kernel_size)
        self.activation = nn.ReLU()

    def forward(self, input):
        input = input.transpose(1, 2)
        input = self.convolution_layer(input)
        input = self.activation(input)
        input = F.max_pool1d(input, kernel_size=input.size()[2])
        input = input.transpose(1, 2)
        return input


class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(2 * hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, input):
        input = self.linear1(input)
        input = self.activation(input)
        input = self.linear2(input)
        input = self.activation(input)
        input = self.linear3(input)
        return input

class LookupEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrain_embedding=None):
        super(LookupEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if pretrain_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrain_embedding))

    def forward(self, batch):
        return self.word_embeddings(batch)

# class QuestionEncoder(nn.Module):
#     def __init__(self, embed_size, hidden_size, kernel_size):
#         super(QuestionEncoder, self).__init__()
#         self.convolution_layer=nn.Conv1d(embed_size, hidden_size, kernel_size)
#         self.activation = nn.ReLU
#         self.pool = F.max_pool1d(input, kernel_size=input.size()[2])

#     def forward(self):

#         return

# class ChunkEncoder(nn.Module):
#     def __init__(self):
#         super(ChunkEncoder, self).__init__()

#     def forward(self):
#         return