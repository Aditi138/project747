import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class ChunkScore(nn.Module):
    def __init__(self, args, vocab_size, pretrain_embedding):
        super(ChunkScore, self).__init__()
        self.embedding_layer = LookupEncoder(vocab_size, args.embed_size, pretrain_embedding=pretrain_embedding)
        # self.question_encoder = EncoderBlock(args.embed_size, args.hidden_size, 3)
        # self.chunk_encoder = EncoderBlock(args.embed_size, args.hidden_size, 3)
        self.answer_encoder = RNNEncoder(args.embed_size, args.hidden_size)
        self.question_encoder = RNNEncoder(args.embed_size, args.hidden_size)
        self.chunk_encoder = RNNEncoder(args.embed_size, args.hidden_size)
        self.modeling_layer = MLP(args.hidden_size)

        
    def forward(self, chunks, question, gold_index):

        chunks_embedded = self.embedding_layer(chunks)
        question_embedded=self.embedding_layer(question)
        chunks_encoded = self.chunk_encoder(chunks_embedded)
        question_encoded=self.question_encoder(question_embedded)

        combined_representation=torch.cat((chunks_encoded, question_encoded), 2).squeeze(dim=1)
        scores=self.modeling_layer(combined_representation).squeeze().unsqueeze(0)
        loss=F.binary_cross_entropy(F.sigmoid(scores), gold_index) / chunks.size(0)
        return loss, scores.data.cpu().numpy()


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, hidden_size, kernel_size):
        super(EncoderBlock, self).__init__()
        self.convolution_layer1=nn.Conv1d(embed_size, hidden_size, kernel_size, padding=(kernel_size-1)/2)
        self.convolution_layer2=nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=(kernel_size-1)/2)
        self.convolution_layer3=nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=(kernel_size-1)/2)        
        self.convolution_layer4=nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=(kernel_size-1)/2)

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
            input=input*mask.unsqueeze(1)
        input1 = F.max_pool1d(input, kernel_size=input.size()[2])
        input2 = F.avg_pool1d(input, kernel_size=input.size()[2])
        input = torch.cat((input1, input2),  1)
        input = input.transpose(1, 2)
        return input

class RNNEncoder(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(RNNEncoder, self).__init__()
        self.encoding_layer = nn.GRU(embed_size, hidden_size, bidirectional=True)

    def forward(self, input, mask=None):
        input = self.encoding_layer(input)[0]
        input = input.transpose(1, 2)
        if mask is not None:
            input=input*mask.unsqueeze(1)
        input1 = F.max_pool1d(input, kernel_size=input.size()[2])
        input2 = F.avg_pool1d(input, kernel_size=input.size()[2])
        input = torch.cat((input1, input2),  1)
        input = input.transpose(1, 2)
        return input


class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(8 * hidden_size, 8 * hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(8 * hidden_size, 4 * hidden_size)
        self.activation = nn.ReLU()
        self.linear3 = nn.Linear(4 * hidden_size, 1)

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