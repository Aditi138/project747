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
        self.encoder = RecurrentContext(args.embed_size, args.hidden_size, args.num_layers)
        #self.question_encoder = RecurrentContext(args.embed_size, args.hidden_size,args.num_layers)
        #self.chunk_encoder = RecurrentContext(args.embed_size, args.hidden_size,args.num_layers)
        self.modeling_layer = MLP(args.hidden_size)
        self.loss_function=nn.CrossEntropyLoss()
        
    def forward(self, chunks, question, gold_index, context_len,question_len, answer_len, answer):

        chunks_embedded = self.embedding_layer(chunks)
        question_embedded=self.embedding_layer(question)

        chunks_encoded,chunks_encoded_hidden = self.encoder(chunks_embedded, context_len)
        chunks_encoded_hidden = torch.cat([chunks_encoded_hidden[-2], chunks_encoded_hidden[-1]], dim=1)

        question_encoded,question_encoded_hidden=self.encoder(question_embedded,question_len)
        question_encoded_hidden = torch.cat([question_encoded_hidden[-2], question_encoded_hidden[-1]], dim=1)
        question_expanded=question_encoded_hidden.expand(chunks_encoded_hidden.size())

        answer_embedded = self.embedding_layer(answer)
        answer_encoded,answer_encoded_hidden= self.encoder(answer_embedded,answer_len)
        answer_encoded_hidden = torch.cat([answer_encoded_hidden[-2], answer_encoded_hidden[-1]], dim=1)

        answer_expanded = answer_encoded_hidden.expand(chunks_encoded_hidden.size())


        combined_representation=torch.cat([chunks_encoded_hidden, question_expanded, answer_expanded], dim=1)
        scores=self.modeling_layer(combined_representation).squeeze().unsqueeze(0)
        loss=self.loss_function(scores, gold_index)
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
        self.linear1 = nn.Linear(6 * hidden_size, 4 * hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(4 * hidden_size, 2 * hidden_size)
        self.activation = nn.ReLU()
        self.linear3 = nn.Linear(2 * hidden_size, 1)

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