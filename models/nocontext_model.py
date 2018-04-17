from seq2seq_attn import EncoderRNN, BahdanauAttnDecoderRNN, Attn
import torch
from torch import nn


class NoContext(nn.Module):
    def __init__(self, args, vocab_size):
        super(NoContext, self).__init__()
        
        hidden_size = args.hidden_size
        embed_size = args.embed_size

        #Simple Seq2Seq with Bahdanau attention
        self.encoder = EncoderRNN(vocab_size, embed_size, hidden_size)
        self.decoder = BahdanauAttnDecoderRNN(hidden_size, embed_size, vocab_size)

    def forward(self,batch_query, batch_query_length, batch_candidate, batch_candidate_lengths,batch_candidate_unsort,gold_answer_index, negative_indices, batch_metrics, batch_len):

        encoder_output, encoder_hidden = self.encoder(batch_query.unsqueeze(0), batch_query_length)
        query_expanded = encoder_hidden.expand(batch_len, encoder_hidden.size(0), encoder_hidden.size(1))

        answer_encoder_output, answer_encoder_hidden = self.encoder(batch_candidate, batch_candidate_lengths)

        question_answer_dot = torch.bmm(query_expanded,answer_encoder_hidden.unsqueeze(1).transpose(1,2))

        #Unsort the candidates back to original
        question_answer_dot_unsort = torch.index_select(question_answer_dot, 0, batch_candidate_unsort)

        gold_features = torch.index_select(question_answer_dot_unsort,0,index=gold_answer_index)
        negative_features = torch.index_select(question_answer_dot_unsort,0,index=negative_indices)

        negative_metrics = torch.index_select(batch_metrics,0,index=negative_indices)
        negative_features = negative_features.squeeze(2) + negative_metrics.unsqueeze(1)
        max_negative_feature, max_negative_index = torch.max(negative_features, 0)

        loss = max_negative_feature - gold_features
        return loss, max_negative_index

    def eval(self,batch_query, batch_query_length, batch_candidate, batch_candidate_lengths,batch_candidate_unsort,gold_answer_index,batch_metrics, batch_len):
        if self.args.use_cuda:
            batch_query = batch_query.cuda()


        encoder_output, encoder_hidden = self.encoder(batch_query.unsqueeze(0), batch_query_length)
        query_expanded = encoder_hidden.expand(batch_len, encoder_hidden.size(0), encoder_hidden.size(1))

        answer_encoder_output, answer_encoder_hidden = self.encoder(batch_candidate, batch_candidate_lengths)

        question_answer_dot = torch.bmm(query_expanded,answer_encoder_hidden.unsqueeze(1).transpose(1,2))

        #Unsort the candidates back to original
        question_answer_dot_unsort = torch.index_select(question_answer_dot, 0, batch_candidate_unsort)

        sorted, indices = torch.sort(question_answer_dot_unsort,dim=0, descending=True)
        return indices





