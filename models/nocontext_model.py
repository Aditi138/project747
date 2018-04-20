from seq2seq_attn import EncoderRNN, BahdanauAttnDecoderRNN, Attn
import torch
from torch import nn



class NoContext(nn.Module):
    def __init__(self, args, vocab):
        super(NoContext, self).__init__()
        
        hidden_size = args.hidden_size
        embed_size = args.embed_size
        ner_dim = args.ner_dim
        pos_dim = args.pos_dim
        input_size = vocab.get_length()
        ner_tag_size = vocab.ner_tag_size()
        pos_tag_size = vocab.pos_tag_size()
        self.args = args

        #Embedding layer
        self.embedding = nn.Embedding(input_size, embed_size)
        self.ner_embedding = nn.Embedding(ner_tag_size, ner_dim)
        self.pos_embedding = nn.Embedding(pos_tag_size, pos_dim)
        embed_rep = embed_size + ner_dim + pos_dim

        #Simple Seq2Seq with Bahdanau attention
        self.encoder = EncoderRNN(input_size, embed_rep, hidden_size, n_layers=args.num_layers)
        self.answer_encoder = EncoderRNN(input_size, embed_rep, hidden_size,n_layers=args.num_layers)

        # self.decoder = BahdanauAttnDecoderRNN(hidden_size, embed_size, input_size)

    def forward(self,batch_query, batch_query_ner, batch_query_pos,batch_query_length, batch_candidate, batch_candidate_ner_sorted, batch_candidate_pos_sorted,
             batch_candidate_lengths,batch_candidate_unsort,gold_answer_index,negative_indices,batch_metrics, batch_len):

        query_embedded = self.embedding(batch_query.unsqueeze(0))
        query_ner_embedded = self.ner_embedding(batch_query_ner.unsqueeze(0))
        query_pos_embedded = self.ner_embedding(batch_query_pos.unsqueeze(0))

        query_rep = torch.cat([query_embedded,query_ner_embedded,query_pos_embedded], dim=2)

        encoder_output, encoder_hidden = self.encoder(query_rep, batch_query_length)
        encoder_hidden = torch.cat([encoder_hidden[-2], encoder_hidden[-1]], dim=1)
        query_expanded = encoder_hidden.expand(batch_len, encoder_hidden.size(0), encoder_hidden.size(1))

        answer_embedded = self.embedding(batch_candidate)
        answer_ner_embedded = self.ner_embedding(batch_candidate_ner_sorted)
        answer_pos_embedded = self.pos_embedding(batch_candidate_pos_sorted)

        answer_rep = torch.cat([answer_embedded, answer_ner_embedded, answer_pos_embedded], dim =2)

        answer_encoder_output, answer_encoder_hidden = self.answer_encoder(answer_rep, batch_candidate_lengths)
        answer_encoder_hidden =  torch.cat([answer_encoder_hidden[-2], answer_encoder_hidden[-1]], dim=1)
        
        question_answer_dot = torch.bmm(query_expanded,answer_encoder_hidden.unsqueeze(1).transpose(1,2))

        #Unsort the candidates back to original
        question_answer_dot_unsort = torch.index_select(question_answer_dot, 0, batch_candidate_unsort)

        gold_features = torch.index_select(question_answer_dot_unsort,0,index=gold_answer_index)
        negative_features = torch.index_select(question_answer_dot_unsort,0,index=negative_indices)

        #negative_metrics = torch.index_select(batch_metrics,0,index=negative_indices)
        #negative_features = negative_features.squeeze(2) + negative_metrics.unsqueeze(1)
        max_negative_feature, max_negative_index = torch.max(negative_features, 0)

        loss = torch.clamp(1 - gold_features + max_negative_feature, 0)
        return loss, max_negative_index

    def eval(self,batch_query, batch_query_ner, batch_query_pos,batch_query_length, batch_candidate, batch_candidate_ner_sorted, batch_candidate_pos_sorted,
             batch_candidate_lengths,batch_candidate_unsort,gold_answer_index,batch_metrics, batch_len):
        if self.args.use_cuda:
            batch_query = batch_query.cuda()

        query_embedded = self.embedding(batch_query.unsqueeze(0))
        query_ner_embedded = self.ner_embedding(batch_query_ner.unsqueeze(0))
        query_pos_embedded = self.ner_embedding(batch_query_pos.unsqueeze(0))
        query_rep = torch.cat([query_embedded, query_ner_embedded, query_pos_embedded], dim=2)

        encoder_output, encoder_hidden = self.encoder(query_rep, batch_query_length)
        encoder_hidden = torch.cat([encoder_hidden[-2], encoder_hidden[-1]], dim=1)
        query_expanded = encoder_hidden.expand(batch_len, encoder_hidden.size(0), encoder_hidden.size(1))

        answer_embedded = self.embedding(batch_candidate)
        answer_ner_embedded = self.ner_embedding(batch_candidate_ner_sorted)
        answer_pos_embedded = self.pos_embedding(batch_candidate_pos_sorted)

        answer_rep = torch.cat([answer_embedded, answer_ner_embedded, answer_pos_embedded], dim=2)

        answer_encoder_output, answer_encoder_hidden = self.answer_encoder(answer_rep, batch_candidate_lengths)
        answer_encoder_hidden = torch.cat([answer_encoder_hidden[-2], answer_encoder_hidden[-1]], dim=1)

        question_answer_dot = torch.bmm(query_expanded,answer_encoder_hidden.unsqueeze(1).transpose(1,2))

        #Unsort the candidates back to original
        question_answer_dot_unsort = torch.index_select(question_answer_dot, 0, batch_candidate_unsort)

        sorted, indices = torch.sort(question_answer_dot_unsort,dim=0, descending=True)
        return indices





