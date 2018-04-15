from seq2seq_attn import *


class Model(nn.Module):
    def __init__(self, args, dataloader):
        super(Model, self).__init__()
        self.args = args
        self.dataloader = dataloader

        hidden_size = args.hidden_size
        embed_size = args.embed_size
        input_size = dataloader.vocab.get_length()

        #Simple Seq2Seq with Bahdanau attention
        self.encoder = EncoderRNN(input_size, embed_size, hidden_size)
        self.decoder = BahdanauAttnDecoderRNN(hidden_size, embed_size, input_size)

        if args.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

    def forward(self,batch_query, batch_query_length, batch_candidate, batch_candidate_lengths,batch_candidate_unsort,gold_answer_index,batch_metrics, batch_len):
        if self.args.use_cuda:
            batch_query = batch_query.cuda()


        encoder_output, encoder_hidden = self.encoder(batch_query.unsqueeze(0), batch_query_length)
        query_expanded = encoder_hidden.expand(batch_len, encoder_hidden.size(0), encoder_hidden.size(1))

        answer_encoder_output, answer_encoder_hidden = self.encoder(batch_candidate, batch_candidate_lengths)

        question_answer_dot = torch.bmm(query_expanded,answer_encoder_hidden.unsqueeze(1).transpose(1,2))

        #Unsort the candidates back to original
        question_answer_dot_unsort = torch.index_select(question_answer_dot, 0, batch_candidate_unsort)

        gold_index = Variable(torch.LongTensor([gold_answer_index]))
        negative_indices = [idx for idx in range(batch_len)]
        negative_indices.pop(gold_answer_index)
        negative_indices = Variable(torch.LongTensor(negative_indices))

        if self.args.use_cuda:
            gold_index = gold_index.cuda()
            negative_indices = negative_indices.cuda()

        gold_features = torch.index_select(question_answer_dot_unsort,0,index=gold_index)
        negative_features = torch.index_select(question_answer_dot_unsort,0,index=negative_indices)
        
        negative_metrics = torch.index_select(batch_metrics,0,index=negative_indices)
        negative_features = negative_features.squeeze(2) + negative_metrics.unsqueeze(1)
        max_negtaive_feature, max_negative_index = torch.max(negative_features, 0)

        loss = max_negtaive_feature - gold_features
        return loss,max_negative_index

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





