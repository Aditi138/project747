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

    def forward(self, input_sequence,input_lengths,batch_size,output_seq_length):
        if self.args.use_cuda:
            input_sequence = input_sequence.cuda()
        encoder_output, encoder_hidden = self.encoder(input_sequence, input_lengths)

        decoder_input = Variable(torch.LongTensor([self.dataloader.SOS_Token] * batch_size))
        all_decoder_outputs = Variable(torch.zeros(output_seq_length, batch_size, self.word_vocab_size))



