try:
    import cPickle as pickle
except:
    import pickle
from allennlp.modules.elmo import Elmo
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper
elmo_tokenize=ELMoCharacterMapper.convert_word_to_char_ids
import torch
from torch.autograd import Variable
import numpy as np
from numpy import random
from dataloaders.utility import options_url, weights_url, use_cuda, pad_elmo
import argparse
from models.no_context import BILSTMsim
from timeit import default_timer as timer


def convert_document(document):
    document.answers = []
    for query in document.queries:
        query.answer1 = len(document.answers)
        document.answers.append(query.answer1_tokens)
        query.answer2 = len(document.answers)
        document.answers.append(query.answer2_tokens)

# def train(elmo, model):


def main(args):

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    with open(args.train_path, "rb") as file:
        train_summaries = pickle.load(file, encoding='utf-8')
    
    with open(args.valid_path, "rb") as file:
        valid_summaries = pickle.load(file, encoding='utf-8')

    for summary in train_summaries:
        convert_document(summary)
    
    for summary in valid_summaries:
        convert_document(summary)

    elmo_instance = Elmo(options_url, weights_url, 1)
    if use_cuda:
        elmo_instance.cuda()

    begin=timer()
    total_answers=0
    for summary in train_summaries[:10]:
        answers=[[elmo_tokenize(word) for word in answer] for answer in summary.answers]
        answers=pad_elmo(answers)
        batch=Variable(torch.LongTensor(answers))
        a=elmo_instance(batch)
        total_answers+=len(answers)

    end=timer()
    print("Total time elapsed: {}".format(end-begin))
    print("Time per thousand answers: {}".format((end-begin)*1000/total_answers))


    # model = BILSTMsim()


    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str,
                        default="/home/michiel/Dropbox/CMU/747/project/narrativeqa-master/pickle/train_summaries.pickle")
    parser.add_argument("--valid_path", type=str,
                        default="/home/michiel/Dropbox/CMU/747/project/narrativeqa-master/pickle/valid_summaries.pickle")
    parser.add_argument("--test_path", type=str,
                        default="/home/michiel/Dropbox/CMU/747/project/narrativeqa-master/pickle/test_summaries.pickle")

    parser.add_argument("--seed", type=int,
                        default=0)

    args = parser.parse_args()

    main(args)

    
    # DL = DataLoader()

    # print("Loading training data")
    # training_data = DL.load_data(args.train_path, args.max_train)
    # print("Loading validation data")
    # valid_data = DL.load_data(args.valid_path, args.max_valid)
    # print("Loading test data")
    # test_data = DL.load_data(args.test_path, args.max_test)

    # model = ASReader(DL.data_vocab.get_length(), args.embedding_dim, args.encoding_dim)
    # if USE_CUDA:
    #     model.cuda()

    # print("Starting training")
    # train(model, training_data, valid_data, test_data, DL, batch_size=args.batch_size, bucket_size=args.bucket_size,
    #       learning_rate=args.learning_rate, eval_interval=args.eval_interval, num_epochs=args.num_epochs, model_path=args.model_path)


# Summaries loaded in final form above here somehow
####################


# def train():


# create batches
# for each epoch: shuffle
# run q, a through elmo
# run q, a through model
# predict
# gradient step
#  evaluate


# print(vars(summaries[0]))
# def create_batches(data)


# load data
# extract questions and answers
# create embedding
# create model
# train model
# evaluate model
