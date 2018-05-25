import argparse
import sys
from dataloaders.squad_dataloader import SquadDataloader
from models.score_model import ChunkScore
from dataloaders.utility import get_pretrained_emb
import torch
from torch import optim
from dataloaders.utility import variable, view_data_point, pad_seq
import numpy as np
from time import time
import random
import pickle
import codecs
from collections import defaultdict
import random




def evaluate(model, batches):
    model.train(False)
    correct = 0
    denom = 0
    for i, batch in enumerate(batches):
        questions = variable(torch.LongTensor(batch[0]))
        chunks = variable(torch.LongTensor(batch[1]))
        gold = variable(torch.FloatTensor(batch[2]))
        loss, prediction = model(chunks, questions, gold.unsqueeze(0))

        prediction = [1 if p > 0.5 else 0 for p in prediction[0]]

        #random
        #prediction = [random.random() for _ in range(len(batch[0]))]
        #prediction = [1 if p > 0.5 else 0 for p in prediction]
        correct += sum([1 if y1 == y2 else 0 for y1, y2 in zip(prediction, batch[2])])
        denom += len(batch[0])

    accuracy = correct * 1.0 / denom
    model.train(True)
    return accuracy

def view_data(batch, articles):
    article_id=batch["article"]
    print(" ".join([loader.vocab.get_word(word_id) for word_id in batch["question"]]))
    # print(" ".join([loader.vocab.get_word(word_id) for word_id in articles[article_id][batch["gold"]]]))
    for paragraph_id in batch["paragraphs"]:
        print(" ".join([loader.vocab.get_word(word_id) for word_id in articles[article_id][paragraph_id]]))
    


def make_bucket_batches(data, batch_size):
    # Data are bucketed according to the length of the first item in the data_collections.
    # Data are bucketed according to the length of the first item in the data_collections.
    buckets = defaultdict(list)
    tot_items = len(data[0])
    for data_item in data:
        src = data_item[0]
        buckets[len(src)].append(data_item)

    batches = []
    # np.random.seed(2)
    for src_len in buckets:
        bucket = buckets[src_len]
        np.random.shuffle(bucket)

        num_batches = int(np.ceil(len(bucket) * 1.0 / batch_size))
        for i in range(num_batches):
            cur_batch_size = batch_size if i < num_batches - 1 else len(bucket) - batch_size * i
            batches.append([[bucket[i * batch_size + j][k] for j in range(cur_batch_size)] for k in range(tot_items)])
    np.random.shuffle(batches)
    return batches


def train_epochs(model,  train_questions,train_sentences,train_gold, valid_questions, valid_sentences, valid_gold ,args):

    clip_threshold = args.clip_threshold
    eval_interval = args.eval_interval

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loss = 0
    denom = 0
    correct = 0

    valid_batches = make_bucket_batches(zip(valid_questions, valid_sentences, valid_gold ), args.batch_length)

    for epoch in range(args.num_epochs):
        train_batches = make_bucket_batches(zip(train_questions,train_sentences,train_gold),args.batch_length)

        for i, batch in enumerate(train_batches):
            optimizer.zero_grad()
            questions = variable(torch.LongTensor(batch[0]))
            chunks = variable(torch.LongTensor(batch[1]))
            gold = variable(torch.FloatTensor(batch[2]))
            loss, prediction = model(chunks, questions, gold.unsqueeze(0))

            prediction = [1 if p > 0.5 else 0 for p in prediction[0]]
            correct += sum([1 if y1 == y2 else 0 for y1, y2 in zip(prediction, batch[2])])
            denom += len(batch[0])
            train_loss += loss.data.cpu().numpy()[0] * len(batch[0])

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), clip_threshold)
            optimizer.step()


            if i % eval_interval == 0:
                print("Iteration: {} Training loss: {} Training accuracy: {}".format(i,train_loss * 1.0 / denom, correct * 1.0 / denom))
                valid_accuracy=evaluate(model, valid_batches)
                print("Validation accuracy: {}".format(valid_accuracy))



    print("All epochs done")
    

if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf8')
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="/../narrativeqa/summaries/small_summaries.pickle")
    parser.add_argument("--valid_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--job_size", type=int, default=5)
    parser.add_argument("--pretrain_path", type=str, default=None, help="Path to the pre-trained word embeddings")
    parser.add_argument("--max_documents", type=int, default=0, help="If greater than 0, load at most this many documents")
    parser.add_argument("--max_valid", type=int, default=0, help="If greater than 0, load at most this many documents")



    # Model parameters
    parser.add_argument("--competing_paragraphs", type=int, default=1, help="number of paragraphs to consider in parallel to the gold paragraph")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--embed_size", type=int, default=64)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--batch_length", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--clip_threshold", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--meteor_path", type=str, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--reduced", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.cuda and torch.cuda.is_available():
        vars(args)['use_cuda'] = True
    else:
        vars(args)['use_cuda'] = False


    start = time()
    loader = SquadDataloader(args)
    train_questions, train_sentences, train_gold = loader.load_documents_with_sentences(args.train_path, max_documents= args.max_documents)
    valid_questions, valid_sentences, valid_gold = loader.load_documents_with_sentences(args.valid_path, max_documents=args.max_valid)
    end = time()

    print("Time loading data: {}s".format(end - start))

    # print(len(train_articles))
    # print(np.max([len(train_articles[id]) for id in train_articles]))


    # Get pre_trained embeddings
    loader.pretrain_embedding = None
    if args.pretrain_path is not None:
        word_embedding = get_pretrained_emb(args.pretrain_path, loader.vocab.vocabulary, args.embed_size)
        loader.pretrain_embedding = word_embedding

    model = ChunkScore(args, loader.vocab.get_length(), loader.pretrain_embedding)

    if args.use_cuda:
        model = model.cuda()

    train_epochs(model, train_questions,train_sentences,train_gold,valid_questions, valid_sentences, valid_gold ,args)