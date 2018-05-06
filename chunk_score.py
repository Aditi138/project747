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


def evaluate(model, batches, articles, args):
    model.train(False)
    accuracy=0
    for iteration in range(len(batches)):
        batch=batches[iteration]
        article_id=batch["article"]
        chunks= [pad_seq(articles[article_id][idd], batch["max_length"]) for idd in batch["paragraphs"]]
        chunks=variable(torch.LongTensor(chunks))
        question = variable(torch.LongTensor(np.array(batch["question"]))).unsqueeze(0)
        gold_index = batch["paragraphs"].index(batch["gold"])
        gold_variable = variable(torch.LongTensor([gold_index]))
        mask = variable(torch.FloatTensor(batch["mask"]))
        loss, scores = model(chunks, question, gold_variable, mask)
        if np.argmax(scores) == gold_index:
            accuracy+=(1.0/len(batches))
    model.train(True)
    return accuracy

def view_data(batch, articles):
    article_id=batch["article"]
    print(" ".join([loader.vocab.get_word(word_id) for word_id in batch["question"]]))
    # print(" ".join([loader.vocab.get_word(word_id) for word_id in articles[article_id][batch["gold"]]]))
    for paragraph_id in batch["paragraphs"]:
        print(" ".join([loader.vocab.get_word(word_id) for word_id in articles[article_id][paragraph_id]]))
    

def create_batches(data, articles):

    batches=[]    
    random.shuffle(data)
    for point in data:
        batch={}
        batch["question"] = point.question_tokens
        if len(point.question_tokens) <3:
            continue
        article_id=point.article_id
        batch["article"] = article_id
        paragraphs = set()
        paragraphs.add(point.gold_paragraph_id)
        sample_paragraphs=[i for i in range(len(articles[article_id])) if i!=point.gold_paragraph_id]
        paragraphs = paragraphs | set(np.random.choice(sample_paragraphs, size=min(args.competing_paragraphs, len(articles[article_id])), replace=False))
        batch["paragraphs"] = list(paragraphs)
        # batch["paragraphs"] = [i for i in range(len(articles[article_id]))]
        article_lengths=[len(articles[article_id][paragraph_id]) for paragraph_id in batch["paragraphs"]]
        batch["max_length"] = np.max(article_lengths)
        batch["mask"] =  np.array([[int(i < article_lengths[j]) for i in range(batch["max_length"])] for j in range(len(paragraphs))])
        batch["gold"] = point.gold_paragraph_id
        batches.append(batch)

    return batches

def pad_batch(chunks, max_length):
    padded_batch=[pad_seq(chunk, max_length) for chunk in chunks]
    return padded_batch


def train_epochs(model, train_data,train_articles,valid_data,valid_articles, args):

    clip_threshold = args.clip_threshold
    eval_interval = args.eval_interval

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    train_loss = 0
    train_accuracy = 0

    valid_batches = create_batches(valid_data, valid_articles)

    for epoch in range(args.num_epochs):
        train_batches = create_batches(train_data, train_articles)

        for i in range(len(train_batches)):
            optimizer.zero_grad()
            batch = train_batches[i]
            article_id=batch["article"]
            chunks= [pad_seq(train_articles[article_id][idd], batch["max_length"]) for idd in batch["paragraphs"]]
            chunks=variable(torch.LongTensor(chunks))
            question = variable(torch.LongTensor(np.array(batch["question"]))).unsqueeze(0)
            gold_index = batch["paragraphs"].index(batch["gold"])
            gold_variable = variable(torch.LongTensor([gold_index]))
            mask = variable(torch.FloatTensor(batch["mask"]))
            loss, scores = model(chunks, question, gold_variable, mask)
            if np.argmax(scores) == gold_index:
                train_accuracy+=(1.0/eval_interval)
            train_loss += loss.data.cpu().numpy()[0]/eval_interval
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), clip_threshold)
            optimizer.step()


            if i % eval_interval == 0:
                print("Iteration: {}".format(i))
                print("Training loss: {}".format(train_loss))
                print("Training accuracy: {}".format(train_accuracy))
                valid_accuracy=evaluate(model, valid_batches, valid_articles, args)
                print("Validation accuracy: {}".format(valid_accuracy))

                train_loss=0
                train_accuracy=0

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
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--batch_length", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--clip_threshold", type=int, default=10)

    parser.add_argument("--meteor_path", type=str, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--reduced", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(2)

    if args.cuda and torch.cuda.is_available():
        vars(args)['use_cuda'] = True
    else:
        vars(args)['use_cuda'] = False


    start = time()
    loader = SquadDataloader(args)
    train_data, train_articles = loader.load_documents_with_paragraphs(args.train_path + "questions.pickle", args.train_path + "paragraphs.pickle", max_documents= args.max_documents)
    valid_data, valid_articles = loader.load_documents_with_paragraphs(args.valid_path + "questions.pickle", args.valid_path + "paragraphs.pickle", max_documents=args.max_valid)
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

    train_epochs(model, train_data,train_articles,valid_data,valid_articles, args)
