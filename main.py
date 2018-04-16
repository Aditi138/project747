import argparse
import sys

from dataloaders.dataloader import DataLoader, create_batches
from models.model import *

torch.manual_seed(2)
from torch import optim
from dataloaders.utility import *
import numpy as np

def evaluate(batches):
    global model
    global best_rouge
    global best_bleu1
    global best_bleu4

    mrr_value = []
    for iteration in range(len(batches)):

        batch = batches[iteration]

        batch_query_lengths = batch['qlengths']
        batch_candidates = batch["candidates"]
        batch_answer_indices = batch['answer_indices']
        batch_size = len(batch_query_lengths)

        for index,query in enumerate(batch['queries']):

            # query tokens
            batch_query = variable(torch.LongTensor(query))
            batch_query_length = [batch['qlengths'][index]]

            #Sort the candidates by length
            batch_candidate_lengths = np.array(batch_candidates["anslengths"][index])
            candidate_sort = np.argsort(batch_candidate_lengths)[::-1].copy()
            batch_candidates_sorted = variable(torch.LongTensor(batch_candidates["answers"][index][candidate_sort,...]))
            batch_candidate_lengths_sorted = batch_candidate_lengths[candidate_sort]

            batch_len = len(batch_candidate_lengths_sorted)
            batch_candidate_unsort = variable(torch.LongTensor(np.argsort(candidate_sort)))
            batch_metrics =  variable(torch.FloatTensor(batch['metrics'][index]))
            indices = model.eval(batch_query,batch_query_length,
                             batch_candidates_sorted, batch_candidate_lengths_sorted ,batch_candidate_unsort, batch_answer_indices[index],
                                                         batch_metrics,batch_len)

            if args.use_cuda:
                indices = indices.data.cpu()

            else:
                indices = indices.data

            position_gold_sorted = (indices ==  batch_answer_indices[index]).nonzero().numpy()[0][0]

            index= position_gold_sorted + 1

            mrr_value.append(1.0 / (index))


    mean_rr = np.mean(mrr_value)
    print("MRR :{0}".format(mean_rr))


def train_epochs(train_batches):
    global model
    global valid_batches
    global test_batches
    clip_threshold = args.clip_threshold
    eval_interval = args.eval_interval

    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    train_loss = 0
    train_denom = 0
    validation_history = []
    bad_counter = 0
    patience = 30


    for epoch in range(args.num_epochs):

        print("Starting epoch {}".format(epoch))

        saved = False
        for iteration in range(len(train_batches)):

            if (iteration + 1) % eval_interval == 0:
                print("iteration {}".format(iteration + 1))
                print("train loss: {}".format(train_loss / train_denom))

                if iteration != 0:
                    average_bleu = evaluate(valid_batches)
                    validation_history.append(average_bleu)

                    if (iteration + 1) % (eval_interval * 5) == 0:
                        if average_bleu >= max(validation_history):
                            saved = True
                            print("Saving best model seen so far at epoch number {0}".format(iteration))
                            torch.save(model, args.model_path)
                            print("Best on Validation: BLEU_1:{0} BLEU_4:{1} ROUGE_L:{2}".format(best_bleu1, best_bleu4,
                                                                                                 best_rouge))
                            bad_counter = 0
                        else:
                            bad_counter += 1
                        if bad_counter > patience:
                            print("Early Stopping")
                            print("Testing started")
                            evaluate(test_batches)
                            exit(0)
            batch = train_batches[iteration]

            batch_query_lengths = batch['qlengths']
            batch_candidates = batch["candidates"]
            batch_answer_indices = batch['answer_indices']
            batch_size = len(batch_query_lengths)

            for index,query in enumerate(batch['queries']):
                optimizer.zero_grad()
                # query tokens
                batch_query = variable(torch.LongTensor(query))
                batch_query_length = [batch['qlengths'][index]]

                #Sort the candidates by length
                batch_candidate_lengths = np.array(batch_candidates["anslengths"][index])
                candidate_sort = np.argsort(batch_candidate_lengths)[::-1].copy()
                batch_candidates_sorted = variable(torch.LongTensor(batch_candidates["answers"][index][candidate_sort,...]))
                batch_candidate_lengths_sorted = batch_candidate_lengths[candidate_sort]

                batch_len = len(batch_candidate_lengths_sorted)
                batch_candidate_unsort = variable(torch.LongTensor(np.argsort(candidate_sort)))
                batch_metrics =  variable(torch.FloatTensor(batch['metrics'][index]))
                loss,second_best = model(batch_query,batch_query_length,
                             batch_candidates_sorted, batch_candidate_lengths_sorted ,batch_candidate_unsort, batch_answer_indices[index],
                                                         batch_metrics,batch_len)


                if args.use_cuda:
                    train_loss += loss.data.cpu().numpy()[0][0][0]
                    second_best = second_best.data.cpu().numpy()[0]

                else:
                    train_loss += loss.data.numpy()[0][0][0]
                    second_best = second_best.data.numpy()[0]


                loss.backward()

                #gold_answer= " ".join([loader.vocab.get_word(id) for id in batch_candidates["answers"][index][batch_answer_indices[index]]])
                #second_best_answer = " ".join([loader.vocab.get_word(id) for id in batch_candidates["answers"][index][second_best]] )
                #print("Gold: {0} Model Selected: {1}".format(gold_answer.replace("pad",""),second_best_answer.replace("pad","")))

                torch.nn.utils.clip_grad_norm(model.parameters(), clip_threshold)
                optimizer.step()


                ## Todo: evaluate current train batch
            train_denom += batch_size

        if not saved:
            print("Saving model after epoch {0}".format(epoch))
            torch.save(model, args.model_path + ".dummy")

    print("All epochs done")




if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf8')
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="/../narrativeqa/out/summary/train.pickle")
    parser.add_argument("--valid_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--job_size", type=int, default=5)
    parser.add_argument("--pretrain_path", type=str, default=None, help="Path to the pre-trained word embeddings")

    #Model parameters
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--embed_size", type=int, default=100)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--batch_length", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--clip_threshold", type=int, default=10)

    parser.add_argument("--meteor_path", type=str, default=10)

    args = parser.parse_args()

    if args.cuda and torch.cuda.is_available():
        vars(args)['use_cuda'] = True
    else:
        vars(args)['use_cuda'] = False

    loader = DataLoader(args)

    train_documents = loader.load_documents( args.train_path, summary_path=None)
    valid_documents = loader.load_documents(args.valid_path, summary_path=None)
    test_documents = loader.load_documents(args.test_path, summary_path=None)

    loader.create_id_to_vocabulary()

    train_batches = create_batches(train_documents,args.batch_length, args.job_size)
    valid_batches = create_batches(valid_documents,args.batch_length,args.job_size)
    test_batches = create_batches(test_documents,args.batch_length,args.job_size)

    model = Model(args, loader)
    if args.use_cuda:
        model = model.cuda()
    train_epochs(train_batches)

