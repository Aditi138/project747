import argparse
import sys

from dataloaders.dataloader import create_batches, view_batch, make_bucket_batches
from dataloaders.squad_dataloader import SquadDataloader
from models.span_prediction_model import ContextMRR

import torch
from torch import optim
from dataloaders.utility import variable, view_data_point
import numpy as np
from time import time
import random
import cProfile


def get_random_batch_from_training(batches, num):
	small = []
	for i in range(num):
		index = random.randint(0, len(batches)-1)
		small.append(batches[index])
	return small


def evaluate(model, batches):
	all_start_correct = 0.0
	all_end_correct = 0.0
	all_span_correct = 0.0
	count = 0.0

	model.train(False)

	for iteration in range(len(batches)):

		batch = batches[iteration]
		# view_batch(batch,loader.vocab)
		batch_query_lengths = batch['qlengths']

		batch_start_indices = variable(torch.LongTensor(batch['start_indices']))
		batch_end_indices = variable(torch.LongTensor(batch['end_indices']))

		batch_size = len(batch_query_lengths)

		batch_query = variable(torch.LongTensor(batch['queries']))
		batch_query_length = np.array([batch['qlengths']])
		batch_question_mask = variable(torch.FloatTensor(batch['q_mask']))

		# context tokens
		batch_context = variable(torch.LongTensor(batch['contexts']))
		batch_context_length = np.array([batch['clengths']])
		batch_context_mask = variable(torch.FloatTensor(batch['context_mask']))

		start_correct, end_correct, span_correct = model.eval(batch_query, batch_query_length, batch_question_mask,
															   batch_context, batch_context_length, batch_context_mask,
															   batch_start_indices, batch_end_indices)

		all_start_correct += start_correct
		all_end_correct += end_correct
		all_span_correct += span_correct

	all_start_correct = (all_start_correct * 1.0)/ count
	all_end_correct = (all_end_correct * 1.0) / count
	all_span_correct = (all_span_correct * 1.0) / count
	return all_start_correct, all_end_correct, all_span_correct

def train_epochs(model, vocab):
	clip_threshold = args.clip_threshold
	eval_interval = args.eval_interval

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9,0.9))
	train_loss = 0
	train_denom = 0
	validation_history = []
	bad_counter = 0
	best_mrr = -1.0
	all_start_correct = 0.0
	all_end_correct = 0.0
	all_span_correct  = 0.0

	patience = 10

	valid_batches = make_bucket_batches(valid_documents, args.batch_length, vocab)[:200]


	for epoch in range(args.num_epochs):

		print("Creating train batches")
		train_batches = make_bucket_batches(train_documents, args.batch_length, vocab)
		print("Starting epoch {}".format(epoch))

		saved = False
		for iteration in range(len(train_batches)):
			optimizer.zero_grad()
			if (iteration + 1) % eval_interval == 0:
				print("iteration {}".format(iteration + 1))
				print("train loss: {}".format(train_loss / train_denom))

				if iteration != 0:
					start, end, span = evaluate(model, valid_batches)
					validation_history.append(span)

					if (iteration + 1) % (eval_interval * 5) == 0:
						if span >= max(validation_history):
							saved = True
							print("Saving best model seen so far itr number {0}".format(iteration))
							torch.save(model, args.model_path)
							print("Best on Validation: Start:{0} End:{1} Span:{2}".format(start, end, span))
							bad_counter = 0
						else:
							bad_counter += 1
						if bad_counter > patience:
							print("Early Stopping")
							print("Testing started")
							evaluate(model, valid_batches)
							exit(0)

			batch = train_batches[iteration]
			# view_batch(batch,loader.vocab)
			batch_query_lengths = batch['qlengths']

			batch_start_indices = variable(torch.LongTensor(batch['start_indices']))
			batch_end_indices = variable(torch.LongTensor(batch['end_indices']))
			
			batch_size = len(batch_query_lengths)


			batch_query = variable(torch.LongTensor(batch['queries']))
			batch_query_length = np.array([batch['qlengths']])
			batch_question_mask = variable(torch.FloatTensor(batch['q_mask']))

			# context tokens
			batch_context = variable(torch.LongTensor(batch['contexts']))
			batch_context_length = np.array([batch['clengths']])
			batch_context_mask =variable(torch.FloatTensor(batch['context_mask']))
			
			loss, start_correct, end_correct, span_correct = model(batch_query, batch_query_length,batch_question_mask,
				      batch_context, batch_context_length, batch_context_mask,
						 batch_start_indices, batch_end_indices)

			all_start_correct += start_correct
			all_end_correct += end_correct
			all_span_correct += span_correct
			
			loss = loss / batch_size
			loss.backward()
			optimizer.step()
			
			if args.use_cuda:
				train_loss += loss.data.cpu().numpy()[0] * batch_size

			else:
				train_loss += loss.data.numpy()[0] * batch_size

			train_denom += batch_size

		if not saved:
			print("Saving model after epoch {0}".format(epoch))
			torch.save(model, args.model_path + ".dummy")

		print("Train Accuracy: start_index = {0}, end_index = {1}, span = {2}".format(all_start_correct*1.0 / train_denom, all_end_correct*1.0 / train_denom, all_span_correct*1.0 / train_denom))

	print("All epochs done")


if __name__ == "__main__":
	reload(sys)
	sys.setdefaultencoding('utf8')
	parser = argparse.ArgumentParser()

	parser.add_argument("--train_path", type=str, default="/../narrativeqa/summaries/small_summaries.pickle")
	parser.add_argument("--valid_path", type=str, default=None)
	parser.add_argument("--test_path", type=str, default=None)
	parser.add_argument("--summary_path", type=str, default=None)
	parser.add_argument("--model_path", type=str, default=None)
	parser.add_argument("--job_size", type=int, default=5)
	parser.add_argument("--pretrain_path", type=str, default=None, help="Path to the pre-trained word embeddings")
	parser.add_argument("--max_documents", type=int, default=0, help="If greater than 0, load at most this many documents")

	# Model parameters
	parser.add_argument("--hidden_size", type=int, default=100)
	parser.add_argument("--embed_size", type=int, default=100)
	parser.add_argument("--cuda", action="store_true", default=True)
	parser.add_argument("--batch_length", type=int, default=40)
	parser.add_argument("--eval_interval", type=int, default=2)
	parser.add_argument("--learning_rate", type=float, default=0.0001)
	parser.add_argument("--num_epochs", type=int, default=20)
	parser.add_argument("--clip_threshold", type=int, default=5)
	parser.add_argument("--num_layers", type=int, default=3)
	parser.add_argument("--ner_dim", type=int, default=32)
	parser.add_argument("--pos_dim", type=int, default=32)
	parser.add_argument("--dropout", type=float, default=0.2)


	parser.add_argument("--meteor_path", type=str, default=10)
	parser.add_argument("--profile", action="store_true")

	args = parser.parse_args()

	torch.manual_seed(2)

	if args.cuda and torch.cuda.is_available():
		vars(args)['use_cuda'] = True
	else:
		vars(args)['use_cuda'] = False

	loader = SquadDataloader(args)

	start = time()
	train_documents = loader.load_docuements(args.train_path, summary_path=args.summary_path, max_documents=args.max_documents)
	valid_documents = loader.load_docuements(args.valid_path, summary_path=None, max_documents=args.max_documents)


	end = time()
	print(end - start)

	model = ContextMRR(args, loader.vocab)

	if args.use_cuda:
		model = model.cuda()

	
	train_epochs(model, loader.vocab)