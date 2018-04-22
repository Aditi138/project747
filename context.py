import argparse
import sys

from dataloaders.dataloader import DataLoader, create_batches, view_batch
from dataloaders.squad_dataloader import SquadDataloader
from models.context_model import ContextMRR

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
	mrr_value = []
	model.train(False)
	for iteration in range(len(batches)):

		batch = batches[iteration]
		batch_candidates = batch["candidates"]
		batch_answer_indices = batch['answer_indices']

		for index, query in enumerate(batch['queries']):

			# query tokens
			batch_query = variable(torch.LongTensor(query), volatile=True)
			batch_query_length = [batch['qlengths'][index]]
			batch_question_mask = variable(torch.FloatTensor(batch['q_mask'][index]))
			# batch_query_ner = variable(torch.LongTensor(batch['q_ner'][index]))
			# batch_query_pos = variable(torch.LongTensor(batch['q_pos'][index]))

			# Sort the candidates by length
			batch_candidate_lengths = np.array(batch_candidates["anslengths"][index])
			batch_candidate_mask = np.array(batch_candidates['mask'][index])
			candidate_sort = np.argsort(batch_candidate_lengths)[::-1].copy()
			batch_candidates_sorted = variable(
				torch.LongTensor(batch_candidates["answers"][index][candidate_sort, ...]), volatile=True)
			batch_candidate_lengths_sorted = batch_candidate_lengths[candidate_sort]
			batch_candidate_masks_sorted = variable(torch.FloatTensor(batch_candidate_mask[candidate_sort]))
			# batch_candidate_ner_sorted = variable(torch.LongTensor(batch_candidates['ner'][index][candidate_sort, ...]))
			# batch_candidate_pos_sorted = variable(
			# 	torch.LongTensor(batch_candidates['pos'][index][candidate_sort, ...]))

			# context tokens
			batch_context = variable(torch.LongTensor(batch['contexts'][index]))
			batch_context_length = np.array([batch['clengths'][index]])
			batch_context_mask = variable(torch.FloatTensor(batch['context_mask'][index]))


			batch_len = len(batch_candidate_lengths_sorted)
			batch_candidate_unsort = variable(torch.LongTensor(np.argsort(candidate_sort)), volatile=True)

			indices = model.eval(batch_query, batch_query_length,batch_question_mask,
								 batch_context, batch_context_length,batch_context_mask,
								 batch_candidates_sorted, batch_candidate_lengths_sorted, batch_candidate_masks_sorted,batch_candidate_unsort)

			if args.use_cuda:
				indices = indices.data.cpu()

			else:
				indices = indices.data

			position_gold_sorted = (indices == batch_answer_indices[index]).nonzero().numpy()[0][0]

			index = position_gold_sorted + 1

			mrr_value.append(1.0 / (index))

	mean_rr = np.mean(mrr_value)
	print("MRR :{0}".format(mean_rr))
	model.train(True)
	return mean_rr


def train_epochs(model, vocab):
	clip_threshold = args.clip_threshold
	eval_interval = args.eval_interval

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	train_loss = 0
	train_denom = 0
	validation_history = []
	bad_counter = 0
	best_mrr = -1.0

	patience = 30

	valid_batches = create_batches(valid_documents, args.batch_length, args.job_size, vocab)[:200]
	test_batches = create_batches(test_documents,args.batch_length,args.job_size, vocab)

	for epoch in range(args.num_epochs):

		print("Creating train batches")
		train_batches = create_batches(train_documents, args.batch_length, args.job_size, vocab)
		print("Starting epoch {}".format(epoch))

		saved = False
		for iteration in range(len(train_batches)):
			optimizer.zero_grad()
			if (iteration + 1) % eval_interval == 0:
				print("iteration {}".format(iteration + 1))
				print("train loss: {}".format(train_loss / train_denom))

				if iteration != 0:
					average_rr = evaluate(model, valid_batches)
					validation_history.append(average_rr)


					if (iteration + 1) % (eval_interval * 5) == 0:
						print("Validation MRR:{0}".format(average_rr))
						if average_rr >= max(validation_history):
							saved = True
							print("Saving best model seen so far itr number {0}".format(iteration))
							torch.save(model, args.model_path)
							print("Best on Validation: MRR:{0}".format(average_rr))
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
			batch_candidates = batch["candidates"]
			batch_answer_indices = batch['answer_indices']
			batch_size = len(batch_query_lengths)
			losses = variable(torch.zeros(batch_size))
			for index, query in enumerate(batch['queries']):
				# query tokens
				batch_query = variable(torch.LongTensor(query))
				batch_query_length = np.array([batch['qlengths'][index]])
				batch_question_mask = variable(torch.FloatTensor(batch['q_mask'][index]))

				# batch_query_ner = variable(torch.LongTensor(batch['q_ner'][index]))
				# batch_query_pos = variable(torch.LongTensor(batch['q_pos'][index]))

				# Sort the candidates by length (only required if using an RNN)
				batch_candidate_lengths = np.array(batch_candidates["anslengths"][index])
				batch_candidate_mask = np.array(batch_candidates['mask'][index])
				candidate_sort = np.argsort(batch_candidate_lengths)[::-1].copy()

				batch_candidates_sorted = variable(
					torch.LongTensor(batch_candidates["answers"][index][candidate_sort, ...]))
				batch_candidate_lengths_sorted = batch_candidate_lengths[candidate_sort]
				batch_candidate_unsort = variable(torch.LongTensor(np.argsort(candidate_sort)))
				batch_candidate_masks_sorted = variable(torch.FloatTensor(batch_candidate_mask[candidate_sort]))

				# batch_candidate_ner_sorted = variable(
				# 	torch.LongTensor(batch_candidates['ner'][index][candidate_sort, ...]))
				# batch_candidate_pos_sorted = variable(
				# 	torch.LongTensor(batch_candidates['pos'][index][candidate_sort, ...]))


				batch_len = len(batch_candidate_lengths)
				batch_metrics = variable(torch.FloatTensor(batch['metrics'][index]))

				# context tokens
				batch_context = variable(torch.LongTensor(batch['contexts'][index]))
				batch_context_length = np.array([batch['clengths'][index]])
				batch_context_mask =variable(torch.FloatTensor(batch['context_mask'][index]))

				gold_index = variable(torch.LongTensor([batch_answer_indices[index]]))
				negative_indices = [idx for idx in range(batch_len)]
				negative_indices.pop(batch_answer_indices[index])
				negative_indices = variable(torch.LongTensor(negative_indices))



				loss, second_best = model(batch_query, batch_query_length,batch_question_mask,
									batch_context, batch_context_length, batch_context_mask,
									batch_candidates_sorted, batch_candidate_lengths_sorted, batch_candidate_masks_sorted,
										  batch_candidate_unsort,
									gold_index, negative_indices, batch_metrics
									)

				losses[index] = loss

			# loss.backward()
			# torch.nn.utils.clip_grad_norm(model.parameters(), clip_threshold)
			# optimizer.step()

			mean_loss = losses.mean(0)
			mean_loss.backward()
			optimizer.step()
			if args.use_cuda:
				train_loss += mean_loss.data.cpu().numpy()[0] * batch_size

			else:
				train_loss += mean_loss.data.numpy()[0] * batch_size

			train_denom += batch_size


		if not saved:
			print("Saving model after epoch {0}".format(epoch))
			torch.save(model, args.model_path + ".dummy")

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
	parser.add_argument("--hidden_size", type=int, default=128)
	parser.add_argument("--embed_size", type=int, default=128)
	parser.add_argument("--cuda", action="store_true", default=True)
	parser.add_argument("--batch_length", type=int, default=1)
	parser.add_argument("--eval_interval", type=int, default=2)
	parser.add_argument("--learning_rate", type=float, default=0.0001)
	parser.add_argument("--num_epochs", type=int, default=10)
	parser.add_argument("--clip_threshold", type=int, default=10)
	parser.add_argument("--num_layers", type=int, default=3)
	parser.add_argument("--ner_dim", type=int, default=32)
	parser.add_argument("--pos_dim", type=int, default=32)

	parser.add_argument("--meteor_path", type=str, default=10)
	parser.add_argument("--profile", action="store_true")
	parser.add_argument("--squad", action="store_true")

	args = parser.parse_args()

	torch.manual_seed(2)

	if args.cuda and torch.cuda.is_available():
		vars(args)['use_cuda'] = True
	else:
		vars(args)['use_cuda'] = False


	start = time()
	if args.squad:
		loader = SquadDataloader(args)
		train_documents = loader.load_documents_with_candidates(args.train_path)
		valid_documents = loader.load_documents_with_candidates(args.valid_path)
		test_documents = loader.load_documents_with_candidates(args.valid_path)
	else:
		loader = DataLoader(args)
		train_documents = loader.load_documents(args.train_path, summary_path=args.summary_path, max_documents=args.max_documents)
		valid_documents = loader.load_documents(args.valid_path, summary_path=None, max_documents=args.max_documents)
		test_documents = loader.load_documents(args.test_path, summary_path=None, max_documents=args.max_documents)

	end = time()
	print(end - start)


	end = time()
	print(end - start)

	model = ContextMRR(args, loader.vocab)

	if args.use_cuda:
		model = model.cuda()

	
	train_epochs(model, loader.vocab)