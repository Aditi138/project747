import argparse
import sys

from dataloaders.dataloader import create_batches, view_batch, make_bucket_batches, DataLoader
from dataloaders.squad_dataloader import SquadDataloader
from models.multi_paragraph_model import MultiParagraph
import torch
from torch import optim
from dataloaders.utility import variable, view_data_point,view_span_data_point,get_pretrained_emb, pad_seq_elmo
import numpy as np
from time import time
import random
import cProfile
import pickle


class Document_All_Embed(object):
    def __init__(self,id, qaps,candidates_embed, candidates, document_tokens, document_embed):
        self.id = id
        self.qaps = qaps
        self.candidates_embed = candidates_embed
        self.candidates = candidates
        self.document_tokens = document_tokens
        self.document_embed = document_embed

class Query_Embed(object):
    def __init__(self, question_tokens, answer_indices, query_embed=None):
        self.question_tokens = question_tokens
        self.answer_indices = answer_indices
        self.query_embed = query_embed

def test_model(model, documents, vocab, context_per_docid):
    test_batches = make_bucket_batches(documents, args.batch_length, vocab)
    print("Testing!")
    start,end,span = evaluate(model, test_batches,te_context_per_docid)
    print("Testing Accuracy: Start:{0} End:{1} Span:{2}".format(start, end, span))


def get_random_batch_from_training(batches, num):
	small = []
	for i in range(num):
		index = random.randint(0, len(batches)-1)
		small.append(batches[index])
	return small


def evaluate(model, batches,context_per_docid):
	all_start_correct = 0.0
	all_end_correct = 0.0
	all_span_correct = 0.0
	count = 0.0

	model.train(False)

	model._span_start_accuracy_valid = Accuracy()
	model._span_end_accuracy_valid = Accuracy()
	model._span_accuracy_valid = BooleanAccuracy()

	for iteration in range(len(batches)):

		batch = batches[iteration]
		# view_batch(batch,loader.vocab)
		batch_query_lengths = batch['qlengths']

		batch_start_indices = variable(torch.LongTensor(batch['start_indices']))
		batch_end_indices = variable(torch.LongTensor(batch['end_indices']))

		batch_size = len(batch_query_lengths)

		#batch_query = variable(torch.LongTensor(batch['queries']))
		batch_query_embed = variable(torch.FloatTensor(batch['q_embed']))
		batch_query_length = np.array([batch['qlengths']])
		batch_question_mask = variable(torch.FloatTensor(batch['q_mask']))

		# context tokens
		#batch_context = variable(torch.LongTensor(batch['contexts']))
		batch_context_length = np.array([batch['clengths']])
		batch_context_mask = variable(torch.FloatTensor(batch['context_mask']))

		doc_ids = batch['doc_ids']
		batch_context_embed = []
		max_batch_context = len(batch['context_mask'][0])

		for doc_id in doc_ids:
			batch_context_embed.append(context_per_docid[doc_id])

		batch_context_embed_padded = np.array(
			[pad_seq_elmo(context_embed, max_batch_context) for context_embed in batch_context_embed])

		batch_context_embed_padded = variable(torch.FloatTensor(batch_context_embed_padded))

		start_correct, end_correct, span_correct = model.eval(batch_query_embed, batch_query_length, batch_question_mask,
															  batch_context_embed_padded, batch_context_length, batch_context_mask,
															   batch_start_indices, batch_end_indices)

		all_start_correct = start_correct
		all_end_correct = end_correct
		all_span_correct = span_correct
		count += batch_size

	all_start_correct = (all_start_correct * 1.0)/ count
	all_end_correct = (all_end_correct * 1.0) / count
	all_span_correct = (all_span_correct * 1.0) / count
	return all_start_correct, all_end_correct, all_span_correct


def evaluate_with_candidate_spans(model, batches):
	mrr_value = []
	model.train(False)

	for iteration in range(len(batches)):
		batch = batches[iteration]
		batch_candidates = batch["candidates"]
		batch_answer_indices = batch['answer_indices']

		for index, query in enumerate(batch['queries']):
			# query tokens
			batch_query = variable(torch.LongTensor(query), volatile=True)
			batch_query_length = np.array([batch['qlengths'][index]])
			batch_question_mask = variable(torch.FloatTensor(batch['q_mask'][index]))

			# context tokens
			batch_context = variable(torch.LongTensor(batch['contexts'][index]))
			batch_context_length = np.array([batch['clengths'][index]])
			batch_context_mask = variable(torch.FloatTensor(batch['context_mask'][index]))

			## batch_candidates
			batch_start_indices = variable(torch.LongTensor([span[0] for span in batch['candidates'][index]]))
			batch_end_indices = variable(torch.LongTensor([span[1] for span in batch['candidates'][index]]))
			batch_len = len(batch_start_indices)

			indices = model.eval(batch_query, batch_query_length,batch_question_mask,
								 batch_context, batch_context_length,batch_context_mask,
								 batch_start_indices, batch_end_indices, batch_len)

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

def train_mrr(index, indices, batch_answer_indices):
	if args.use_cuda:
		indices = indices.data.cpu()
	else:
		indices = indices.data
	position_gold_sorted = (indices == batch_answer_indices[index]).nonzero().numpy()[0][0]
	index = position_gold_sorted + 1
	return (1.0 / (index))

def train_epochs_with_candidate_spans(model, vocab):
	clip_threshold = args.clip_threshold
	eval_interval = args.eval_interval

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.9))
	train_loss = 0
	train_denom = 0
	validation_history = []
	bad_counter = 0

	patience = 10
	## last true argument makes it collect multiple candidate spans
	valid_batches = make_bucket_batches(valid_documents, args.batch_length, vocab, True)
	mrr_value = []
	for epoch in range(args.num_epochs):

		print("Creating train batches")
		train_batches = make_bucket_batches(train_documents, args.batch_length, vocab, True)
		print("Starting epoch {}".format(epoch))

		saved = False
		for iteration in range(len(train_batches)):
			optimizer.zero_grad()
			if (iteration + 1) % eval_interval == 0:
				print("iteration {}".format(iteration + 1))
				print("train loss: {}".format(train_loss / train_denom))
				if iteration != 0:
					## last true argument makes it collect multiple candidate spans
					average_rr = evaluate_with_candidate_spans(model, valid_batches)
					validation_history.append(average_rr)
					train_average_rr = np.mean(mrr_value)
					if (iteration + 1) % (eval_interval * 5) == 0:
						print("Validation MRR:{0}".format(average_rr))
						print("Train MRR:{0}".format(train_average_rr))
						mrr_value = []
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
							evaluate_with_candidate_spans(model, valid_batches)
							exit(0)

			batch = train_batches[iteration]
			# view_batch(batch,loader.vocab)
			batch_candidates = batch["candidates"]
			batch_answer_indices = batch['answer_indices']
			batch_size = len(batch_candidates)
			batch_query_lengths = batch['qlengths']
			losses = variable(torch.zeros(batch_size))
			for index, query in enumerate(batch['queries']):

				## queries
				batch_query = variable(torch.LongTensor(query))
				batch_query_length = np.array([batch['qlengths'][index]])
				batch_question_mask = variable(torch.FloatTensor(batch['q_mask'][index]))

				## context
				batch_context = variable(torch.LongTensor(batch['contexts'][index]))
				batch_context_length = np.array([batch['clengths'][index]])
				batch_context_mask = variable(torch.FloatTensor(batch['context_mask'][index]))

				## metrics
				batch_metrics = variable(torch.FloatTensor(batch['metrics'][index]))


				## batch_candidates
				batch_start_indices = variable(torch.LongTensor([span[0] for span in batch['candidates'][index]]))
				batch_end_indices = variable(torch.LongTensor([span[1] for span in batch['candidates'][index]]))
				batch_len = len(batch_start_indices)

				## gold (all 0 for squad)
				gold_index = variable(torch.LongTensor([batch_answer_indices[index]]))

				loss, indices = model(batch_query, batch_query_length,
								   batch_question_mask,
								   batch_context, batch_context_length,
								   batch_context_mask,
								   batch_start_indices, batch_end_indices, gold_index, batch_metrics, batch_len)
				losses[index] = loss
				mrr_value.append(train_mrr(index, indices, batch_answer_indices))

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


def train_epochs(model, vocab):
	clip_threshold = args.clip_threshold
	eval_interval = args.eval_interval

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	train_loss = 0
	train_denom = 0
	validation_history = []
	bad_counter = 0
	all_start_correct = 0.0
	all_end_correct = 0.0
	all_span_correct  = 0.0
	coutn = 0
	patience = 10

	valid_batches = make_bucket_batches(valid_documents, args.batch_length, vocab)
	test_batches = make_bucket_batches(test_documents, args.batch_length, vocab)

	for epoch in range(args.num_epochs):

		print("Creating train batches")
		train_batches = make_bucket_batches(train_documents, args.batch_length, vocab)
		print("Starting epoch {}".format(epoch))

		model._span_start_accuracy = Accuracy()
		model._span_end_accuracy = Accuracy()
		model._span_accuracy = BooleanAccuracy()
		count = 0
		saved = False
		for iteration in range(len(train_batches)):
			optimizer.zero_grad()
			if (iteration + 1) % eval_interval == 0:
				print("iteration {}".format(iteration + 1))
				print("train loss: {}".format(train_loss / train_denom))

				if iteration != 0:
					start, end, span = evaluate(model, valid_batches, v_context_per_docid)
					validation_history.append(span)

					if (iteration + 1) % (eval_interval * 5) == 0:
						print("Train Accuracy: start_index = {0}, end_index = {1}, span = {2}".format(
							all_start_correct * 1.0 / count, all_end_correct * 1.0 /count,
							all_span_correct * 1.0 / count))
						print("Validation Accuracy: Start:{0} End:{1} Span:{2}".format(start, end, span))
						if span >= max(validation_history):
							saved = True
							print("Saving best model seen so far itr number {0}".format(iteration))
							torch.save(model.state_dict(), args.model_path)
							print("Best on Validation: Start:{0} End:{1} Span:{2}".format(start, end, span))
							bad_counter = 0
						else:
							bad_counter += 1
						if bad_counter > patience:
							print("Early Stopping")
							print("Testing started")
							model = MultiParagraph(args, loader)
							if args.use_cuda:
								model = model.cuda()
							model.load_state_dict(torch.load(args.model_path))
							evaluate(model, test_batches,te_context_per_docid)
							exit(0)

			batch = train_batches[iteration]
			#view_batch(batch,loader.vocab)
			batch_query_lengths = batch['qlengths']

			batch_start_indices = variable(torch.LongTensor(batch['start_indices']))
			batch_end_indices = variable(torch.LongTensor(batch['end_indices']))
			
			batch_size = len(batch_query_lengths)



			if args.elmo:
				batch_query= variable(torch.FloatTensor(batch['q_embed']))
				doc_ids = batch['doc_ids']
				batch_context_embed = []
				max_batch_context = len(batch['context_mask'][0])

				for doc_id in doc_ids:
					batch_context_embed.append(t_context_per_docid[doc_id])

				batch_context_embed_padded = np.array(
					[pad_seq_elmo(context_embed, max_batch_context) for context_embed in batch_context_embed])

				batch_context = variable(torch.FloatTensor(batch_context_embed_padded))

			else:
				batch_query = variable(torch.LongTensor(batch['queries']))
				batch_context = variable(torch.LongTensor(batch['contexts']))

			batch_query_length = np.array([batch['qlengths']])
			batch_question_mask = variable(torch.FloatTensor(batch['q_mask']))
			batch_context_length = np.array([batch['clengths']])
			batch_context_mask =variable(torch.FloatTensor(batch['context_mask']))


			identity_context = variable(torch.eye(batch_context_mask.size(1)) * -20000)
			
			loss, start_correct, end_correct, span_correct = model(batch_query, batch_query_length,batch_question_mask,
																   batch_context, batch_context_length, batch_context_mask,
						 batch_start_indices, batch_end_indices,identity_context)

			all_start_correct = start_correct
			all_end_correct = end_correct
			all_span_correct = span_correct
			
			loss = loss / batch_size
			loss.backward()
			optimizer.step()
			
			if args.use_cuda:
				train_loss += loss.data.cpu().numpy()[0] * batch_size

			else:
				train_loss += loss.data.numpy()[0] * batch_size

			train_denom += batch_size
			count += batch_size

		if not saved:
			print("Saving model after epoch {0}".format(epoch))
			torch.save(model.state_dict(), args.model_path + ".dummy")



	print("All epochs done")
	print("Testing started")
	model = MultiParagraph(args, loader)
	if args.use_cuda:
		model = model.cuda()
	model.load_state_dict(torch.load(args.model_path))
	evaluate(model, test_batches,te_context_per_docid)

if __name__ == "__main__":
	reload(sys)
	sys.setdefaultencoding('utf8')
	parser = argparse.ArgumentParser()

	parser.add_argument("--train_path", type=str, default="/../narrativeqa/summaries/small_summaries.pickle")
	parser.add_argument("--train_paragraph_path", type=str, default=None)
	parser.add_argument("--valid_path", type=str, default=None)
	parser.add_argument("--valid_paragraph_path", type=str, default=None)
	parser.add_argument("--test_path", type=str, default="../test_summaries.pickle")
	parser.add_argument("--summary_path", type=str, default=None)
	parser.add_argument("--model_path", type=str, default=None)
	parser.add_argument("--job_size", type=int, default=5)
	parser.add_argument("--pretrain_path", type=str, default=None, help="Path to the pre-trained word embeddings")
	parser.add_argument("--max_documents", type=int, default=0, help="If greater than 0, load at most this many documents")

	# Model parameters
	parser.add_argument("--hidden_size", type=int, default=10)
	parser.add_argument("--embed_size", type=int, default=1024)
	parser.add_argument("--cuda", action="store_true", default=True)
	parser.add_argument("--test", action="store_true", default=False)
	parser.add_argument("--batch_length", type=int, default=40)
	parser.add_argument("--eval_interval", type=int, default=2)
	parser.add_argument("--learning_rate", type=float, default=0.5)
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

	#For running squad
	loader = SquadDataloader(args)
	start = time()
	train_documents, train_articles = loader.load_documents_with_paragraphs(args.train_path, args.train_paragraph_path, max_documents=args.max_documents)
	valid_documents, valid_articles = loader.load_documents_with_paragraphs(args.valid_path, args.valid_paragraph_path, max_documents=args.max_documents)
	test_documents, test_articles = loader.load_documents_with_paragraphs(args.test_path, args.valid_paragraph_path, max_documents=args.max_documents)

	print("Train documents:{0} valid documents:{1} test documents:{2}".format(len(train_documents), len(valid_documents), len(test_documents)))
	end = time()
	print(end - start)

	model = MultiParagraph(args, loader)

	if args.use_cuda:
		model = model.cuda()

	# Get pre_trained embeddings
	if args.pretrain_path is not None:
		word_embedding = get_pretrained_emb(args.pretrain_path, loader.vocab.vocabulary, args.embed_size)
		loader.pretrain_embedding = word_embedding

	if args.test:
		model.load_state_dict(torch.load(args.model_path))
		test_model(model, test_documents, loader.vocab,te_context_per_docid)
	else:
		train_epochs(model, loader.vocab)
