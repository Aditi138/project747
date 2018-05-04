import argparse
import sys

from dataloaders.dataloader import create_batches, view_batch, make_bucket_batches, DataLoader
from dataloaders.squad_dataloader import SquadDataloader
from models.multi_paragraph_model import MultiParagraph, Accuracy, BooleanAccuracy
import torch
from torch import optim
from dataloaders.utility import variable, view_data_point,view_span_data_point,get_pretrained_emb, pad_seq_elmo, pad_seq
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

def test_model(model, documents, vocab, context_per_docid=None):
    test_batches = make_bucket_batches(documents, args.batch_length, vocab)
    print("Testing!")
    start,end,span = evaluate(model, test_batches)
    print("Testing Accuracy: Start:{0} End:{1} Span:{2}".format(start, end, span))


def get_random_batch_from_training(batches, num):
	small = []
	for i in range(num):
		index = random.randint(0, len(batches)-1)
		small.append(batches[index])
	return small


def evaluate(model, batches,context_per_docid=None):
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
		# view_batch(batch,loader.vocab
		if iteration % 500 == 0:
			print("Processed :{0} documents".format(iteration))

		if args.elmo:
			batch_query = variable(torch.FloatTensor(batch['q_embed']))
			doc_ids = batch['doc_ids']
			batch_context_embed = []
			max_batch_context = len(batch['context_mask'][0])

			for doc_id in doc_ids:
				batch_context_embed.append(context_per_docid[doc_id])

			batch_context_embed_padded = np.array(
				[pad_seq_elmo(context_embed, max_batch_context) for context_embed in batch_context_embed])

			batch_context = variable(torch.FloatTensor(batch_context_embed_padded))
		else:
			batch_query = variable(torch.LongTensor(batch.question_tokens))
			batch_query_length = np.array([len(batch.question_tokens)])

		batch_question_mask = variable(torch.FloatTensor([1 for i in range(len(batch.question_tokens))]))

		all_paragraphs = np.array(valid_articles[batch.article_id])
		top_paragraphs = all_paragraphs[batch.top_paragraph_ids]
		batch_context_lengths = np.array([len(paragraph) for paragraph in top_paragraphs])
		maximum_context_length = max(batch_context_lengths)
		contexts = np.array(
			[pad_seq(paragraph, maximum_context_length) for paragraph in top_paragraphs])
		batch_context_mask = np.array([[int(x < batch_context_lengths[i])
										for x in range(maximum_context_length)] for i in range(len(top_paragraphs))])
		context_sort = np.argsort(batch_context_lengths)[::-1].copy()
		batch_context_sorted = variable(torch.LongTensor(contexts[context_sort, ...]))
		batch_context_lengths_sorted = batch_context_lengths[context_sort]
		batch_context_unsort = variable(torch.LongTensor(np.argsort(context_sort)))
		batch_context_masks_sorted = variable(torch.FloatTensor(batch_context_mask[context_sort]))
		paragraph_ids_sorted = np.array(batch.top_paragraph_ids)[context_sort]
		offset_paragraph_index = np.where(paragraph_ids_sorted == batch.gold_paragraph_id)[0]
		if len(offset_paragraph_index) == 0:
			print("Gold paragraph not found in top k:{0} Gold:{1}" .format(batch.top_paragraph_ids,batch.gold_paragraph_id))
			continue
		offset_paragraph_index = offset_paragraph_index[0]
		start_index = batch.span_indices[0]
		end_index = batch.span_indices[1]

		batch_start_indices = variable(
			torch.LongTensor([start_index + offset_paragraph_index * maximum_context_length]))
		batch_end_indices = variable(torch.LongTensor([end_index + offset_paragraph_index * maximum_context_length]))

		identity_context = variable(torch.eye(batch_context_masks_sorted.size(1)) * -20000)

		start_correct, end_correct, span_correct = model.eval(batch_query, batch_query_length, batch_question_mask,
															   batch_context_sorted, batch_context_lengths_sorted,
															   batch_context_masks_sorted,
															    batch_start_indices,
															   batch_end_indices, identity_context)

		all_start_correct = start_correct
		all_end_correct = end_correct
		all_span_correct = span_correct
		count += 1

	all_start_correct = (all_start_correct * 1.0)/ count
	all_end_correct = (all_end_correct * 1.0) / count
	all_span_correct = (all_span_correct * 1.0) / count
	return all_start_correct, all_end_correct, all_span_correct


def train_epochs(model, vocab,t_context_per_docid=None):
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
	l = 0
	patience = 10

	valid_batches = valid_documents
	#test_batches = make_bucket_batches(test_documents, args.batch_length, vocab)

	for epoch in range(args.num_epochs):

		print("Creating train batches")
		random.shuffle(train_documents)
		print("Starting epoch {}".format(epoch))

		model._span_start_accuracy = Accuracy()
		model._span_end_accuracy = Accuracy()
		model._span_accuracy = BooleanAccuracy()
		count = 0
		saved = False
		optimizer.zero_grad()
		losses = variable(torch.zeros(args.batch_length))


		for iteration in range(len(train_documents)):


			if (count + 1) % args.batch_length == 0:
				batch_size = losses.size(0)
				mean_loss = torch.mean(losses)
				mean_loss.backward()
				torch.nn.utils.clip_grad_norm(model.parameters(), clip_threshold)
				optimizer.step()
				optimizer.zero_grad()
				if args.use_cuda:
					train_loss += loss.data.cpu().numpy()[0] * batch_size

				else:
					train_loss += loss.data.numpy()[0] * batch_size
				l = 0
				losses = variable(torch.zeros(args.batch_length))



			if (iteration + 1) % eval_interval == 0:
				print("iteration {}".format(iteration + 1))
				print("train loss: {}".format(train_loss / train_denom))

				if iteration != 0:
					start, end, span = evaluate(model, valid_batches)
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
							evaluate(model, valid_batches)
							exit(0)

			batch = train_documents[iteration]

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
				batch_query = variable(torch.LongTensor(batch.question_tokens))
				batch_query_length = np.array([len(batch.question_tokens)])

			batch_question_mask = variable(torch.FloatTensor([1 for i in range(len(batch.question_tokens))]))

			all_paragraphs = np.array(train_articles[batch.article_id])
			top_paragraphs = all_paragraphs[batch.top_paragraph_ids]
			batch_context_lengths = np.array([len(paragraph) for paragraph in top_paragraphs])
			maximum_context_length = max(batch_context_lengths)
			contexts = np.array(
				[pad_seq(paragraph, maximum_context_length) for paragraph in top_paragraphs])
			batch_context_mask = np.array([[int(x < batch_context_lengths[i])
											for x in range(maximum_context_length)] for i in range(len(top_paragraphs))])



			#Sort the paragraphs
			context_sort = np.argsort(batch_context_lengths)[::-1].copy()
			batch_context_sorted = variable(torch.LongTensor(contexts[context_sort, ...]))
			batch_context_lengths_sorted = batch_context_lengths[context_sort]
			batch_context_unsort = variable(torch.LongTensor(np.argsort(context_sort)))
			batch_context_masks_sorted = variable(torch.FloatTensor(batch_context_mask[context_sort]))
			paragraph_ids_sorted = np.array(batch.top_paragraph_ids)[context_sort]
			offset_paragraph_index = np.where(paragraph_ids_sorted == batch.gold_paragraph_id)[0]
			if len(offset_paragraph_index) == 0:
				print("Gold paragraph not found in top k")
				continue
			offset_paragraph_index = offset_paragraph_index[0]
			start_index = batch.span_indices[0]
			end_index = batch.span_indices[1]

			batch_start_indices = variable(torch.LongTensor([start_index + offset_paragraph_index * maximum_context_length]))
			batch_end_indices = variable(torch.LongTensor([end_index + offset_paragraph_index * maximum_context_length]))

			identity_context = variable(torch.eye(batch_context_masks_sorted.size(1)) * -20000)
			
			loss, start_correct, end_correct, span_correct = model(batch_query, batch_query_length,batch_question_mask,
																   batch_context_sorted, batch_context_lengths_sorted, batch_context_masks_sorted,
																   batch_context_unsort,batch_start_indices, batch_end_indices,identity_context)

			losses[l] = loss
			l+=1
			all_start_correct = start_correct
			all_end_correct = end_correct
			all_span_correct = span_correct
			train_denom += 1
			count += 1

			

		#Handling last batch
		if count < len(train_documents):
			batch_size = losses.size(0)
			mean_loss = torch.mean(losses)

			mean_loss.backward()
			torch.nn.utils.clip_grad_norm(model.parameters(), clip_threshold)
			optimizer.step()
			optimizer.zero_grad()
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
	evaluate(model, test_batches)

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
	parser.add_argument("--elmo", action="store_true", default=False)
	parser.add_argument("--test", action="store_true", default=False)
	parser.add_argument("--batch_length", type=int, default=10)
	parser.add_argument("--eval_interval", type=int, default=20)
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
	#test_documents, test_articles = loader.load_documents_with_paragraphs(args.test_path, args.valid_paragraph_path, max_documents=args.max_documents)

	print("Train documents:{0} valid documents:{1} test documents:{2}".format(len(train_documents), len(valid_documents),0))
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
		test_model(model, test_documents, loader.vocab)
	else:
		train_epochs(model, loader.vocab)
