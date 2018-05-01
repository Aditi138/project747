import argparse
import sys
from dataloaders.dataloader import DataLoader, create_batches, view_batch, make_bucket_batches
from dataloaders.squad_dataloader import SquadDataloader
from models.context_model import ContextMRR
from models.context_model_sep import ContextMRR_Sep
from models.context_model_sep_switched import  ContextMRR_Sep_Switched
from dataloaders.utility import get_pretrained_emb
import torch
from torch import optim
from dataloaders.utility import variable, view_data_point
import numpy as np
from time import time
import random
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

def get_random_batch_from_training(batches, num):
	small = []
	for i in range(num):
		index = random.randint(0, len(batches)-1)
		small.append(batches[index])
	return small


def evaluate(model, batches,  candidates_embed_docid, context_per_docid ):
	mrr_value = []
	model.train(False)
	for iteration in range(len(batches)):

		batch = batches[iteration]
		batch_doc_ids = batch['doc_ids']
		batch_candidates = batch["candidates"]
		batch_answer_indices = batch['answer_indices']
		batch_reduced_context_indices = batch['chunk_indices']
		for index, query_embed in enumerate(batch['q_embed']):

			# query tokens
			batch_query = variable(torch.FloatTensor(query_embed), volatile=True)
			batch_query_length = np.array([batch['qlengths'][index]])
			batch_question_mask = variable(torch.FloatTensor(np.array([1 for x in range(batch_query_length)])))



			# Sort the candidates by length
			batch_candidate_lengths = np.array(batch_candidates["anslengths"][index])
			batch_candidate_mask = np.array(batch_candidates['mask'][index])
			candidate_sort = np.argsort(batch_candidate_lengths)[::-1].copy()

			doc_id = batch_doc_ids[index]
			batch_candidates_embed_sorted = variable(
				torch.FloatTensor(candidates_embed_docid[doc_id][candidate_sort, ...]))
			batch_candidate_lengths_sorted = batch_candidate_lengths[candidate_sort]
			batch_candidate_masks_sorted = variable(torch.FloatTensor(batch_candidate_mask[candidate_sort]))


			# context tokens
			## if using reduced context
			if args.reduced:
				context_embeddings =  context_per_docid[doc_id]
				reduced_context_embeddings = []
				ranges = batch_reduced_context_indices[index]
				for r in ranges:
				   reduced_context_embeddings += context_embeddings[r[0]:r[1]].tolist()
				batch_context = variable(torch.FloatTensor(reduced_context_embeddings))
			else:
				batch_context = variable(torch.FloatTensor(context_per_docid[doc_id]))

			batch_context_length = np.array([batch_context.size(0)])
			batch_context_mask = variable(torch.FloatTensor(np.array([1 for x in range(batch_context_length[0])])))

			batch_len = len(batch_candidate_lengths_sorted)
			batch_candidate_unsort = variable(torch.LongTensor(np.argsort(candidate_sort)), volatile=True)

			indices = model.eval(batch_query, batch_query_length,batch_question_mask,
								 batch_context, batch_context_length,batch_context_mask,
								 batch_candidates_embed_sorted, batch_candidate_lengths_sorted, batch_candidate_masks_sorted,batch_candidate_unsort)

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


	patience = 30

	valid_batches = create_batches(valid_documents, args.batch_length,args.job_size, vocab)
	test_batches = create_batches(test_documents,args.batch_length,args.job_size, vocab)

	mrr_value = []
	for epoch in range(args.num_epochs):

		print("Creating train batches")
		train_batches = make_bucket_batches(train_documents, args.batch_length, vocab)
		print("Starting epoch {}".format(epoch))

		saved = False
		for iteration in range(len(train_batches)):
			optimizer.zero_grad()
			if (iteration + 1) % eval_interval == 0:
				print("iteration: {0} train loss: {1}".format(iteration + 1, train_loss / train_denom))

				if iteration != 0:
					average_rr = evaluate(model, valid_batches, valid_candidates_embed_docid, valid_context_per_docid)
					validation_history.append(average_rr)
					train_average_rr = np.mean(mrr_value)
					if (iteration + 1) % (eval_interval) == 0:
						print("Train MRR:{0}  Validation MRR:{1}".format(train_average_rr,average_rr))

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
							model = torch.load(args.model_path)
							evaluate(model, test_batches, test_candidates_embed_docid, test_context_per_docid )
							exit(0)

			batch = train_batches[iteration]
			# view_batch(batch,loader.vocab)
			batch_query_lengths = batch['qlengths']
			batch_candidates = batch["candidates"]
			batch_doc_ids = batch['doc_ids']
			batch_reduced_context_indices = batch['chunk_indices']
			batch_answer_indices = batch['answer_indices']
			batch_size = len(batch_query_lengths)
			losses = variable(torch.zeros(batch_size))
			for index, query_embed in enumerate(batch['q_embed']):
				# query tokens
				batch_query = variable(torch.FloatTensor(query_embed))
				batch_query_length = np.array([batch['qlengths'][index]])
				batch_question_mask = variable(torch.FloatTensor(np.array([1 for x in range(batch_query_length)])))

				# Sort the candidates by length (only required if using an RNN)
				batch_candidate_lengths = np.array(batch_candidates["anslengths"][index])
				batch_candidate_mask = np.array(batch_candidates['mask'][index])
				candidate_sort = np.argsort(batch_candidate_lengths)[::-1].copy()

				# get candidates_embed from doc_id
				doc_id = batch_doc_ids[index]
				batch_candidates_embed_sorted = variable(
					torch.FloatTensor(train_candidates_embed_docid[doc_id][candidate_sort, ...]))

				batch_candidate_lengths_sorted = batch_candidate_lengths[candidate_sort]
				batch_candidate_unsort = variable(torch.LongTensor(np.argsort(candidate_sort)))
				batch_candidate_masks_sorted = variable(torch.FloatTensor(batch_candidate_mask[candidate_sort]))

				batch_len = len(batch_candidate_lengths)

				# context tokens
				## if using reduced context
				if args.reduced:
					context_embeddings =  train_context_per_docid[doc_id]
					reduced_context_embeddings = []
					ranges = batch_reduced_context_indices[index]
					for r in ranges:
						reduced_context_embeddings += context_embeddings[r[0]:r[1]].tolist()
					batch_context = variable(torch.FloatTensor(reduced_context_embeddings))
				else:
					batch_context = variable(torch.FloatTensor(train_context_per_docid[doc_id]))

				batch_context_length = np.array([batch_context.size(0)])
				batch_context_mask =variable(torch.FloatTensor(np.array([1 for x in range(batch_context_length[0])])))

				gold_index = variable(torch.LongTensor([batch_answer_indices[index]]))
				negative_indices = [idx for idx in range(batch_len)]
				negative_indices.pop(batch_answer_indices[index])
				negative_indices = variable(torch.LongTensor(negative_indices))



				loss, indices = model(batch_query, batch_query_length,batch_question_mask,
									batch_context, batch_context_length, batch_context_mask,
									  batch_candidates_embed_sorted, batch_candidate_lengths_sorted, batch_candidate_masks_sorted,
										  batch_candidate_unsort,
									gold_index, negative_indices
									)

				losses[index] = loss
				mrr_value.append(train_mrr(index, indices, batch_answer_indices))

			# loss.backward()
			# torch.nn.utils.clip_grad_norm(model.parameters(), clip_threshold)
			# optimizer.step()

			mean_loss = losses.mean(0)
			mean_loss.backward()
			torch.nn.utils.clip_grad_norm(model.parameters(), clip_threshold)
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
	model = torch.load(args.model_path)
	evaluate(model, test_batches, test_candidates_embed_docid, test_context_per_docid )

def train_mrr(index, indices, batch_answer_indices):
	if args.use_cuda:
		indices = indices.data.cpu()
	else:
		indices = indices.data
	position_gold_sorted = (indices == batch_answer_indices[index]).nonzero().numpy()[0][0]
	index = position_gold_sorted + 1
	return (1.0 / (index))

def test_model(model, documents,vocab):
    test_batches = create_batches(documents,args.batch_length,args.job_size, vocab)
    print("Testing!")
    evaluate(model, test_batches)

if __name__ == "__main__":
	reload(sys)
	sys.setdefaultencoding('utf8')
	parser = argparse.ArgumentParser()

	parser.add_argument("--train_path", type=str, default="/../narrativeqa/summaries/small_summaries.pickle")
	parser.add_argument("--valid_path", type=str, default=None)
	parser.add_argument("--test_path", type=str, default=None)
	parser.add_argument("--summary_path", type=str, default=None)
	parser.add_argument("--model_path", type=str, default=None)
	parser.add_argument("--pickle_folder", type=str, default=None)
	parser.add_argument("--job_size", type=int, default=5)
	parser.add_argument("--pretrain_path", type=str, default=None, help="Path to the pre-trained word embeddings")
	parser.add_argument("--max_documents", type=int, default=0, help="If greater than 0, load at most this many documents")

	# Model parameters
	parser.add_argument("--hidden_size", type=int, default=128)
	parser.add_argument("--embed_size", type=int, default=1024)
	parser.add_argument("--cuda", action="store_true", default=True)
	parser.add_argument("--test", action="store_true", default=False)
	parser.add_argument("--elmo", action="store_true", default=False)
	parser.add_argument("--batch_length", type=int, default=10)
	parser.add_argument("--eval_interval", type=int, default=2)
	parser.add_argument("--learning_rate", type=float, default=0.0001)
	parser.add_argument("--dropout", type=float, default=0.2)
	parser.add_argument("--num_epochs", type=int, default=10)
	parser.add_argument("--clip_threshold", type=int, default=10)
	parser.add_argument("--num_layers", type=int, default=3)
	parser.add_argument("--ner_dim", type=int, default=32)
	parser.add_argument("--pos_dim", type=int, default=32)

	parser.add_argument("--meteor_path", type=str, default=10)
	parser.add_argument("--profile", action="store_true")
	parser.add_argument("--squad", action="store_true")
	parser.add_argument("--reduced", action="store_true")

	args = parser.parse_args()

	torch.manual_seed(2)

	if args.cuda and torch.cuda.is_available():
		vars(args)['use_cuda'] = True
	else:
		vars(args)['use_cuda'] = False

	print(args)

	start = time()
	if args.squad:
		loader = SquadDataloader(args)
		train_documents = loader.load_documents_with_candidates(args.train_path)
		valid_documents = loader.load_documents_with_candidates(args.valid_path)
		test_documents = loader.load_documents_with_candidates(args.valid_path)
	elif args.elmo:
		loader = DataLoader(args)
		with open(args.train_path, "r") as fin:
			t_documents = pickle.load(fin)
		with open(args.valid_path, "r") as fin:
			v_documents = pickle.load(fin)
		with open(args.test_path, "r") as fin:
			te_documents = pickle.load(fin)

		train_documents, train_candidates_embed_docid,train_context_per_docid = loader.load_documents_elmo(t_documents)
		valid_documents,valid_candidates_embed_docid,valid_context_per_docid = loader.load_documents_elmo(v_documents)
		test_documents, test_candidates_embed_docid,test_context_per_docid = loader.load_documents_elmo(te_documents)
	elif args.reduced:
		loader = DataLoader(args)
		with open(args.train_path, "r") as fin:
			t_documents = pickle.load(fin)
		with open(args.valid_path, "r") as fin:
			v_documents = pickle.load(fin)
		with open(args.test_path, "r") as fin:
			te_documents = pickle.load(fin)
		print("Loading training documents")
		train_documents, train_candidates_embed_docid, train_context_per_docid = loader.load_documents_split_sentences(t_documents)
		print("Loading validation documents")
		valid_documents, valid_candidates_embed_docid, valid_context_per_docid = loader.load_documents_split_sentences(v_documents)
		print("Loading testing documents")
		test_documents, test_candidates_embed_docid, test_context_per_docid = loader.load_documents_split_sentences(te_documents)
		with open(args.pickle_folder + "train_reduced_summaries.pickle", "wb") as fout:
			pickle.dump(train_documents, fout)
		with open(args.pickle_folder + "valid_reduced_summaries.pickle", "wb") as fout:
			pickle.dump(valid_documents, fout)
		with open(args.pickle_folder + "test_reduced_summaries.pickle", "wb") as fout:
			pickle.dump(test_documents, fout)
	else:

		loader = DataLoader(args)
		train_documents = loader.load_documents(args.train_path, summary_path=args.summary_path, max_documents=args.max_documents)
		valid_documents = loader.load_documents(args.valid_path, summary_path=None, max_documents=args.max_documents)
		test_documents = loader.load_documents(args.test_path, summary_path=None, max_documents=args.max_documents)

	end = time()
	print(end - start)

	# Get pre_trained embeddings
	if args.pretrain_path is not None:
		word_embedding = get_pretrained_emb(args.pretrain_path, loader.vocab.vocabulary, args.embed_size)
		loader.pretrain_embedding = word_embedding

	#model = ContextMRR_Sep(args, loader)
	model = ContextMRR_Sep_Switched(args, loader)

	if args.use_cuda:
		model = model.cuda()

	if args.test:
		model = torch.load(args.model_path)
		test_model(model, test_documents, loader.vocab)
	else:
		train_epochs(model, loader.vocab)
