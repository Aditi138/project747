import argparse
import sys
from dataloaders.dataloader import DataLoader, create_batches, view_batch, make_bucket_batches
#from dataloaders.squad_dataloader import SquadDataloader
from models.tri_attn_model import  TriAttn
from dataloaders.utility import get_pretrained_emb
import torch
from torch import optim
from dataloaders.utility import variable, view_data_point, pad_seq
import numpy as np
from time import time
import random
import pickle
import codecs



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


def evaluate(model, batches,  candidates_embed_docid, context_per_docid, candidates_per_docid, context_tokens_per_docid, context_ranges_per_docid, file):
	mrr_value = []
	model.train(False)
	s_file =  codecs.open(args.s_file,"w", encoding='utf-8')
	fout = codecs.open(file, "w", encoding='utf-8')
	for iteration in range(len(batches)):

		batch = batches[iteration]
		batch_doc_ids = batch['doc_ids']
		batch_q_tokens = batch['q_tokens']
		batch_candidates = batch["candidates"]
		batch_answer_indices = batch['answer_indices']
		batch_reduced_context_indices = batch['chunk_indices']
		for index, query_embed in enumerate(batch['q_embed']):

			if fout is not None:
				fout.write("\nQ: {0}".format(" ".join(batch_q_tokens[index])))
			# query tokens

			if args.emb_elmo:
				batch_query = variable(torch.FloatTensor(query_embed))
			else:
				batch_query = variable(torch.LongTensor(query_embed))


			batch_query_length = np.array([batch_query.size(0)])
			batch_question_mask = variable(torch.FloatTensor(np.array([1 for x in range(batch_query_length)])))



			# Sort the candidates by length
			if args.mcq == False:
				# Sort the candidates by length (only required if using an RNN)
				batch_candidate_lengths = np.array(batch_candidates["anslengths"][index])
				batch_candidate_mask = np.array(batch_candidates['mask'][index])
				gold_index = batch_answer_indices[index]
				candidate_sort = np.argsort(batch_candidate_lengths)[::-1].copy()
				batch_candidate_unsort = variable(torch.LongTensor(np.argsort(candidate_sort)))
				batch_candidate_masks_sorted = variable(torch.FloatTensor(batch_candidate_mask[candidate_sort]))
				batch_candidate_lengths_sorted = batch_candidate_lengths[candidate_sort]
			else:
				num_mcq = 10
				num_candidates = len(batch_candidates["anslengths"][index])
				other_candidates = []
				while len(other_candidates) != num_mcq - 1:
					random_index = random.randint(0, num_candidates - 1)
					if random_index != batch_answer_indices[index] and random_index not in other_candidates:
						other_candidates.append(random_index)
				all_candidates = other_candidates + [batch_answer_indices[index]]
				np.random.shuffle(all_candidates)

				gold_index = all_candidates.index(batch_answer_indices[index])

				batch_candidate_lengths = np.array(batch_candidates['anslengths'][index])[all_candidates]
				batch_candidate_mask = np.array(batch_candidates['mask'][index])[all_candidates]
				all_candidates = np.array(all_candidates)
				index_lookup = np.argsort(batch_candidate_lengths)[::-1].copy()
				candidate_sort = all_candidates[index_lookup]
				batch_candidate_unsort = variable(torch.LongTensor(np.argsort(index_lookup)))
				changed_batch_candidate_masks = change_masks(batch_candidate_mask[index_lookup],
															 np.max(batch_candidate_lengths))
				batch_candidate_masks_sorted = variable(torch.FloatTensor(changed_batch_candidate_masks))
				batch_candidate_lengths_sorted = batch_candidate_lengths[index_lookup]

			gold_index_variable = variable(torch.LongTensor([gold_index]))
			batch_answer_indices[index] = gold_index
			doc_id = batch_doc_ids[index]


			if args.emb_elmo:
				batch_candidates_embed_sorted = variable(
					torch.FloatTensor(candidates_embed_docid[doc_id][candidate_sort, ...]))
			else:
				batch_candidates_embed_sorted = variable(
					torch.LongTensor(candidates_embed_docid[doc_id][candidate_sort, ...]))


			batch_len = len(batch_candidate_lengths)
			negative_indices = list(range(batch_len))
			negative_indices.pop(gold_index)
			negative_indices = variable(torch.LongTensor(negative_indices))


			# context tokens
			## if using reduced context
			if args.sentence_scoring:
				## context is a set of ranges, pad them and context is a matrix, context_batch_size, weights based on gold ranges
				## no support for emb_elmo
				context_embeddings = context_per_docid[doc_id]  ## ids
				golden_ids = batch_reduced_context_indices[index]
				full_ranges = context_ranges_per_docid[doc_id]
				context_batch_length = len(full_ranges)
				context_lengths = np.array([r[1] - r[0] for r in full_ranges])
				max_context_chunk_length = max(context_lengths)
				batch_context_mask = np.array([[int(x < context_lengths[i])
												for x in range(max_context_chunk_length)]
											   for i in range(context_batch_length)])
				batched_context_embeddings = []
				for r in full_ranges:
					batched_context_embeddings.append(
						pad_seq(context_embeddings[r[0]:r[1]], max_context_chunk_length))
				batch_context = variable(torch.LongTensor(batched_context_embeddings))
				## dummyscores
				batch_context_scores = np.array([-10000] * context_batch_length)
				for g_id in golden_ids:
					batch_context_scores[g_id] = 0

				length_sort = np.argsort(context_lengths)[::-1].copy()
				batch_context_unsort = variable(torch.LongTensor(np.argsort(length_sort)))
				batch_context_mask_sorted = variable(torch.FloatTensor(batch_context_mask[length_sort]))
				batch_context_lengths_sorted = context_lengths[length_sort]
				batch_context_sorted = variable(
					torch.LongTensor(np.array(batched_context_embeddings)[length_sort, ...]))
				batch_context_scores_sorted = variable(torch.FloatTensor(batch_context_scores[length_sort]))
			else:
				if args.reduced:
					context_embeddings =  context_per_docid[doc_id]
					reduced_context_embeddings = []
					reduced_context = []
					ranges = batch_reduced_context_indices[index]
					if args.emb_elmo:
						for r in ranges:
							reduced_context_embeddings += context_embeddings[r[0]:r[1]].tolist()
							reduced_context += context_tokens_per_docid[doc_id][r[0]:r[1]]
						batch_context = variable(torch.FloatTensor(reduced_context_embeddings))
					else:
						for r in ranges:
							reduced_context_embeddings += context_embeddings[r[0]:r[1]]
							reduced_context += context_tokens_per_docid[doc_id][r[0]:r[1]]
						batch_context = variable(torch.LongTensor(reduced_context_embeddings))

					s_file.write("@@".join(reduced_context) + "\n" + " ".join(batch_q_tokens[index])  + "\n")
				else:
					if args.emb_elmo:
						batch_context = variable(torch.FloatTensor(context_per_docid[doc_id]))
					else:
						batch_context = variable(torch.LongTensor(context_per_docid[doc_id]))
					s_file.write("@@".join(context_tokens_per_docid[doc_id]) + "\n" + " ".join(batch_q_tokens[index])  + "\n")

				batch_context_length = np.array([batch_context.size(0)])
				batch_context_mask = variable(torch.FloatTensor(np.array([1 for x in range(batch_context_length[0])])))

			batch_len = len(batch_candidate_lengths_sorted)
			batch_candidate_unsort = variable(torch.LongTensor(np.argsort(candidate_sort)), volatile=True)

			if args.sentence_scoring:
				indices,c2q_attention_matrix,answer_scores_sorted = model.eval(batch_query, batch_query_length,batch_question_mask,
								 batch_context_sorted, batch_context_lengths_sorted,batch_context_mask_sorted,batch_context_scores_sorted,
								 batch_candidates_embed_sorted, batch_candidate_lengths_sorted, batch_candidate_masks_sorted,batch_candidate_unsort)
			else:
				indices, c2q_attention_matrix, answer_scores_sorted = model.eval(batch_query, batch_query_length, batch_question_mask,
							     batch_context, batch_context_length, batch_context_mask,
								 batch_candidates_embed_sorted, batch_candidate_lengths_sorted, batch_candidate_masks_sorted, batch_candidate_unsort)

			if args.use_cuda:
				indices = indices.data.cpu()
				c2q_attention_matrix = c2q_attention_matrix.data.cpu()
				answer_scores_sorted = answer_scores_sorted.data.cpu()

			else:
				indices = indices.data
				c2q_attention_matrix = c2q_attention_matrix.data.numpy()
				answer_scores_sorted = answer_scores_sorted.data.numpy()

			rows = c2q_attention_matrix.shape[1]
			cols = c2q_attention_matrix.shape[2]

			for row in c2q_attention_matrix.tolist()[0]:
				s_file.write(" ".join([str(c) for c in row]) + " ")
			s_file.write("\n")
			s_file.write(str(rows) + " " + str(cols) + "\n\n")
			position_gold_sorted = (indices == batch_answer_indices[index]).nonzero().numpy()[0][0]
			gold_index =  batch_answer_indices[index]
			index = position_gold_sorted + 1

			mrr_value.append(1.0 / (index))


			if args.mcq:
				candidates = [candidates_per_docid[doc_id][a] for a in all_candidates]
			else:
				candidates = candidates_per_docid[doc_id]
			if fout is not None:
				fout.write("\nRank: {0} / {1}   Gold: {2}\n".format(index, len(candidates), " ".join(
					candidates[indices[position_gold_sorted]])))
				for cand in range(10):
					fout.write("C: {0} Score:{1}\n".format(" ".join(candidates[indices[cand]]),
														   str(answer_scores_sorted[cand])))

	mean_rr = np.mean(mrr_value)
	print("MRR :{0}".format(mean_rr))
	model.train(True)
	return mean_rr


def change_masks(masks, max_length):
	changed_masks = []
	for mask in masks:
		changed_masks.append(mask[:max_length])
	return changed_masks

def train_epochs(model, vocab):


	clip_threshold = args.clip_threshold
	eval_interval = args.eval_interval

	#optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.learning_rate)
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
					average_rr = evaluate(model, valid_batches, valid_candidates_embed_docid, valid_context_per_docid, valid_candidate_per_docid, valid_context_tokens_per_docid, valid_context_ranges_per_docid , args.debug_file)
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
							evaluate(model, test_batches, test_candidates_embed_docid, test_context_per_docid, test_candidate_per_docid, test_context_tokens_per_docid, test_context_ranges_per_docid, args.debug_file+".test")
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
				if args.emb_elmo:
					batch_query = variable(torch.FloatTensor(query_embed))
				else:
					batch_query = variable(torch.LongTensor(query_embed))

				batch_query_length = np.array([batch_query.size(0)])
				batch_question_mask = variable(torch.FloatTensor(np.array([1 for x in range(batch_query_length[0])])))

				if args.mcq == False:
					# Sort the candidates by length (only required if using an RNN)
					batch_candidate_lengths = np.array(batch_candidates["anslengths"][index])
					batch_candidate_mask = np.array(batch_candidates['mask'][index])
					gold_index = batch_answer_indices[index]
					candidate_sort = np.argsort(batch_candidate_lengths)[::-1].copy()
					batch_candidate_unsort = variable(torch.LongTensor(np.argsort(candidate_sort)))
					batch_candidate_masks_sorted = variable(torch.FloatTensor(batch_candidate_mask[candidate_sort]))
					batch_candidate_lengths_sorted = batch_candidate_lengths[candidate_sort]
				else:
					num_mcq = 10
					num_candidates = len(batch_candidates["anslengths"][index])
					other_candidates = []
					while len(other_candidates) != num_mcq - 1:
						random_index = random.randint(0, num_candidates-1)
						if random_index != batch_answer_indices[index] and random_index not in other_candidates:
							other_candidates.append(random_index)
					all_candidates = other_candidates + [batch_answer_indices[index]]
					np.random.shuffle(all_candidates)

					gold_index = all_candidates.index(batch_answer_indices[index])

					batch_candidate_lengths = np.array(batch_candidates['anslengths'][index])[all_candidates]
					batch_candidate_mask = np.array(batch_candidates['mask'][index])[all_candidates]
					all_candidates = np.array(all_candidates)
					index_lookup = np.argsort(batch_candidate_lengths)[::-1].copy()
					candidate_sort = all_candidates[index_lookup]
					batch_candidate_unsort = variable(torch.LongTensor(np.argsort(index_lookup)))
					changed_batch_candidate_masks = change_masks(batch_candidate_mask[index_lookup], np.max(batch_candidate_lengths))
					batch_candidate_masks_sorted = variable(torch.FloatTensor(changed_batch_candidate_masks))
					batch_candidate_lengths_sorted = batch_candidate_lengths[index_lookup]

				gold_index_variable = variable(torch.LongTensor([gold_index]))
				batch_answer_indices[index] = gold_index
				doc_id = batch_doc_ids[index]
				if args.emb_elmo:
					batch_candidates_embed_sorted = variable(
					torch.FloatTensor(train_candidates_embed_docid[doc_id][candidate_sort, ...]))
				else:
					batch_candidates_embed_sorted = variable(
						torch.LongTensor(train_candidates_embed_docid[doc_id][candidate_sort, ...]))


				batch_len = len(batch_candidate_lengths)
				negative_indices = list(range(batch_len))
				negative_indices.pop(gold_index)
				negative_indices = variable(torch.LongTensor(negative_indices))

				# context tokens
				## if using reduced context
				if args.sentence_scoring:
					## context is a set of ranges, pad them and context is a matrix, context_batch_size, weights based on gold ranges
					## no support for emb_elmo
					context_embeddings = train_context_per_docid[doc_id]  ## ids
					golden_ids = batch_reduced_context_indices[index]
					full_ranges = train_context_ranges_per_docid[doc_id]
					context_batch_length = len(full_ranges)
					context_lengths = np.array([r[1] - r[0] for r in full_ranges])
					max_context_chunk_length = max(context_lengths)
					batch_context_mask = np.array([[int(x < context_lengths[i])
													for x in range(max_context_chunk_length)]
												   for i in range(context_batch_length)])
					batched_context_embeddings = []
					for r in full_ranges:
						batched_context_embeddings.append(
							pad_seq(context_embeddings[r[0]:r[1]], max_context_chunk_length))
					batch_context = variable(torch.LongTensor(batched_context_embeddings))
					## dummyscores
					batch_context_scores = np.array([-10000] * context_batch_length)
					for g_id in golden_ids:
						batch_context_scores[g_id] = 0


					length_sort = np.argsort(context_lengths)[::-1].copy()
					batch_context_unsort = variable(torch.LongTensor(np.argsort(length_sort)))
					batch_context_mask_sorted = variable(torch.FloatTensor(batch_context_mask[length_sort]))
					batch_context_lengths_sorted = context_lengths[length_sort]
					batch_context_sorted = variable(
						torch.LongTensor(np.array(batched_context_embeddings)[length_sort, ...]))
					batch_context_scores_sorted = variable(torch.FloatTensor(batch_context_scores[length_sort]))
				else:
					if args.reduced:
						context_embeddings =  train_context_per_docid[doc_id]
						reduced_context_embeddings = []
						ranges = batch_reduced_context_indices[index]

						if args.emb_elmo:
							for r in ranges:
								reduced_context_embeddings += context_embeddings[r[0]:r[1]].tolist()
								batch_context = variable(torch.FloatTensor(reduced_context_embeddings))
						else:
							for r in ranges:
								reduced_context_embeddings += context_embeddings[r[0]:r[1]]
								batch_context = variable(torch.LongTensor(reduced_context_embeddings))
					else:
						if args.emb_elmo:
							batch_context = variable(torch.FloatTensor(train_context_per_docid[doc_id]))
						else:
							batch_context = variable(torch.LongTensor(train_context_per_docid[doc_id]))

					batch_context_length = np.array([batch_context.size(0)])
					batch_context_mask =variable(torch.FloatTensor(np.array([1 for x in range(batch_context_length[0])])))



				if args.sentence_scoring:
					loss, indices = model(batch_query, batch_query_length,batch_question_mask,
									batch_context_sorted, batch_context_lengths_sorted, batch_context_mask_sorted, batch_context_scores_sorted,
									batch_candidates_embed_sorted, batch_candidate_lengths_sorted, batch_candidate_masks_sorted,
										  batch_candidate_unsort,
									gold_index_variable
									)
				else:
					loss, indices = model(batch_query, batch_query_length, batch_question_mask,
										  batch_context, batch_context_length, batch_context_mask,
										  batch_candidates_embed_sorted, batch_candidate_lengths_sorted,
										  batch_candidate_masks_sorted,
										  batch_candidate_unsort,
										  gold_index_variable
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
	evaluate(model, test_batches, test_candidates_embed_docid, test_context_per_docid, test_candidate_per_docid, test_context_tokens_per_docid,test_context_ranges_per_docid,args.debug_file+".test")

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
    evaluate(model, test_batches, test_candidates_embed_docid, test_context_per_docid, test_candidate_per_docid, test_context_tokens_per_docid,test_context_ranges_per_docid, args.debug_file + ".test")

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
	parser.add_argument("--debug_file", type=str, default="./debug_outputs.txt")
	parser.add_argument("--s_file", type=str, default="./attention.txt")

	# Model parameters
	parser.add_argument("--hidden_size", type=int, default=128)
	parser.add_argument("--embed_size", type=int, default=1024)
	parser.add_argument("--cuda", action="store_true", default=True)
	parser.add_argument("--test", action="store_true", default=False)
	parser.add_argument("--elmo", action="store_true", default=False)
	parser.add_argument("--batch_length", type=int, default=10)
	parser.add_argument("--eval_interval", type=int, default=2)
	parser.add_argument("--learning_rate", type=float, default=0.0001)
	parser.add_argument("--dropout", type=float, default=0.4)
	parser.add_argument("--dropout_emb", type=float, default=0.4)
	parser.add_argument("--num_epochs", type=int, default=10)
	parser.add_argument("--clip_threshold", type=int, default=10)
	parser.add_argument("--num_layers", type=int, default=3)
	parser.add_argument("--ner_dim", type=int, default=32)
	parser.add_argument("--pos_dim", type=int, default=32)

	parser.add_argument("--meteor_path", type=str, default=10)
	parser.add_argument("--seed", type=int, default=1234)
	parser.add_argument("--profile", action="store_true")
	parser.add_argument("--squad", action="store_true")
	parser.add_argument("--reduced", action="store_true", default=False)
	parser.add_argument("--emb_elmo", action="store_true",default=False)
	parser.add_argument("--mcq", action="store_true", default=False)
	parser.add_argument("--sentence_scoring", action="store_true", default=False)

	args = parser.parse_args()
	
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

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

		train_documents, train_candidates_embed_docid,train_candidate_per_docid,train_context_per_docid,train_context_tokens_per_docid,_ = loader.load_documents_elmo(t_documents,split=False,train=True)
		valid_documents,valid_candidates_embed_docid,valid_candidate_per_docid,valid_context_per_docid,valid_context_tokens_per_docid,_ = loader.load_documents_elmo(v_documents,split=False)
		test_documents, test_candidates_embed_docid,test_candidate_per_docid,test_context_per_docid,test_context_tokens_per_docid,_ = loader.load_documents_elmo(te_documents,split=False)

		# fout = codecs.open("manual_check.txt", "w", encoding='utf-8')
		# for i in range(20):
		# 	q = "Q:" + " ".join(train_documents[i].question_tokens)
		# 	p = " ".join(train_context_tokens_per_docid[train_documents[i].doc_id])
		# 	c = [" ".join(c) for c in train_documents[i].candidates]
		# 	idx = train_documents[i].answer_indices[0]
		# 	correct = " ".join(train_documents[i].candidates[idx])
		# 	fout.write(q + "\n")
		# 	for cc in c:
		# 		fout.write("C:" + cc + "\n")
		# 	fout.write("Correct:{0}".format(idx) + "\n\n")
		#exit(0)

	elif args.reduced:
		loader = DataLoader(args)
		with open(args.train_path, "r") as fin:
			t_documents = pickle.load(fin)
		with open(args.valid_path, "r") as fin:
			v_documents = pickle.load(fin)
		with open(args.test_path, "r") as fin:
			te_documents = pickle.load(fin)
		print("Loading training documents")
		train_documents, train_candidates_embed_docid, train_candidate_per_docid,train_context_per_docid,train_context_tokens_per_docid, train_context_ranges_per_docid = loader.load_documents_split_sentences(t_documents)
		print("Loading validation documents")
		valid_documents, valid_candidates_embed_docid, valid_candidate_per_docid,valid_context_per_docid,valid_context_tokens_per_docid, valid_context_ranges_per_docid = loader.load_documents_split_sentences(v_documents)
		print("Loading testing documents")
		test_documents, test_candidates_embed_docid, test_candidate_per_docid,test_context_per_docid,test_context_tokens_per_docid, test_context_ranges_per_docid = loader.load_documents_split_sentences(te_documents)

		# fout = codecs.open("manual_check.txt","w", encoding='utf-8')
		# for i in range(20):
		# 	q = "Q:" + " ".join(train_documents[i].question_tokens)
		# 	doc_id = train_documents[i].doc_id
		# 	context = train_context_tokens_per_docid[doc_id]
		# 	reduced_context_ = []
		# 	ranges = train_documents[i].chunk_indices
		# 	for r in ranges:
		# 		reduced_context_ += context[r[0]:r[1]]
		# 	p = " ".join(reduced_context_)
		# 	c = [" ".join(c) for c in train_documents[i].candidates]
		# 	idx = train_documents[i].answer_indices[0]
		# 	correct = " ".join(train_documents[i].candidates[idx])
		# 	fout.write(q + "\n")
		# 	fout.write(p + "\n")
		# 	for cc in c:
		# 		fout.write("C:"+ cc+ "\n")
		# 	fout.write("Correct:{0}".format(idx) + "\n\n")
		# exit(0)

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
	model = TriAttn(args, loader)

	if args.use_cuda:
		model = model.cuda()

	if args.test:
		model = torch.load(args.model_path)
		test_model(model, test_documents, loader.vocab)
	else:
		train_epochs(model, loader.vocab)
