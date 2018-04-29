import json
import os
from nltk.tokenize import word_tokenize
from data import Span_Data_Point, Data_Point
from nltk.stem import PorterStemmer as NltkPorterStemmer
from collections import Counter, defaultdict
import spacy
import pickle
import argparse
from test_metrics import Performance
import random, math

class SquadDataloader():
	def __init__(self, args):
		self.vocab = Vocabulary()
		self.stemmer = NltkPorterStemmer()
		self.performance = Performance(args)
		self.nlp =  spacy.load('en')

	def tokenize(self, text, chunk_length = 20, chunk= False):
		# tokens = [self.stemmer.stem(token) for token in word_tokenize(text.lower())]
		#tokenized_sents = [[token.string.strip() for token in s] for s in doc.sents]
		doc = self.nlp(text)
		tokens = [t for t in self.nlp(text) if not t.is_space]
		raw_tokens = [t.text for t in tokens]
		chunk_storage = []


		if chunk:
			total_chunks = len(raw_tokens) // chunk_length
			if len(raw_tokens) % chunk_length > 0:
				total_chunks += 1

			for chunk_number in range(total_chunks - 1):
				chunk = raw_tokens[chunk_length * chunk_number:chunk_length * (chunk_number + 1)]
				chunk_storage.append(chunk)

			# Handle last chunk separately
			chunk = raw_tokens[-chunk_length:]
			chunk_storage.append(chunk)

		return chunk_storage,tokens

	def char_span_to_token_span(self, token_offsets, character_span):
		"""
		Converts a character span from a passage into the corresponding token span in the tokenized
		version of the passage.  If you pass in a character span that does not correspond to complete
		tokens in the tokenized version, we'll do our best, but the behavior is officially undefined.
		We return an error flag in this case, and have some debug logging so you can figure out the
		cause of this issue (in SQuAD, these are mostly either tokenization problems or annotation
		problems; there's a fair amount of both).
		The basic outline of this method is to find the token span that has the same offsets as the
		input character span.  If the tokenizer tokenized the passage correctly and has matching
		offsets, this is easy.  We try to be a little smart about cases where they don't match exactly,
		but mostly just find the closest thing we can.
		The returned ``(begin, end)`` indices are `inclusive` for both ``begin`` and ``end``.
		So, for example, ``(2, 2)`` is the one word span beginning at token index 2, ``(3, 4)`` is the
		two-word span beginning at token index 3, and so on.
		Returns
		-------
		token_span : ``Tuple[int, int]``
			`Inclusive` span start and end token indices that match as closely as possible to the input
			character spans.
		error : ``bool``
			Whether the token spans match the input character spans exactly.  If this is ``False``, it
			means there was an error in either the tokenization or the annotated character span.
		"""
		# We have token offsets into the passage from the tokenizer; we _should_ be able to just find
		# the tokens that have the same offsets as our span.
		error = False
		start_index = 0
		while start_index < len(token_offsets) and token_offsets[start_index][0] < character_span[0]:
			start_index += 1
		# start_index should now be pointing at the span start index.
		if token_offsets[start_index][0] > character_span[0]:
			# In this case, a tokenization or labeling issue made us go too far - the character span
			# we're looking for actually starts in the previous token.  We'll back up one.
			pass
			# print("Bad labelling or tokenization - start offset doesn't match")
			start_index -= 1
		if token_offsets[start_index][0] != character_span[0]:
			error = True
		end_index = start_index
		while end_index < len(token_offsets) and token_offsets[end_index][1] < character_span[1]:
			end_index += 1
		if end_index == start_index and token_offsets[end_index][1] > character_span[1]:
			# Looks like there was a token that should have been split, like "1854-1855", where the
			# answer is "1854".  We can't do much in this case, except keep the answer as the whole
			# token.
			pass
			# print("Bad tokenization - end offset doesn't match")
		elif token_offsets[end_index][1] > character_span[1]:
			# This is a case where the given answer span is more than one token, and the last token is
			# cut off for some reason, like "split with Luckett and Rober", when the original passage
			# said "split with Luckett and Roberson".  In this case, we'll just keep the end index
			# where it is, and assume the intent was to mark the whole token.
			pass
			# print("Bad labelling or tokenization - end offset doesn't match")
		if token_offsets[end_index][1] != character_span[1]:
			error = True
		return (start_index, end_index), error

	def load_docuements(self, path, summary_path=None, max_documents=0):
		final_data_points = []
		with open(path, "rb") as fin:
			if max_documents > 0:
				data_points = pickle.load(fin)[:max_documents]
			else:
				data_points = pickle.load(fin)
		for data_point in data_points:
			q_tokens = self.vocab.add_and_get_indices(data_point.question_tokens)
			c_tokens = self.vocab.add_and_get_indices(data_point.context_tokens)
			final_data_points.append(Span_Data_Point(q_tokens, c_tokens, data_point.span_indices))
		return final_data_points

	def load_documents_with_candidates(self, path, summary_path=None, max_documents=0):
		final_data_points = []
		with open(path, "rb") as fin:
			if max_documents > 0:
				data_points = pickle.load(fin)[:max_documents]
			else:
				data_points = pickle.load(fin)
		for data_point in data_points:
			q_tokens = self.vocab.add_and_get_indices(data_point.question_tokens)
			c_tokens = self.vocab.add_and_get_indices(data_point.context_tokens)
			candidate_per_question = []
			anonymized_candidates_per_question = []
			correct_answer = data_point.context_tokens[data_point.span_indices[0]:data_point.span_indices[1] + 1]
			anonymized_correct_answer = c_tokens[data_point.span_indices[0]:data_point.span_indices[1] + 1]
			correct_start = data_point.span_indices[0]
			correct_answer_length = data_point.span_indices[1] - data_point.span_indices[0] + 1
			context_length = len(c_tokens)
			for i in range(19):
				## random span of same length as answer, but not correct span
				start_index = random.randint(0, context_length-correct_answer_length+1)
				while start_index == correct_start:
					start_index = random.randint(0, context_length-correct_answer_length+1)
				candidate_per_question.append(data_point.context_tokens[start_index:start_index + correct_answer_length])
				anonymized_candidates_per_question.append(c_tokens[start_index:start_index + correct_answer_length])
			## correct answer:
			candidate_per_question = correct_answer + candidate_per_question
			metrics = []

			for candidate in candidate_per_question:
				self.performance.computeMetrics(candidate, correct_answer)
				metrics.append(1.0-self.performance.bleu1)

			final_data_points.append(Data_Point(q_tokens, [0], anonymized_candidates_per_question, metrics, [], [], [], [] ,c_tokens))
		return final_data_points

	def load_documents_with_candidate_spans(self, path, summary_path=None, max_documents=0):
		final_data_points = []
		with open(path, "rb") as fin:
			if max_documents > 0:
				data_points = pickle.load(fin)[:max_documents]
			else:
				data_points = pickle.load(fin)
		for data_point in data_points:
			q_tokens = self.vocab.add_and_get_indices(data_point.question_tokens)
			c_tokens = self.vocab.add_and_get_indices(data_point.context_tokens)
			candidate_per_question = []
			anonymized_candidates_per_question = []
			correct_answer = data_point.context_tokens[data_point.span_indices[0]:data_point.span_indices[1] + 1]
			anonymized_correct_answer = c_tokens[data_point.span_indices[0]:data_point.span_indices[1] + 1]
			correct_start = data_point.span_indices[0]
			correct_answer_length = data_point.span_indices[1] - data_point.span_indices[0] + 1
			context_length = len(c_tokens)
			for i in range(19):
				## random span of same length as answer, but not correct span
				start_index = random.randint(0, context_length - correct_answer_length)
				while start_index == correct_start:
					start_index = random.randint(0, context_length - correct_answer_length)
				candidate = data_point.context_tokens[start_index:start_index + correct_answer_length]
				candidate_per_question.append(
					data_point.context_tokens[start_index:start_index + correct_answer_length])
				anonymized_candidates_per_question.append([start_index,start_index + correct_answer_length-1])
			## correct answer:
			candidate_per_question = [correct_answer] + candidate_per_question
			anonymized_candidates_per_question =  anonymized_candidates_per_question + [[data_point.span_indices[0],data_point.span_indices[1]]]
			metrics = []

			for candidate in candidate_per_question:
				self.performance.computeMetrics(candidate, [correct_answer])
				metrics.append(1.0 - self.performance.bleu1)

			final_data_points.append(
				Data_Point(q_tokens, [19], anonymized_candidates_per_question, metrics, [], [], [], [], c_tokens))
		return final_data_points

	def pickle_data(self, path, output_path):
		data_points = []
		chunk_length = 20

		with open(path) as data_file:
			dataset = json.load(data_file)['data']
		article_count = 1
		for article in dataset:
			print(article_count)
			#if article_count == 10:
			#	break
			article_count += 1
			for paragraph_json in article['paragraphs']:
				paragraph = paragraph_json["context"]
				tokenized_chunks, tokenized_paragraph = self.tokenize(paragraph, chunk_length=chunk_length, chunk=True)

				for question_answer in paragraph_json['qas']:
					question_text = question_answer["question"].strip().replace("\n", "")
					_, question_tokens = self.tokenize(question_text)
					answer_texts = [answer['text'] for answer in question_answer['answers']]
					span_starts = [answer['answer_start'] for answer in question_answer['answers']]
					span_ends = [start + len(answer) for start, answer in zip(span_starts, answer_texts)]
					token_spans = []
					passage_offsets = [(token.idx, token.idx + len(token.text)) for token in tokenized_paragraph]
					char_spans = zip(span_starts, span_ends)
					for char_span_start, char_span_end in char_spans:
						(span_start, span_end), error = self.char_span_to_token_span(passage_offsets,
																					 (char_span_start, char_span_end))
						## not logging errors
						token_spans.append((span_start, span_end))
					candidate_answers = Counter()
					for span_start, span_end in token_spans:
						candidate_answers[(span_start, span_end)] += 1
					span_start, span_end = candidate_answers.most_common(1)[0][0]

					## convert into normal format
					question_tokens = [token.text for token in question_tokens]
					copy_tokenized_paragraph = [token.text for token in tokenized_paragraph]
					answer_text = copy_tokenized_paragraph[span_start:span_end + 1]

					tokens_covered = 0
					new_context = []
					gold_sentence_index = []
					sentence_bleu = []
					sentence_found = False
					continue_span = False
					last_chunk_idx = len(tokenized_chunks) - 1
					for sent_idx,sent_tokens in enumerate(tokenized_chunks):
						tokens_covered += len(sent_tokens)
						new_context += sent_tokens
						if not sentence_found:
							if sent_idx != last_chunk_idx:
								if span_start < tokens_covered and span_end < tokens_covered:
									gold_sentence_index.append(sent_idx)
									sentence_found = True
									continue_span = False
								elif span_start < tokens_covered:
									gold_sentence_index.append(sent_idx)
									continue_span = True
									# sentence_found = True
								elif span_end < tokens_covered:
									gold_sentence_index.append(sent_idx)
									sentence_found = True
									continue_span = False
								elif continue_span:
									gold_sentence_index.append(sent_idx)
							else:
								#print("special case")
								original_context_len = len(copy_tokenized_paragraph)
								new_span_start = chunk_length - (original_context_len - (chunk_length * sent_idx)) + span_start
								span_start = new_span_start
								span_end = new_span_start + len(answer_text) - 1
								sentence_found = True
								gold_sentence_index.append(sent_idx)
								#print(new_context[span_start: span_end+1])
								#print(answer_text)

						sentence_bleu.append(self.performance.bleu(answer_text, sent_tokens))

					data_points.append(
						Span_Data_Point(question_tokens, tokenized_chunks, [span_start, span_end],sentence_bleu, answer_tokens=answer_text, gold_sentence_index= gold_sentence_index ))
		with open(output_path, "wb") as fout:
			pickle.dump(data_points, fout)


class Vocabulary(object):
    def __init__(self, pad_token='pad', unk='unk', sos='<sos>',eos='<eos>' ):

        self.vocabulary = dict()
        self.id_to_vocab = dict()
        self.pad_token = pad_token
        self.unk = unk
        self.vocabulary[pad_token] = 0
        self.vocabulary[unk] = 1
        self.vocabulary[sos] = 2
        self.vocabulary[eos] = 3

        self.id_to_vocab[0] = pad_token
        self.id_to_vocab[1] = unk
        self.id_to_vocab[2] = sos
        self.id_to_vocab[3] = eos

        self.nertag_to_id = dict()
        self.postag_to_id = dict()
        self.id_to_nertag = dict()
        self.id_to_postag = dict()


    def add_and_get_index(self, word):
        if word in self.vocabulary:
            return self.vocabulary[word]
        else:
            length = len(self.vocabulary)
            self.vocabulary[word] = length
            self.id_to_vocab[length] = word
            return length

    def add_and_get_indices(self, words):
        return [self.add_and_get_index(word) for word in words]

    def get_index(self, word):
        return self.vocabulary.get(word, self.vocabulary[self.unk])

    def get_length(self):
        return len(self.vocabulary)

    def get_word(self,index):
        if index < len(self.id_to_vocab):
            return self.id_to_vocab[index]
        else:
            return ""

    def ner_tag_size(self):
        return len(self.nertag_to_id)

    def pos_tag_size(self):
        return len(self.postag_to_id)

    def add_and_get_indices_NER(self, words):
        return [self.add_and_get_index_NER(str(word)) for word in words]

    def add_and_get_indices_POS(self, words):
        return [self.add_and_get_index_POS(str(word)) for word in words]

    def add_and_get_index_NER(self, word):
        if word in self.nertag_to_id:
            return self.nertag_to_id[word]
        else:
            length = len(self.nertag_to_id)
            self.nertag_to_id[word] = length
            self.id_to_nertag[length] = word
            return length

    def add_and_get_index_POS(self, word):
        if word in self.postag_to_id:
            return self.postag_to_id[word]
        else:
            length = len(self.postag_to_id)
            self.postag_to_id[word] = length
            self.id_to_postag[length] = word
            return length


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("--train_path", type=str, default="../../squad/train-v1.1.json")
	parser.add_argument("--t_output_path", type=str, default="../../squad/train-v1.1-sentwise.pickle")
	parser.add_argument("--valid_path", type=str, default="../../squad/dev-v1.1.json")
	parser.add_argument("--valid_output_path", type=str, default="../../squad/dev-v1.1-sentwise.pickle")
	parser.add_argument("--test_path", type=str, default=None)
	args = parser.parse_args()

	squad_dataloader = SquadDataloader(args)
	squad_dataloader.pickle_data(args.train_path, args.t_output_path)
	squad_dataloader.pickle_data(args.valid_path, args.valid_output_path)
	#data_points = squad_dataloader.load_docuements(args.t_output_path)
