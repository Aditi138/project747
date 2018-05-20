import numpy as np
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.summarization.bm25 import BM25
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel


class Chunk(object):
	def __init__(self, chunk_text, sentence_boundaries):
		self.tokens = chunk_text.split()
		self.sentence_boundaries = sentence_boundaries
		self.num_sentences = len(sentence_boundaries)


	def get_sentences(self):
		sentences = []
		for i in range(self.num_sentences - 1):
			sentences.append(self.tokens[self.sentence_boundaries[i]:self.sentence_boundaries[i+1]])
		sentences.append(self.tokens[self.sentence_boundaries[-1]:])
		return sentences

	def serialize_sentence_boundaries(self):
		return

class Chunking(object):
	def __init__(self, args):
		self.nlp =  spacy.load('en')
		self.args = args
		self.stop_words = list(stopwords.words('english'))
		self.lemmatizer = WordNetLemmatizer()


	def improve_sentence_splitting(self, original_sentences, maximum_sentence_length):
		## TODO : One person's dialogue is broken into multiple sentences. We may have to write a dedicated dialogue parser for scripts
		bad_counter = 0
		split_sentences = []
		## split sentences by characters that spacey cannot identify as sentence demarkers [;,-] for sentences longer than maximum_sentence_length
		for e, sent in enumerate(original_sentences):
			if len(sent) > maximum_sentence_length:
				phrases = filter(None, re.split("[;:\-]+", " ".join(sent)))
				for p in phrases:
					if len(p.split()) != 0:
						split_sentences.append(p.split())
			else:
				if len(sent) != 0:
					split_sentences.append(sent)
		## join sentences by characters that spacey unnecesarily splits a sentence on [-]
		## if the previous word is all CAPS, join with the next sentence (supposed dialogue)
		e = 0
		joint_sentences = []
		while e < len(split_sentences)-1:
			if split_sentences[e][-1] == u'-' or split_sentences[e+1][0] == u'-' or " ".join(split_sentences[e]).isupper():
				joint_sentences.append(list(split_sentences[e]) + list(split_sentences[e+1]))
				e += 2
			elif e == len(split_sentences) - 2:
				## handle corner case of final two sentences
				joint_sentences.append(list(split_sentences[e]))
				joint_sentences.append(list(split_sentences[e+1]))
				e += 1
			else:
				joint_sentences.append(list(split_sentences[e]))
				e += 1

		## split the final set of sentences that are still very long into maximum_sentence_length size
		final_sentences = []
		for s in joint_sentences:
			if len(s) > maximum_sentence_length:
				splits = len(s)/maximum_sentence_length
				remaining = len(s)%maximum_sentence_length
				for i in range(splits):
					final_sentences.append(s[i*maximum_sentence_length:(i+1)*maximum_sentence_length])
				if remaining != 0:
					final_sentences.append(s[-remaining:])
				bad_counter += (splits + 1)
			else:
				final_sentences.append(s)
		return bad_counter, final_sentences

	def retrieve_chunks(self, context, references, chunk_length=200, num_chunks=1):

		joint_context = " ".join(context)
		joint_references = []
		for r in references:
			joint_references.append(" ".join(r))
		## break the document down to 4 parts and send to spacy sentence splitter, then combine while still handling sentence split on boundary
		if len(context) > 100000:
			q1 = len(context) / 4
			first_quarter = context[0:q1]
			second_quarter = context[q1:q1 * 2]
			third_quarter = context[q1 * 2:q1 * 3]
			fourth_quarter = context[q1 * 3:]
			context_tokens1 = self.nlp(" ".join(first_quarter))
			context_tokens2 = self.nlp(" ".join(second_quarter))
			context_tokens3 = self.nlp(" ".join(third_quarter))
			context_tokens4 = self.nlp(" ".join(fourth_quarter))
			sentences1 = [[token.string.strip() for token in s] for s in list(context_tokens1.sents)]
			sentences2 = [[token.string.strip() for token in s] for s in list(context_tokens2.sents)]
			sentences3 = [[token.string.strip() for token in s] for s in list(context_tokens3.sents)]
			sentences4 = [[token.string.strip() for token in s] for s in list(context_tokens4.sents)]
			sentences = sentences1[:-1] + [sentences1[-1] + sentences2[0]] + \
						sentences2[1:-1] + [sentences2[-1] + sentences3[0]] + \
						sentences3[1:-1] + [sentences3[-1] + sentences4[0]] + \
						sentences4[1:]
		else:
			context_tokens = self.nlp(joint_context)
			sentences = list(context_tokens.sents)
			sentences = [[token.string.strip() for token in s] for s in sentences]

		chunk_storage = []
		sentence_boundaries_storage = []

		print("Maximum sentence length before improvement: {0}".format(max([len(s) for s in sentences])))
		print("Minimum sentence length before improvement: {0}".format(min([len(s) for s in sentences])))
		## function to improve dialogue sentence splitting: sentences that are split on hyphen are rejoint, sentences are split on ; and over long sentences are chunked
		bad_counter, sentences = self.improve_sentence_splitting(sentences, 50)
		print("Maximum sentence length after improvement: {0}".format(max([len(s) for s in sentences])))
		print("Minimum sentence length after improvement: {0}".format(min([len(s) for s in sentences])))
		print("Total Number of sentences: {0}".format(len(sentences)))
		print("Proportion (in %) of sentences roughly splits: {0:.3f}".format(100.0*float(bad_counter)/len(sentences)))

		e = 0
		while e < len(sentences):
			previous_size = 0
			current_chunk_size = 0
			current_chunk = []
			sentence_boundaries = []
			while e < len(sentences) and current_chunk_size < chunk_length:
				current_chunk += sentences[e]
				previous_size = current_chunk_size
				current_chunk_size += len(sentences[e])
				sentence_boundaries.append(previous_size)
				e += 1
			## guard against previous size being zero, guard against sentence size >= chunk_size, gaurd against e-=1 infinite loop
			if abs(chunk_length - previous_size) < abs(current_chunk_size - chunk_length) and e != len(sentences):
				current_chunk = current_chunk[:previous_size]
				sentence_boundaries = sentence_boundaries[:-1]
				## restart from the previous chunk in this case
				e -= 1
				if len(current_chunk) > 0:
					chunk_storage.append(" ".join(current_chunk))
					sentence_boundaries_storage.append(sentence_boundaries)
			else:
				## if out of sentences, use the last chunk as is
				if len(current_chunk) > 0:
					chunk_storage.append(" ".join(current_chunk))
					sentence_boundaries_storage.append(sentence_boundaries)

		print("maximum chunk size: {0}".format(max([len(chunk.split()) for chunk in chunk_storage])))
		print("minimum chunk size: {0}".format(min([len(chunk.split()) for chunk in chunk_storage])))
		print("Total Number of chunks: {0}\n\n".format(len(chunk_storage)))

		## checking if all chunking was done correctly
		# print(" ".join(chunk_storage))
		# print(joint_context)

		top_chunks = []
		top_chunks_ids = []
		top_chunk_scores = []
		gold_chunk_id = []

		if self.args.ir_model == "tfidf":
			length = len(chunk_storage)
			## append queries to the end of the vector
			for reference in joint_references:
				chunk_storage.append(reference)
			vectorizer = CountVectorizer(preprocessor=self.lemmatizer.lemmatize, stop_words = self.stop_words, ngram_range = (1,2))
			transformer = TfidfTransformer(sublinear_tf = True)
			counts = vectorizer.fit_transform(chunk_storage)
			tfidf = transformer.fit_transform(counts)
			chunk_docs = tfidf[0:length]
			reference_docs = tfidf[length:]
			related_docs_indices = linear_kernel(reference_docs, chunk_docs).argsort()[:, -num_chunks:]
			related_docs_scores = np.sort(linear_kernel(reference_docs, chunk_docs))[:, -num_chunks:]
			for idx in range(len(references)):
				chunks_per_ref = []
				doc_ids = related_docs_indices[idx][::-1]
				doc_scores = related_docs_scores[idx][::-1]
				gold_chunk = doc_ids[0]
				doc_scores = doc_scores[doc_ids.argsort()]
				doc_ids = sorted(doc_ids)
				gold_chunk_id.append(doc_ids.index(gold_chunk))
				for doc_id in range(len(doc_ids)):
					chunks_per_ref.append(Chunk(chunk_storage[doc_ids[doc_id]] , sentence_boundaries_storage[doc_ids[doc_id]]))
				top_chunks.append(chunks_per_ref)
				top_chunks_ids.append(doc_ids)
				top_chunk_scores.append(doc_scores)
		elif self.args.ir_model == "bm25":
			## bm25 standard parameters
			# PARAM_K1 = 1.5
			# PARAM_B = 0.75
			# EPSILON = 0.25
			## remove stop words, lowerase and lemmatize from chunks
			lemmatized_chunk_storage = []
			for e, chunk in enumerate(chunk_storage):
				lemmatized_chunk_storage.append([self.lemmatizer.lemmatize(w.lower()) for w in chunk.split() if not w.lower() in self.stop_words])
			lemmatized_references = []
			for e, reference in enumerate(references):
				lemmatized_references.append([self.lemmatizer.lemmatize(w.lower()) for w in reference if w.lower() not in self.stop_words])
			bm25_object = BM25(lemmatized_chunk_storage)
			average_idf = sum(map(lambda k: float(bm25_object.idf[k]), bm25_object.idf.keys())) / len(bm25_object.idf.keys())
			for idx in range(len(lemmatized_references)):
				related_docs_indices = np.argsort(bm25_object.get_scores(lemmatized_references[idx], average_idf))[-num_chunks:]
				chunks_per_ref = []
				doc_ids = related_docs_indices[::-1]
				for doc_id in range(len(doc_ids)):
					chunks_per_ref.append(Chunk(chunk_storage[doc_ids[doc_id]], sentence_boundaries_storage[doc_ids[doc_id]]))
				top_chunks.append(chunks_per_ref)
				top_chunks_ids.append(doc_ids)
			return top_chunks, top_chunks_ids
		elif self.args.ir_model == "language_model":
			## TODO: Ponte Croft LM based retrieval
			pass
		elif self.args.ir_model == "sentvec":
			top_chunks_ids = []
			top_chunks = []
			## TODO: Sanjeev arora model for sentence similarity aplied to chunks
			pass
		elif self.args.ir_model == "srl":
			##TODO: Semantic role labelling based matching
			pass

		## test sentence boundaries
		# some_chunks = top_chunks[0]
		# for chunk in some_chunks:
		# 	print "\n".join([" ".join(sent) for sent in chunk.get_sentences()])
		# 	print "*"*50

		return top_chunks, top_chunks_ids, top_chunk_scores, gold_chunk_id