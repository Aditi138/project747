import codecs
import argparse
import os
import glob
from nltk import word_tokenize
from csv import reader
import sys
from utility import start_tags, end_tags, start_tags_with_attributes, Query
from ir_chunker import *
import json

class ChunkRetriever():
	def __init__(self, args):
		self.args = args
		self.chunkRetrieval = Chunking(args)

	def load_data(self):
		reload(sys)
		sys.setdefaultencoding('utf8')
		## assuming every unique id has one summary only
		summaries = {}
		with codecs.open(self.args.summary_file, "r", encoding='utf-8', errors='replace') as fin:
			for line in reader(fin):
				id = line[0]
				summary_tokens = line[3]
				summaries[id] = summary_tokens.split()
		print("Loaded summaries")

		qaps = {}
		with codecs.open(self.args.qap_file, "r") as fin:
			for line in reader(fin):
				id = line[0]
				if id in qaps:
					qaps[id].append(Query(line[5].split(), line[6].split(), line[7].split()))
				else:
					qaps[id] = []
					qaps[id].append(Query(line[5].split(), line[6].split(), line[7].split()))
		print("Loaded question answer pairs")

		documents = {}  # reading documents.csv
		train_ids = []
		test_ids = []
		with codecs.open(self.args.doc_file, "r") as fin:
			index = 0
			for line in reader(fin):

				# print line
				tokens = line
				assert len(tokens) == 10

				if index > 0:
					doc_id = tokens[0]
					set = tokens[1]
					kind = tokens[2]
					start_tag = tokens[8]
					end_tag = tokens[9]
					documents[doc_id] = (set, kind, start_tag, end_tag)
					if set == "train":
						train_ids.append(doc_id)
					if set == "test":
						test_ids.append(doc_id)
				index = index + 1
		print("Loaded documents file")

		if self.args.mode == "test":
			## get K random train and test files
			train_ids = np.array(train_ids)
			test_ids = np.array(test_ids)
			random_train_documents = train_ids[np.random.randint(0, len(train_ids), 30)]
			random_test_documents = test_ids[np.random.randint(0, len(test_ids), 30)]

		document_tokens = []

		train_raw_chunks = []
		valid_raw_chunks = []
		test_raw_chunks = []

		train_questions = []
		valid_questions = []
		test_questions = []

		train_answers = []
		valid_answers = []
		test_answers = []

		train_docids = []
		valid_docids = []
		test_docids = []

		train_chunk_ids = []
		valid_chunk_ids = []
		test_chunk_ids = []

		counter = 0
		chunk_size = self.args.chunk_size
		num_chunks = self.args.num_chunks
		for filename in glob.glob(os.path.join(self.args.input_folder, '*.content')):
			doc_id = os.path.basename(filename).replace(".content", "")
			if doc_id not in documents:
				print "Docuemnt id not found: {0}", doc_id
				exit(0)
			(set, kind, start_tag, end_tag) = documents[doc_id]
			if self.args.mode == "test":
				if doc_id not in random_train_documents and doc_id not in random_test_documents:
					continue
			counter += 1
			if kind == "gutenberg":
				try:
					with codecs.open(self.args.input_folder + doc_id + ".content", "r", encoding='utf-8',
									 errors='replace') as fin:
						data = fin.read()
						data = data.replace('"', '')
						tokenized_data = " ".join(word_tokenize(data))
						start_index = tokenized_data.find(start_tag)
						end_index = tokenized_data.rfind(end_tag, start_index)
						filtered_data = tokenized_data[start_index:end_index]
						if len(filtered_data) == 0:
							print "Error in book extraction: ", filename, start_tag, end_tag
						else:
							print(filename)
							filtered_data = filtered_data.replace(" 's ", " s ")
							document_tokens = filtered_data.split()
				except Exception as error:
					print error
					print "Books for which 'utf-8' doesnt work: ", doc_id
			else:
				try:
					with codecs.open(self.args.input_folder + doc_id + ".content", "r", encoding="utf-8",
									 errors="replace") as fin:
						text = fin.read()
						text = text.replace('"', '')
						script_regex = r"<script.*>.*?</script>|<SCRIPT.*>.*?</SCRIPT>"
						text = re.sub(script_regex, '', text)
						for tag in start_tags_with_attributes:
							my_regex = r'{0}.*=.*?>'.format(tag)
							text = re.sub(my_regex, '', text)
						for tag in end_tags:
							text = text.replace(tag, "")
						for tag in start_tags:
							text = text.replace(tag, "")
						## this step was required for few movies so start tags were changed
						start_tag = start_tag.replace(" S ", " 'S ").replace(" s ", " 's ")
						tokenized_data = " ".join(word_tokenize(text))
						start_index = tokenized_data.find(start_tag)
						if start_index == -1:
							pass
						end_index = tokenized_data.rfind(end_tag, start_index)
						filtered_data = tokenized_data[start_index:end_index]
						if len(filtered_data) == 0:
							print "Error in movie extraction: ", filename, start_tag
						else:
							print(filename)
							filtered_data == filtered_data.replace(" 's ", " s ")
							document_tokens = filtered_data.split()

				except:
					print "Movie for which html extraction doesnt work doesnt work: ", doc_id

			## TODO (Aditi): ner code  +  anonymization

			## filtered_data, document tokens
			questions = []
			answers = []

			for i, query in enumerate(qaps[doc_id]):
				question_tokens = query.get_question_tokens()
				(answer1_tokens, answer2_tokens) = query.get_answer_tokens()
				## TODO (Aditi): ner code  + anonymization
				questions.append(question_tokens)
				answers.append(answer1_tokens)

			## chunking
			chunk_size  = self.args.chunk_size
			num_chunks = self.args.num_chunks
			if self.args.load_summary:
				### Todo: Write chunking code for summary
				pass
			else:
				# Storing both documents and the chunks
				if set == "train":
					combined_reference = [q+a for q,a in zip(questions, answers)]
					extracted, ids = self.chunkRetrieval.retrieve_chunks(document_tokens, combined_reference, chunk_size, num_chunks=num_chunks)
					train_docids.append(doc_id)
					train_questions.append([" ".join(q) for q in questions])
					serialized_chunks = [
						[chunk.get_sentences() for chunk in extracted[i]] for i in
						range(len(extracted))]
					train_raw_chunks.append(serialized_chunks)
					train_chunk_ids.append([str(id) for id in ids])
					train_answers.append([" ".join(a) for a in answers])
				elif set == "valid":
					extracted, ids = self.chunkRetrieval.retrieve_chunks(document_tokens, questions, chunk_size, num_chunks=num_chunks)
					serialized_chunks = [
						[chunk.get_sentences() for chunk in extracted[i]] for i in
						range(len(extracted))]
					valid_docids.append(doc_id)
					valid_questions.append([" ".join(q) for q in questions])
					valid_raw_chunks.append(serialized_chunks)
					valid_chunk_ids.append([str(id) for id in ids])
					valid_answers.append([" ".join(a) for a in answers])
				elif set == "test":
					extracted, ids = self.chunkRetrieval.retrieve_chunks(document_tokens, questions, chunk_size, num_chunks=num_chunks)
					serialized_chunks = [
						[chunk.get_sentences() for chunk in extracted[i]] for i in
						range(len(extracted))]
					test_docids.append(doc_id)
					test_questions.append([" ".join(q) for q in questions])
					test_raw_chunks.append(serialized_chunks)
					test_chunk_ids.append([str(id) for id in ids])
					test_answers.append([" ".join(a) for a in answers])

		myObj = {"id": train_docids, "questions": train_questions, "answers":train_answers, "chunks": train_raw_chunks,
				 "order": train_chunk_ids}
		self.saveChunks(myObj, self.args.output_folder + "train_raw_chunks_{0}_{1}.json".format(num_chunks, chunk_size))

		myObj = {"id": valid_docids, "questions": valid_questions, "answers":valid_answers, "chunks": valid_raw_chunks,
				 "order": valid_chunk_ids}
		self.saveChunks(myObj, self.args.output_folder + "valid_raw_chunks_{0}_{1}.json".format(num_chunks, chunk_size))

		myObj = {"id": test_docids, "questions": test_questions, "answers":test_answers, "chunks": test_raw_chunks, "order": test_chunk_ids};
		self.saveChunks(myObj, self.args.output_folder + "test_raw_chunks_{0}_{1}.json".format(num_chunks, chunk_size))

	def saveChunks(self, chunks, output_path):
			with open(output_path, "w") as fout:
				json.dump(chunks, fout)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_folder", type=str)
	parser.add_argument("--doc_file", type=str)
	parser.add_argument("--summary_file", type=str)
	parser.add_argument("--qap_file", type=str)
	parser.add_argument("--output_folder", type=str)
	parser.add_argument("--load_summary", default=False, action="store_true")
	parser.add_argument("--ir_model", type=str)
	parser.add_argument("--mode", type=str, default="train")
	parser.add_argument("--num_chunks", type=int, default=20)
	parser.add_argument("--chunk_size", type=int, default=200)
	args = parser.parse_args()
	dataloader = ChunkRetriever(args)
	dataloader.load_data()


if __name__ == '__main__':
    main()