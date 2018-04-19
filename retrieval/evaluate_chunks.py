from __future__ import division
import json
import random
import argparse
import numpy as np


def load_chunks(chunk_path):
	with open(chunk_path) as file:
		information = json.load(file)

	chunks = information["chunks"]
	questions = information["questions"]
	locations = information["order"]
	answers = information["answers"]
	return chunks, questions, answers, locations


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--chunk_path", type=str,
						default="/home/michiel/Downloads/train_raw_chunks.pickle")
	parser.add_argument("--num_questions", type=int, default=10)
	parser.add_argument("--seed", type=int, default=0)

	args = parser.parse_args()

	chunk_numbers = [1, 5, 10]

	chunks, questions, answers, locations = load_chunks(args.chunk_path)

	random.seed(8)

	document_ids = random.sample(range(len(chunks) - 1), args.num_questions)
	question_ids = [random.randint(0, len(questions[document_ids[i]]) - 1)
					for i in range(args.num_questions)]

	scores = dict()
	for number_chunk in chunk_numbers:
		scores[number_chunk] = 0

	for i in range(args.num_questions):

		doc_id = document_ids[i]

		question_chunks = chunks[document_ids[i]][question_ids[i]]
		question_chunk_locations = locations[document_ids[i]][question_ids[i]]
		question_chunk_locations = question_chunk_locations.strip("[]").split()
		question_chunk_locations = np.array([int(location) for location in question_chunk_locations])
		question_chunk_order = np.argsort(question_chunk_locations)

		for number_chunk in chunk_numbers:

			current_chunks = question_chunks[:number_chunk]
			current_locations = question_chunk_locations[:number_chunk]
			current_order = np.argsort(current_locations)
			ordered_chunks = [question_chunks[current_order[k]] for k in range(len(current_chunks))]

			for j in range(number_chunk):
				## each chunk will be a list of sentences
				print("\n".join([" ".join(sentence) for sentence in ordered_chunks[j]]))
				print("*"*50 + "\n")
			print("Question: {0}".format(questions[doc_id][question_ids[i]]))
			print("True Answer: {0}".format(answers[doc_id][question_ids[i]]))
			print("\n")

			while True:
				answerable = raw_input("Is the question answerable? y/n \n")
				if answerable == "y":
					scores[number_chunk] += 1
					break
				elif answerable == "n":
					break
				else:
					print("Please type y/n only")

	for number_chunk in chunk_numbers:
		print("Accuracy for {0} chunks: {1}".format(
			number_chunk, scores[number_chunk] / args.num_questions))