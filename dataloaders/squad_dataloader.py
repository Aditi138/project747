import json
import os
import spacy

class SquadDataloader():
	def __init__(self, args):
		pass

	def load_documents(self, path, summary_path=None, max_documents=0):
		self.squad = json.load(path)['data']
		for article in self.squad:
			for paragraph in article['paragraphs']:
				# each question is an example
				for qa in paragraph['qas']:
					question = qa['question']
					answers = (a['text'] for a in qa['answers'])
					context = paragraph['context']


