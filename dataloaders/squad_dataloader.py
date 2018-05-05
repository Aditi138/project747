import json
import os
from data import Span_Data_Point, Data_Point, Question, Article
from collections import Counter, defaultdict
import pickle
import argparse
import random, math
import re


class SquadDataloader():
    def __init__(self, args):
        self.vocab = Vocabulary()

    def load_documents_with_paragraphs(self, qap_path, paragraph_path, max_documents=0, num_paragraphs = 4):
        with open(qap_path, "rb") as fin:
            if max_documents > 0:
                data_points = pickle.load(fin)[:max_documents]
            else:
                data_points = pickle.load(fin)
        with open(paragraph_path, "rb") as fin:
            articles = pickle.load(fin)
        for e, data in enumerate(data_points):
            data_points[e].question_tokens = self.vocab.add_and_get_indices(data.question_tokens)
            data_points[e].top_paragraph_ids = data.top_paragraph_ids[:num_paragraphs]
        for key in articles.keys():
            paragraphs = articles[key]
            for e, p in enumerate(paragraphs):
                paragraphs[e] = self.vocab.add_and_get_indices(p)
            articles[key] = paragraphs
        return data_points, articles

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
