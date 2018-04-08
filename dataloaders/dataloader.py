import codecs
import argparse
import os
import glob
from csv import reader
import sys
import re
try:
    import cPickle as pickle
except:
    import pickle
import io
import random
import numpy as np
import math
import random
# import spacy
from nltk import word_tokenize
from data import Document, Query, Summary
from utility import start_tags, end_tags, start_tags_with_attributes


class DataLoader():
    def __init__(self, args):

        # Actually define args here
        # self.x = args.x
        self.vocab = Vocabulary()

    # This function loads raw documents, summaries and queries, processes them, stores them in document class and finally saves to a pickle
    def process_data(self, input_folder, summary_path, qap_path, document_path, pickle_folder, small_number=-1, summary_only=False):

        # # Takes time to load so only do this inside function rather than in constructor
        # self.nlp =spacy.load('en_core_web_md', disable= ["tagger", "parser"])

        # Here we load files that contain the summaries, questions, answers and information about the documents
        # Not the documents themselves
        # assuming every unique id has one summary only
        summaries = {}
        with codecs.open(summary_path, "r", encoding='utf-8', errors='replace') as fin:
            for line in reader(fin):
                id = line[0]
                summary_tokens = line[3]
                summaries[id] = summary_tokens.split()
        print("Loaded summaries")
        qaps = {}
        with codecs.open(qap_path, "r") as fin:
            for line in reader(fin):
                id = line[0]
                if id in qaps:
                    qaps[id].append(
                        Query(line[5].split(), line[6].split(), line[7].split()))
                else:
                    qaps[id] = []
                    qaps[id].append(
                        Query(line[5].split(), line[6].split(), line[7].split()))
        print("Loaded question answer pairs")
        documents = {}
        with codecs.open(document_path, "r") as fin:
            index = 0
            for line in reader(fin):

                tokens = line
                assert len(tokens) == 10

                if index > 0:
                    doc_id = tokens[0]
                    set = tokens[1]
                    kind = tokens[2]
                    start_tag = tokens[8]
                    end_tag = tokens[9]
                    documents[doc_id] = (set, kind, start_tag, end_tag)

                index = index + 1

        train_docs = []
        valid_docs = []
        test_docs = []

        # In case of creation of small test dataset
        if small_number > 0:
            small_docs = []

        # Here we load documents, tokenize them, and create Document class instances (or Summary instances in case of just summaries)
        for filename in glob.glob(os.path.join(input_folder, '*.content')):
            doc_id = os.path.basename(filename).replace(".content", "")

            try:
                (set, kind, _, _) = documents[doc_id]
            except KeyError:
                print("Document id not found: {0}".format(doc_id))
                exit(0)

            if summary_only:
                doc = process_summary(doc_id)
            else:
                doc = process_document(doc_id)

            # If testing, add to test list, pickle and return when sufficient documents retrieved
            if small_number > 0:
                small_docs.append(doc)
                if len(small_docs) == small_number:
                    with open(pickle_folder + "small.pickle", "wb") as fout:
                        pickle.dump(small_docs, fout)
                    return

            else:
                if set == "train":
                    train_docs.append(doc)
                elif set == "valid":
                    valid_docs.append(doc)
                else:
                    test_docs.append(doc)

        with open(pickle_folder + "train.pickle", "wb") as fout:
            pickle.dump(train_docs, fout)
        with open(pickle_folder + "validate.pickle", "wb") as fout:
            pickle.dump(valid_docs, fout)
        with open(pickle_folder + "test.pickle", "wb") as fout:
            pickle.dump(test_docs, fout)

        def process_document(doc_id):
            (set, kind, start_tag, end_tag) = documents[doc_id]
            if kind == "gutenberg":
                try:
                    with codecs.open(input_folder + doc_id + ".content", "r", encoding='utf-8', errors='replace') as fin:
                        data = fin.read()
                        data = data.replace('"', '')
                        tokenized_data = " ".join(word_tokenize(data))
                        start_index = tokenized_data.find(start_tag)
                        end_index = tokenized_data.rfind(end_tag, start_index)
                        filtered_data = tokenized_data[start_index:end_index]
                        if len(filtered_data) == 0:
                            print("Error in book extraction: ",
                                  filename, start_tag, end_tag)
                        else:
                            print(filename)
                            filtered_data = filtered_data.replace(
                                " 's ", " s ")
                            document_tokens = word_tokenize(filtered_data)

                except Exception as error:
                    print(error)
                    print("Books for which 'utf-8' doesnt work: ", doc_id)
            else:
                try:
                    # Here we remove some annotation that is unique to movie scripts
                    with codecs.open(input_folder + doc_id + ".content", "r", encoding="utf-8",
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
                        start_tag = start_tag.replace(
                            " S ", " 'S ").replace(" s ", " 's ")
                        tokenized_data = " ".join(word_tokenize(text))
                        start_index = tokenized_data.find(start_tag)
                        if start_index == -1:
                            pass
                        end_index = tokenized_data.rfind(end_tag, start_index)
                        filtered_data = tokenized_data[start_index:end_index]
                        if len(filtered_data) == 0:
                            print("Error in movie extraction: ",
                                  filename, start_tag)
                        else:
                            print(filename)
                            filtered_data == filtered_data.replace(
                                " 's ", " s ")
                            document_tokens = word_tokenize(filtered_data)

                except Exception as error:
                    print(error)
                    print(
                        "Movie for which html extraction doesnt work doesnt work: ", doc_id)

            document = Document(
                doc_id, set, kind, document_tokens, summaries[doc_id], qaps[doc_id])

            return document

        def process_summary(doc_id):
            (set, kind, _, _) = documents[doc_id]
            document = Summary(doc_id, set, kind,
                               summaries[doc_id], qaps[doc_id])
            return document

    def load_documents(self, path, summary_only=False):
        with open(path, "r") as fin:
            documents = pickle.load(fin)
        for document in documents:
            document.summary_tokens = self.vocab.add_and_get_indices(document.summary_tokens)
            if not summary_only:
                document.document_tokens = self.vocab.add_and_get_indices(document.document_tokens)
            for query in document.queries:
                query.question_tokens=self.vocab.add_and_get_indices(query.question_tokens)
                query.answer1_tokens=self.vocab.add_and_get_indices(query.answer1_tokens)
                query.answer2_tokens=self.vocab.add_and_get_indices(query.answer2_tokens)
        return documents



class Vocabulary(object):
    def __init__(self, pad_token='pad', unk='unk'):

        self.vocabulary = dict()
        self.inverse_vocabulary = dict()
        self.pad_token = pad_token
        self.unk = unk
        self.vocabulary[pad_token] = 0
        self.vocabulary[unk] = 1

    def add_and_get_index(self, word):
        if word in self.vocabulary:
            return self.vocabulary[word]
        else:
            length = len(self.vocabulary)
            self.vocabulary[word] = length
            self.inverse_vocabulary[length] = word
            return length

    def add_and_get_indices(self, words):
        return [self.add_and_get_index(word) for word in words]

    def get_index(self, word):
        return self.vocabulary.get(word, self.vocabulary[self.unk])

    def get_length(self):
        return len(self.vocabulary)
