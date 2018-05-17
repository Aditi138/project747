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

import sys
import spacy
from nltk import word_tokenize
from data import Document, Query, Data_Point, Elmo_Data_Point
from utility import start_tags, end_tags, start_tags_with_attributes, pad_seq, view_data_point, pad_seq_elmo
import random
import numpy as np
from collections import defaultdict
from test_metrics import Performance
from multiprocessing import Pool
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel
from copy import  deepcopy

global vocab


def view_batch(batch, vocab):
    queries = batch['queries']
    ner_queries = batch['q_ner']
    q = []
    a = []
    q_ner = []
    a_ner = []
    for index, question_tokens in enumerate(queries):
        q.append(" ".join([vocab.get_word(id) for id in question_tokens]) + "\n")
        q_ner.append(" ".join([vocab.id_to_nertag[id] for id in ner_queries[index]]) + "\n")
    batch_candidates = batch["candidates"]
    batch_answer_indices = batch['answer_indices']

    for index, answer_tokens in enumerate(batch_candidates['answers']):
        gold_answer_tokens = answer_tokens[batch_answer_indices[index]]
        a.append(" ".join([vocab.get_word(id) for id in gold_answer_tokens]) + "\n")
        a_ner.append(" ".join(
            [vocab.id_to_nertag[id] for id in batch_candidates['ner'][index][batch_answer_indices[index]]]) + "\n")
    for index in range(len(q)):
        print(q[index] + " " + q_ner[index] + " " + a[index] + " " + a_ner[index] + "\n")


def make_bucket_batches(data, batch_size, vocab):
    # Data are bucketed according to the length of the first item in the data_collections.
    buckets = defaultdict(list)

    for data_item in data:
        src = data_item.question_tokens
        buckets[len(src)].append(data_item)

    batch_data = []
    batches = []
    # np.random.seed(2)
    for src_len in buckets:
        bucket = buckets[src_len]
        #np.random.shuffle(bucket)

        num_batches = int(np.ceil(len(bucket) * 1.0 / batch_size))
        for i in range(num_batches):
            cur_batch_size = batch_size if i < num_batches - 1 else len(bucket) - batch_size * i
            begin_index = i * batch_size
            end_index = begin_index + cur_batch_size
            batch_data = list(bucket[begin_index:end_index])
            batch = create_single_batch_elmo(batch_data)
            # view_batch(batch,vocab)
            batches.append(batch)

    np.random.shuffle(batches)
    return batches


def create_single_batch(batch_data):
    batch_query_lengths = np.array([len(data_point.question_tokens) for data_point in batch_data])
    maximum_query_length = max(batch_query_lengths)
    query_length_mask = np.array([[int(x < batch_query_lengths[i])
                                   for x in range(maximum_query_length)] for i in range(len(batch_data))])

    queries = np.array([pad_seq(data_point.question_tokens, maximum_query_length)
                        for data_point in batch_data])

    batch_context_lengths = np.array([len(data_point.context_tokens) for data_point in batch_data])
    maximum_context_length = max(batch_context_lengths)
    contexts = np.array([pad_seq(data_point.context_tokens, maximum_context_length) for data_point in batch_data])
    batch_context_mask = np.array([[int(x < batch_context_lengths[i])
                                    for x in range(maximum_context_length)] for i in range(len(batch_data))])

    queries_ner = np.array([pad_seq(data_point.ner_for_question, maximum_query_length)
                            for data_point in batch_data])

    queries_pos = np.array([pad_seq(data_point.pos_for_question, maximum_query_length)
                            for data_point in batch_data])

    candidate_information = {}
    batch_candidate_answers_padded = []
    batch_candidate_answer_lengths = []
    batch_answer_indices = []
    batch_candidates_ner = []
    batch_candidates_pos = []
    batch_candidate_answer_length_mask = []
    batch_metrics = np.array([data_point.metrics for data_point in batch_data])

    for index, data_point in enumerate(batch_data):
        # create a batch mask over candidates similar to the one over different questions
        candidates = data_point.candidates
        candidates_ner = data_point.ner_for_candidates
        candidates_pos = data_point.pos_for_candidates

        candidate_answer_lengths = [len(answer) for answer in candidates]
        max_candidate_length = max(candidate_answer_lengths)
        candidate_padded_answers = np.array([pad_seq(answer, max_candidate_length) for answer in candidates])
        candidate_padded_answers_ner = np.array([pad_seq(answer, max_candidate_length) for answer in candidates_ner])
        candidate_padded_answers_pos = np.array([pad_seq(answer, max_candidate_length) for answer in candidates_pos])
        candidate_answer_length_mask = np.array([[int(x < candidate_answer_lengths[i])
                                                  for x in range(max_candidate_length)] for i in
                                                 range(len(candidates))])

        batch_candidate_answers_padded.append(candidate_padded_answers)
        batch_candidate_answer_lengths.append(candidate_answer_lengths)
        batch_candidates_ner.append(candidate_padded_answers_ner)
        batch_candidates_pos.append(candidate_padded_answers_pos)
        batch_candidate_answer_length_mask.append(candidate_answer_length_mask)

        batch_answer_indices.append(data_point.answer_indices[0])

    candidate_information["answers"] = batch_candidate_answers_padded
    candidate_information["anslengths"] = batch_candidate_answer_lengths
    candidate_information["ner"] = batch_candidates_ner
    candidate_information["pos"] = batch_candidates_pos
    candidate_information["mask"] = batch_candidate_answer_length_mask

    batch = {}
    batch['queries'] = queries
    batch['contexts'] = contexts
    batch['context_mask'] = batch_context_mask
    batch['q_ner'] = queries_ner
    batch['q_mask'] = query_length_mask
    batch['q_pos'] = queries_pos
    batch['answer_indices'] = batch_answer_indices
    batch['qlengths'] = batch_query_lengths
    batch['clengths'] = batch_context_lengths
    batch["candidates"] = candidate_information
    batch["metrics"] = batch_metrics

    return batch


def create_single_batch_elmo(batch_data):
    doc_ids = [data_point.doc_id for data_point in batch_data]
    chunk_indices = [data_point.chunk_indices for data_point in batch_data]
    batch_query_lengths = [len(data_point.question_tokens) for data_point in batch_data]
    maximum_query_length = max(batch_query_lengths)
    # query_length_mask = np.array([[int(x < batch_query_lengths[i])
    #                              for x in range(maximum_query_length)] for i in range(len(batch_data))])

    queries_embed = [data_point.question_embed for data_point in batch_data]
    question_tokens = [data_point.question_tokens for data_point in batch_data]

    candidate_information = {}
    batch_candidate_answer_lengths = []
    batch_answer_indices = []
    batch_candidate_answer_length_mask = []

    for index, data_point in enumerate(batch_data):
        # create a batch mask over candidates similar to the one over different questions
        candidates = data_point.candidates
        # candidates_embed = data_point.candidates_embed

        candidate_answer_lengths = [len(answer) for answer in candidates]
        max_candidate_length = max(candidate_answer_lengths)
        candidate_answer_length_mask = np.array([[int(x < candidate_answer_lengths[i])
                                                  for x in range(max_candidate_length)] for i in
                                                 range(len(candidates))])

        batch_candidate_answer_lengths.append(candidate_answer_lengths)
        batch_candidate_answer_length_mask.append(candidate_answer_length_mask)

        batch_answer_indices.append(data_point.answer_indices[0])

    candidate_information["anslengths"] = batch_candidate_answer_lengths
    candidate_information["mask"] = batch_candidate_answer_length_mask

    batch = {}
    batch['doc_ids'] = doc_ids
    batch['q_tokens'] = question_tokens
    batch['chunk_indices'] = chunk_indices
    batch['q_embed'] = queries_embed
    batch['answer_indices'] = batch_answer_indices
    batch['qlengths'] = batch_query_lengths
    batch["candidates"] = candidate_information
    return batch


def create_batches(data, batch_size, job_size, vocab):
    vocab = vocab
    job_pool = Pool(job_size)
    end_index = 0
    # shuffle the actual data
    temp_data = list(data)
    #random.shuffle(temp_data)

    # question_lengths = [len(data_point.question_tokens) for data_point in temp_data]
    # # within batch, sort data by length
    # sorted_data = zip(question_lengths, temp_data)
    # sorted_data.sort(reverse=True)

    # question_lengths, temp_data = zip(*sorted_data)

    # Calculate number of batches
    number_batches = len(temp_data) // batch_size + \
                     int((len(temp_data) % batch_size) > 0)

    # Multi-processing
    job_data = []

    for j in range(number_batches - 1):
        begin_index, end_index = j * batch_size, (j + 1) * batch_size
        job_data.append(list(temp_data[begin_index:end_index]))
    # batches = job_pool.map(create_single_batch, job_data)
    batches = job_pool.map(create_single_batch_elmo, job_data)

    job_pool.close()
    job_pool.join()

    # for j in range(number_batches - 1):
    #     begin_index, end_index = j * batch_size, (j + 1) * batch_size
    #     batch_data = list(data[begin_index:end_index])
    #     batch = create_single_batch(batch_data,vocab)
    #     view_batch(batch, vocab)
    # self.view_batch(batch)
    # batches.append(batch)

    # view_batch(batches[1], vocab)
    batch_data = list(temp_data[end_index:])
    # batches.append(create_single_batch(batch_data))
    batches.append(create_single_batch_elmo(batch_data))

    print("Created batches of batch_size {0} and number {1}".format(batch_size, number_batches))
    return batches


class DataLoader():
    def __init__(self, args):

        # Actually define args here
        # self.x = args.x
        self.vocab = Vocabulary()
        self.performance = Performance(args)
        self.args = args
        self.pretrain_embedding = None
        self.nlp = spacy.load('en')
        self.stop_words = list(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    # This function loads raw documents, summaries and queries, processes them, stores them in document class and finally saves to a pickle
    def process_data(self, input_folder, summary_path, qap_path, document_path, pickle_folder, small_number=-1,
                     summary_only=False, interval=50):
        reload(sys)
        sys.setdefaultencoding('utf8')

        # # Takes time to load so only do this inside function rather than in constructor
        # self.nlp =spacy.load('en_core_web_md', disable= ["tagger", "parser"])

        # Here we load files that contain the summaries, questions, answers and information about the documents
        # Not the documents themselves
        # assuming every unique id has one summary only

        to_anonymize = ["GPE", "PERSON", "ORG", "LOC"]

        def _getNER(string_data, entity_dict, other_dict):
            doc = self.nlp(string_data)
            data = string_data.split()
            NE_data = ""
            start_pos = 0
            for ents in doc.ents:
                start = ents.start_char
                end = ents.end_char
                label = ents.label_
                tokens = ents.text
                key = tokens.lower()
                if label in to_anonymize:
                    if key not in data:
                        if key not in entity_dict:
                            entity_dict[key] = "@ent" + str(len(entity_dict)) + "~ner:" + label
                        NE_data += string_data[start_pos:start] + entity_dict[key] + " "
                        start_pos = end + 1
                else:
                    other_dict[key] = tokens + "~ner:" + label
                    NE_data += string_data[start_pos:start] + tokens + "~ner:" + label + " "
                    start_pos = end + 1

            NE_data += string_data[start_pos:]
            return NE_data.split()

        summaries = {}
        with codecs.open(summary_path, "r", encoding='utf-8', errors='replace') as fin:
            first = True
            for line in reader(fin):
                if first:
                    first = False
                    continue
                id = line[0]
                summary_tokens = line[2]
                ner_summary, pos_summary, tokens = self.getNER(line[2])
                summaries[id] = (tokens, ner_summary, pos_summary)
        print("Loaded summaries")
        qaps = {}

        candidates_per_doc = defaultdict(list)
        ner_candidates_per_doc = defaultdict(list)
        pos_candidates_per_doc = defaultdict(list)
        count = 0
        with codecs.open(qap_path, "r") as fin:
            first = True
            for line in reader(fin):

                if first:
                    first = False
                    continue
                id = line[0]

                if id in qaps:

                    ner_answer, pos_answer, tokens = self.getNER(line[3])
                    ner_candidates_per_doc[id].append(ner_answer)
                    pos_candidates_per_doc[id].append(pos_answer)
                    candidates_per_doc[id].append(tokens)

                    ner_answer, pos_answer, tokens = self.getNER(line[4])
                    ner_candidates_per_doc[id].append(ner_answer)
                    pos_candidates_per_doc[id].append(pos_answer)
                    candidates_per_doc[id].append(tokens)

                    indices = [candidate_index, candidate_index + 1]
                    candidate_index += 2

                    ner_question, pos_question, tokens = self.getNER(line[2])
                    qaps[id].append(
                        Query(tokens, ner_question, pos_question, indices))
                else:
                    # print(id)
                    qaps[id] = []
                    candidates_per_doc[id] = []
                    candidate_index = 0

                    ner_answer, pos_answer, tokens = self.getNER(line[3])
                    ner_candidates_per_doc[id].append(ner_answer)
                    pos_candidates_per_doc[id].append(pos_answer)
                    candidates_per_doc[id].append(tokens)

                    ner_answer, pos_answer, tokens = self.getNER(line[4])
                    ner_candidates_per_doc[id].append(ner_answer)
                    pos_candidates_per_doc[id].append(pos_answer)
                    candidates_per_doc[id].append(tokens)

                    indices = [candidate_index, candidate_index + 1]
                    candidate_index += 2

                    ner_question, pos_question, tokens = self.getNER(line[2])
                    qaps[id].append(
                        Query(tokens, ner_question, pos_question, indices))

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

        # Create lists of document objects for the summaries
        train_summaries = []
        valid_summaries = []
        test_summaries = []

        if small_number > 0:
            small_summaries = []

        for doc_id in documents:
            set, kind, _, _ = documents[doc_id]
            tokens, ner_summary, pos_summary = summaries[doc_id]
            summary = Document(doc_id, set, kind, tokens, qaps[doc_id], {}, {}, candidates_per_doc[doc_id],
                               ner_candidates_per_doc[doc_id], pos_candidates_per_doc[doc_id], ner_summary, pos_summary)

            # When constructing small data set, just add to one pile and save when we have a sufficient number
            if small_number > 0:
                small_summaries.append(summary)
                if len(small_summaries) == small_number:
                    with open(pickle_folder + "small_summaries.pickle", "wb") as fout:
                        pickle.dump(small_summaries, fout)
                    break
            else:
                if set == 'train':
                    train_summaries.append(summary)
                elif set == 'valid':
                    valid_summaries.append(summary)
                elif set == 'test':
                    test_summaries.append(summary)

        print("Pickling summaries")
        with open(pickle_folder + "train_summaries.pickle", "wb") as fout:
            pickle.dump(train_summaries, fout)
        with open(pickle_folder + "valid_summaries.pickle", "wb") as fout:
            pickle.dump(valid_summaries, fout)
        with open(pickle_folder + "test_summaries.pickle", "wb") as fout:
            pickle.dump(test_summaries, fout)

        # If only interested in summaries, return here so we don't process the documents
        if summary_only:
            return

        train_docs = []
        valid_docs = []
        test_docs = []

        # In case of creation of small test dataset
        if small_number > 0:
            small_docs = []
            small_train_docs = []
            small_valid_docs = []
            small_test_docs = []

        # Here we load documents, tokenize them, and create Document class instances
        print("Processing documents")
        filenames = glob.glob(os.path.join(input_folder, '*.content'))
        for file_number in range(len(filenames)):
            filename = filenames[file_number]
            doc_id = os.path.basename(filename).replace(".content", "")
            print("Processing:{0}".format(doc_id))
            try:
                (set, kind, start_tag, end_tag) = documents[doc_id]
            except KeyError:
                print("Document id not found: {0}".format(doc_id))
                exit(0)

            if kind == "gutenberg":
                try:
                    with codecs.open(input_folder + doc_id + ".content", "r", encoding='utf-8',
                                     errors='replace') as fin:
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
                            filtered_data == filtered_data.replace(
                                " 's ", " s ")
                            document_tokens = word_tokenize(filtered_data)

                except Exception as error:
                    print(error)
                    print(
                        "Movie for which html extraction doesnt work doesnt work: ", doc_id)

            # Get NER
            entity_dictionary = {}
            other_dictionary = {}
            title_document_tokens = [token.lower() if token.isupper() else token for token in document_tokens]
            string_doc = " ".join(title_document_tokens)
            if len(string_doc) > 1000000:
                q1 = len(string_doc) / 4

                first_quarter = string_doc[0:q1]
                second_quarter = string_doc[q1:q1 * 2]
                third_quarter = string_doc[q1 * 2:q1 * 3]
                fourth_quarter = string_doc[q1 * 3:]
                first_q_tokens = _getNER(first_quarter, entity_dictionary, other_dictionary)
                second_q_tokens = _getNER(second_quarter, entity_dictionary, other_dictionary)
                third_q_tokens = _getNER(third_quarter, entity_dictionary, other_dictionary)
                fourth_q_tokens = _getNER(fourth_quarter, entity_dictionary, other_dictionary)

                NER_document_tokens = first_q_tokens + second_q_tokens + third_q_tokens + fourth_q_tokens
            else:
                NER_document_tokens = _getNER(string_doc, entity_dictionary, other_dictionary)

            doc = Document(
                doc_id, set, kind, NER_document_tokens, qaps[doc_id], entity_dictionary, other_dictionary,
                candidates_per_doc[doc_id], ner_candidates_per_doc[doc_id], pos_candidates_per_doc[doc_id])

            if (file_number + 1) % interval == 0:
                print("Processed {} documents".format(file_number + 1))

            # If testing, add to test list, pickle and return when sufficient documents retrieved
            if small_number > 0:
                small_docs.append(doc)
                if set == "train":
                    small_train_docs.append(doc)
                elif set == "valid":
                    small_valid_docs.append(doc)
                else:
                    small_test_docs.append(doc)
                if len(small_docs) == small_number:
                    with open(pickle_folder + "small_train_docs.pickle", "wb") as fout:
                        pickle.dump(small_train_docs, fout)
                    with open(pickle_folder + "small_valid_docs.pickle", "wb") as fout:
                        pickle.dump(small_valid_docs, fout)
                    with open(pickle_folder + "small_test_docs.pickle", "wb") as fout:
                        pickle.dump(small_test_docs, fout)
                    return


            else:
                if set == "train":
                    train_docs.append(doc)
                elif set == "valid":
                    valid_docs.append(doc)
                else:
                    test_docs.append(doc)

        # Save documents to pickle
        print("Pickling documents")
        with open(pickle_folder + "train_docs.pickle", "wb") as fout:
            pickle.dump(train_docs, fout)
        with open(pickle_folder + "validate_docs.pickle", "wb") as fout:
            pickle.dump(valid_docs, fout)
        with open(pickle_folder + "test_docs.pickle", "wb") as fout:
            pickle.dump(test_docs, fout)

    def replace_entities(self, entity_dictionary, other_dictionary, document_tokens):
        for index, token in enumerate(document_tokens):
            if token.lower() in entity_dictionary:
                document_tokens[index] = entity_dictionary[token.lower()]
            elif token.lower() in other_dictionary:
                document_tokens[index] = other_dictionary[token.lower()]

    def replace_entities_using_ngrams(self, sent, entity_dictionary, other_dictionary):
        ngrams = []
        all_starts = []
        start = [j for j in range(len(sent))]
        for i in range(6, 0, -1):
            ngrams += zip(*[sent[j:] for j in range(i)])
            all_starts += zip(*[start[j:] for j in range(i)])

        label_sent = [None] * len(sent)
        positions_marked = [False for i in range(len(sent))]
        to_remove = []

        for i, ngram in enumerate(ngrams):
            word = " ".join(ngram)
            if word.lower() in entity_dictionary and positions_marked[all_starts[i][0]] == False:
                label_sent[all_starts[i][0]] = entity_dictionary[word.lower()]
                positions_marked[all_starts[i][0]] = True
                for j in range(1, len(ngram)):
                    to_remove.append(all_starts[i][j])

            elif word.lower() in other_dictionary and positions_marked[all_starts[i][0]] == False:
                label_sent[all_starts[i][0]] = other_dictionary[word.lower()]
                positions_marked[all_starts[i][0]] = True
                for j in range(1, len(ngram)):
                    to_remove.append(all_starts[i][j])

        NER_sent = []
        for index in range(len(sent)):
            if index not in to_remove:
                if label_sent[index] is None:
                    NER_sent.append(sent[index])
                else:
                    NER_sent.append(label_sent[index])

        return NER_sent

    def getNER(self, string_data):
        string_data = string_data.decode('utf-8')
        doc = self.nlp(string_data)

        pos_tags = []
        ner_tags = []
        tokens = []

        for index, token in enumerate(doc):
            pos_tags.append(token.pos_)
            tokens.append(token.text)
            type = token.ent_iob_
            if type == "B" or type == "I":
                type = type + "-" + token.ent_type_

            ner_tags.append(type)

        assert (len(ner_tags) == len(pos_tags))
        return ner_tags, pos_tags, tokens

    def load_documents(self, path, summary_path=None, max_documents=0):
        data_points = []
        self.SOS_Token = self.vocab.get_index("<sos>")
        self.EOS_Token = self.vocab.get_index("<eos>")

        anonymize_summary = False
        with open(path, "r") as fin:
            if max_documents > 0:
                documents = pickle.load(fin)[:max_documents]
            else:
                documents = pickle.load(fin)

        if summary_path is not None:
            with open(summary_path, "r") as fin:
                summary_documents = pickle.load(fin)
            anonymize_summary = True
            assert len(summary_documents) == len(documents)

        for index, document in enumerate(documents):

            # self.replace_entities(document.entity_dictionary, document.other_dictionary,document.document_tokens)
            # document.document_tokens = self.vocab.add_and_get_indices(document.document_tokens)
            # if anonymize_summary:
            #     self.replace_entities(document.entity_dictionary, document.other_dictionary, summary_documents[index].document_tokens)

            metrics_per_doc = []

            # document_tokens = self.vocab.add_and_get_indices(document.document_tokens)

            candidate_per_doc_per_answer = []
            candidate_per_doc_per_answer_ner = []
            candidate_per_doc_per_answer_pos = []

            i = 0
            while i < len(document.candidates):
                candidate_per_doc_per_answer.append(document.candidates[i])
                candidate_per_doc_per_answer_ner.append(document.ner_candidates[i])
                candidate_per_doc_per_answer_pos.append(document.pos_candidates[i])
                i += 2

            candidate_per_doc = list(candidate_per_doc_per_answer)

            for query in document.queries:

                # computing bleu with respect to the first correct answer
                metrics = []
                # Pick alternate candidates as they are the first answers

                for candidate_tokens in candidate_per_doc:
                    self.performance.computeMetrics(candidate_tokens, [candidate_per_doc[query.answer_indices[0] / 2]])
                    metrics.append(1.0 - self.performance.bleu1)
                    i += 2

                metrics_per_doc.append(metrics)

                query.question_tokens = self.vocab.add_and_get_indices(query.question_tokens)
                candidate_per_doc_per_answer[query.answer_indices[0] / 2] = self.vocab.add_and_get_indices(
                    candidate_per_doc_per_answer[query.answer_indices[0] / 2])

                query.ner_tokens = self.vocab.add_and_get_indices_NER(query.ner_tokens)
                query.pos_tokens = self.vocab.add_and_get_indices_POS(query.pos_tokens)
                candidate_per_doc_per_answer_ner[query.answer_indices[0] / 2] = self.vocab.add_and_get_indices_NER(
                   candidate_per_doc_per_answer_ner[query.answer_indices[0] / 2])
                candidate_per_doc_per_answer_pos[query.answer_indices[0] / 2] = self.vocab.add_and_get_indices_POS(
                    candidate_per_doc_per_answer_pos[query.answer_indices[0] / 2])

            for idx, query in enumerate(document.queries):
                query.answer_indices[0] = query.answer_indices[0] / 2
                data_points.append(Data_Point
                                   (query.question_tokens, query.answer_indices, candidate_per_doc_per_answer,
                                    metrics_per_doc[idx],
                                    query.ner_tokens, query.pos_tokens, candidate_per_doc_per_answer_ner,
                                    candidate_per_doc_per_answer_pos,
                                    []))

        return data_points

    def load_documents_split_sentences(self, documents,train=False):
        data_points = []
        candidates_embed_docid = {}
        candidate_per_docid = {}
        context_per_docid = {}
        context_tokens_per_docid = {}
        context_ranges_per_docid = {}
        for index, document in enumerate(documents):
            #print(index)
            original_sentences = document.document_tokens
            chunk_length = 40
            num_chunks = 1

            ## each sentence should be fewer than 40 tokens long
            sentences = []
            for e, sent in enumerate(original_sentences):
                if len(sent) > chunk_length:
                    position = 0
                    position_index = 0
                    while position < len(sent):
                        sentences.append(sent[position_index * chunk_length:(position_index + 1) * chunk_length])
                        position_index += 1
                        position += chunk_length
                else:
                    sentences.append(sent)

            chunk_storage = []
            concat_chunk_storage = []
            # sentence_boundaries_storage = []
            chunk_boundaries_storage = []
            e = 0
            rolling_index = 0
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
                        chunk_storage.append(current_chunk)
                        concat_chunk_storage.append(" ".join(current_chunk))
                        chunk_boundaries_storage.append([rolling_index, rolling_index + len(current_chunk)])
                        rolling_index += len(current_chunk)
                        # sentence_boundaries_storage.append(sentence_boundaries)
                else:
                    ## if out of sentences, use the last chunk as is
                    if len(current_chunk) > 0:
                        chunk_storage.append(current_chunk)
                        concat_chunk_storage.append(" ".join(current_chunk))
                        chunk_boundaries_storage.append([rolling_index, rolling_index + len(current_chunk)])
                        rolling_index += len(current_chunk)
                        # sentence_boundaries_storage.append(sentence_boundaries)

            top_chunks = []
            top_chunks_ids = []

            true_candidates = [document.candidates[i] for i in range(0, len(document.candidates), 2)]

            length = len(chunk_storage)
            ## append queries to the end of the vector
            for reference,question in zip(true_candidates, document.qaps):
                chunk_storage.append(reference)
		if train:
		    concat_chunk_storage.append(" ".join(reference) +" "  +  " ".join(question.question_tokens))
		else:
		    concat_chunk_storage.append(" ".join(question.question_tokens))

            vectorizer = CountVectorizer(preprocessor=self.lemmatizer.lemmatize, stop_words=self.stop_words,
                                         ngram_range=(1, 2))
            transformer = TfidfTransformer(sublinear_tf=True)
            counts = vectorizer.fit_transform(concat_chunk_storage)
            tfidf = transformer.fit_transform(counts)
            chunk_docs = tfidf[0:length]
            reference_docs = tfidf[length:]
            related_docs_indices = linear_kernel(reference_docs, chunk_docs).argsort()[:, -num_chunks:]
            for idx in range(len(true_candidates)):
                chunks_per_ref = []
                doc_ids = related_docs_indices[idx][::-1]
                doc_ids = sorted(doc_ids)
                for doc_id in doc_ids:
                    ## these have to be time ordered so that she can just concatenate
                    chunks_per_ref.append(chunk_boundaries_storage[doc_id])
                top_chunks.append(chunks_per_ref)
                top_chunks_ids.append(doc_ids)

            document_tokens = []
            raw_tokens = []
            for sent in document.document_tokens:
                document_tokens += self.vocab.add_and_get_indices(sent)
                raw_tokens += sent

            if self.args.reduced and self.args.emb_elmo:
                context_per_docid[document.id] = np.concatenate(document.document_embed)
            else:
                context_per_docid[document.id] = self.vocab.add_and_get_indices(raw_tokens)


            context_tokens_per_docid[document.id] = raw_tokens
            context_ranges_per_docid[document.id] = chunk_boundaries_storage

            candidate_per_doc_per_answer = []

            candidate_per_doc_per_answer_embed = []
            i = 0
            while i < len(document.candidates):
                candidate_per_doc_per_answer.append(document.candidates[i])
                if self.args.reduced  and self.args.emb_elmo:
                    candidate_per_doc_per_answer_embed.append(document.candidates_embed[i])
                i += 2

            candidate_per_doc_per_answer_raw_tokens = deepcopy(candidate_per_doc_per_answer)
            for query in document.qaps:
                if self.args.reduced  and self.args.emb_elmo:
                    query.query_embed = query.query_embed
                else:
                    query.query_embed = self.vocab.add_and_get_indices(query.question_tokens)

                query.question_tokens = query.question_tokens
                candidate_per_doc_per_answer[query.answer_indices[0] / 2] = self.vocab.add_and_get_indices(
                    candidate_per_doc_per_answer[query.answer_indices[0] / 2])

            candidate_answer_lengths = [len(answer) for answer in candidate_per_doc_per_answer]
            candidate_per_doc_per_answer_indices = deepcopy(candidate_per_doc_per_answer)
            max_candidate_length = max(candidate_answer_lengths)

            if self.args.reduced  and self.args.emb_elmo:
                candidate_padded_answers_embed = np.array(
                [pad_seq_elmo(answer, max_candidate_length) for answer in candidate_per_doc_per_answer_embed])
            else:
                candidate_padded_answers_embed = np.array([pad_seq(answer, max_candidate_length) for answer in candidate_per_doc_per_answer ])


            candidates_embed_docid[document.id] = candidate_padded_answers_embed
            candidate_per_docid[document.id] = candidate_per_doc_per_answer_raw_tokens

            for idx, query in enumerate(document.qaps):
                query.answer_indices[0] = query.answer_indices[0] / 2
                if self.args.sentence_scoring:
                    data_points.append(Elmo_Data_Point
                                       (query.question_tokens, query.query_embed, query.answer_indices,
                                        [], [], candidate_per_doc_per_answer_indices, [], document.id, top_chunks_ids[idx]))
                else:
                    data_points.append(Elmo_Data_Point
                                       (query.question_tokens, query.query_embed, query.answer_indices,
                                        [], [], candidate_per_doc_per_answer_indices, [], document.id, top_chunks[idx]))

        return data_points, candidates_embed_docid, candidate_per_docid, context_per_docid, context_tokens_per_docid, context_ranges_per_docid

    def load_documents_elmo(self, documents, split=True):
        data_points = []
        candidates_embed_docid = {}
        candidate_per_docid = {}
        context_per_docid = {}
        sentence_mask_doc_id = {}
        sentence_lengths_doc = {}
	context_tokens_per_docid = {}
        for index, document in enumerate(documents):

            document_tokens = []
            raw_tokens = []
            sentence_lengths = []
            for sent in document.document_tokens:
                document_tokens += self.vocab.add_and_get_indices(sent)
                raw_tokens += sent
                sentence_lengths.append(len(sent))
            sentence_lengths_doc[document.id] = np.array(sentence_lengths)

            max_sentence_length = max(sentence_lengths)
            sentence_padded_embed = np.array(
                [pad_seq_elmo(sent, max_sentence_length) for sent in document.document_embed])
            sentence_mask_doc_id[document.id] = np.array([[int(x < sentence_lengths[i])
                                                           for x in range(max_sentence_length)] for i in
                                                          range(len(sentence_lengths))])
            if split:
                context_per_docid[document.id] = sentence_padded_embed
            else:
		if self.args.emb_elmo:
		    context_per_docid[document.id] = np.concatenate(document.document_embed)
		else:
		    context_per_docid[document.id] = self.vocab.add_and_get_indices(raw_tokens)

            candidate_per_doc_per_answer = []
            candidate_per_doc_per_answer_embed = []
	    context_tokens_per_docid[document.id] = raw_tokens
            i = 0
	    candidate_per_doc_per_answer_raw_tokens = []
            while i < len(document.candidates):
                candidate_per_doc_per_answer.append(self.vocab.add_and_get_indices(document.candidates[i]))
		candidate_per_doc_per_answer_raw_tokens.append(document.candidates[i])
                if self.args.emb_elmo:
		    candidate_per_doc_per_answer_embed.append(document.candidates_embed[i])
		
		    
                i += 2
		

            for query in document.qaps:
		if self.args.emb_elmo:
		    query.query_embed = query.query_embed
		else:
		    query.query_embed = self.vocab.add_and_get_indices(query.question_tokens)

                # query.question_tokens = self.vocab.add_and_get_indices(query.question_tokens)
                #candidate_per_doc_per_answer[query.answer_indices[0] / 2] = self.vocab.add_and_get_indices(
                 #    candidate_per_doc_per_answer[query.answer_indices[0] / 2])

            candidate_answer_lengths = [len(answer) for answer in candidate_per_doc_per_answer]
            max_candidate_length = max(candidate_answer_lengths)
	    if self.args.emb_elmo:
		candidate_padded_answers_embed = np.array(
                [pad_seq_elmo(answer, max_candidate_length) for answer in candidate_per_doc_per_answer_embed])
	    else:
		candidate_padded_answers_embed = np.array([pad_seq(answer, max_candidate_length) for answer in candidate_per_doc_per_answer ])


            candidates_embed_docid[document.id] = candidate_padded_answers_embed
            candidate_per_docid[document.id] = candidate_per_doc_per_answer_raw_tokens
            for idx, query in enumerate(document.qaps):
                query.answer_indices[0] = query.answer_indices[0] / 2
                data_points.append(Elmo_Data_Point
                                   (query.question_tokens, query.query_embed, query.answer_indices,
                                    [], [], candidate_per_doc_per_answer, [], document.id))

        return data_points, candidates_embed_docid, candidate_per_docid, context_per_docid, context_tokens_per_docid, sentence_lengths_doc


class Vocabulary(object):
    def __init__(self, pad_token='pad', unk='unk', sos='<sos>', eos='<eos>'):

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

    def get_word(self, index):
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

