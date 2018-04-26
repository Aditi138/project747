from allennlp.commands.elmo import ElmoEmbedder
import argparse,pickle, codecs
from csv import reader
import spacy
from collections import defaultdict
import numpy as np
import sys
import json

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


def load_documents(qap_path, pickle_folder,e,nlp,doc_file,  use_doc = False, summary_file= None):
    candidates_per_doc = defaultdict(list)
    candidates_embed_per_doc = defaultdict(list)
    qaps = defaultdict(list)
    answer_indices=  defaultdict(list)
    all_questions = []
    doc_ids = []
    embed_info_per_doc = []


    set_id = {}
    with codecs.open(doc_file, "r") as fin:
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
                set_id[doc_id] = set

            index = index + 1

    index = 0
    with codecs.open(qap_path, "r") as fin:


        first = True
        for line in reader(fin):

            # if index == 20:
            #     break
            if first:
                first = False
                continue
            id = line[0]

            if id in qaps:

                doc = nlp(line[3])
                answer1_tokens = [t.text for t in doc]
                candidates_per_doc[id].append(answer1_tokens)

                doc = nlp(line[4])
                answer2_tokens = [t.text for t in doc]
                candidates_per_doc[id].append(answer2_tokens)

                indices = [candidate_index, candidate_index + 1]
                candidate_index += 2

                doc = nlp(line[2])
                q_tokens = [t.text for t in doc]
                all_questions.append(q_tokens)
                doc_ids.append(id)
                qaps[id].append(q_tokens)
                answer_indices[id].append(indices)
                # qaps[id].append(
                #     Query_Embed(q_tokens,indices))
            else:
                print(id)
                qaps[id] = []
                candidates_per_doc[id] = []
                candidate_index = 0

                doc = nlp(line[3])
                answer1_tokens = [t.text for t in doc]
                candidates_per_doc[id].append(answer1_tokens)

                doc = nlp(line[4])
                answer2_tokens = [t.text for t in doc]
                candidates_per_doc[id].append(answer2_tokens)

                indices = [candidate_index, candidate_index + 1]
                candidate_index += 2

                doc = nlp(line[2])
                q_tokens = [t.text for t in doc]
                all_questions.append(q_tokens)
                doc_ids.append(id)
                qaps[id].append(q_tokens)
                answer_indices[id].append(indices)
                # qaps[id].append(
                #     Query_Embed(q_tokens, indices))

            index+=1
    print("Getting embeddings for all questions")
    qpas_embed = defaultdict(list)
    for doc_id in qaps:
        question_tokens = qaps[doc_id]
        question_embeddings = e.embed_batch(question_tokens)
        for index in range(len(question_embeddings)):
            print(question_embeddings[index].shape)
            mean_question_embeddings = np.mean(question_embeddings[index], axis=0)
            qpas_embed[doc_id].append(Query_Embed(question_tokens[index], answer_indices[doc_id][index],mean_question_embeddings))

    for doc_id in candidates_per_doc:
        print("Getting embeddings for candidate:{0}".format(doc_id))
        candidate_embed = e.embed_batch(candidates_per_doc[doc_id])
        for index in range(len(candidate_embed)):
            candidates_embed_per_doc[doc_id].append(np.mean(candidate_embed[index], axis=0))

    # for doc_id in qaps:
    #     embed_info_per_doc.append(Document_Embed(doc_id, qpas_embed[doc_id], candidates_embed_per_doc[doc_id], candidates_per_doc[doc_id]))
    #
    # print("Pickling question_a embeddings")
    # with open(pickle_folder + "question_answer_embed.pickle", "wb") as fout:
    #     pickle.dump(embed_info_per_doc, fout)

    if use_doc:
        train_summaries = []
        valid_summaries = []
        test_summaries = []


        index = 0
        with codecs.open(summary_file, "r", encoding='utf-8', errors='replace') as fin:
            first = True
            for line in reader(fin):
                # if index == 10:
                #     break
                if first:
                    first = False
                    continue
                id = line[0]
                if id == "0025577043f5090cd603c6aea60f26e236195594":
                    summary_tokens = line[2]
                    doc = nlp(summary_tokens)
                    tokenized_sents = [[token.string.strip() for token in s] for s in doc.sents]
                    embeddings = e.embed_batch(tokenized_sents)
                    mean_embeddings = []
                    for index in range(len(embeddings)):
                        mean_embeddings.append(np.mean(embeddings[index], axis=0))

                    doc = Document_All_Embed(id,qpas_embed[id], candidates_embed_per_doc[id], candidates_per_doc[id], tokenized_sents, list(mean_embeddings) )

                    set = set_id[doc.id]
                    if set == 'train':
                        train_summaries.append(doc)
                    elif set == 'valid':
                        valid_summaries.append(doc)
                    elif set == 'test':
                        test_summaries.append(doc)
                index += 1

        with open(args.pickle_folder + "train_summaries_embed.pickle", "wb") as fout:
            pickle.dump(train_summaries, fout, protocol=2)

        with open(args.pickle_folder + "valid_summaries_embed.pickle", "wb") as fout:
            pickle.dump(valid_summaries, fout, protocol=2)

        with open(args.pickle_folder + "test_summaries_embed.pickle", "wb") as fout:
            pickle.dump(test_summaries, fout, protocol=2)

        print("Loaded summaries")


parser = argparse.ArgumentParser()
parser.add_argument("--pickle_folder", type=str, default=None, help="Input sentences")
parser.add_argument("--qaps_file", type=str, default=None, help="Input sentences")
parser.add_argument("--summary_file", type=str, default=None, help="Input sentences")
parser.add_argument("--doc_file", type=str, default=None, help="Input sentences")
parser.add_argument("--use_doc", action="store_true", default=False)
args = parser.parse_args()

e  =ElmoEmbedder(cuda_device=-1)
nlp = spacy.load('en')
load_documents(args.qaps_file, args.pickle_folder, e,nlp, args.doc_file, args.use_doc, args.summary_file)

#Testing
