class Document:
    def __init__(self, id, set, kind, document_tokens, queries, entity_dictionary,other_dictionary, candidates,ner_candidates, pos_candidates):
        self.document_id = id
        self.kind = kind
        self.set = set
        self.document_tokens = document_tokens
        self.queries = queries
        self.entity_dictionary = entity_dictionary
        self.other_dictionary = other_dictionary
        self.candidates = candidates
        self.ner_candidates = ner_candidates
        self.pos_candidates = pos_candidates

class Query:
    question_tokens = []
    ner_tokens = []
    pos_tokens = []
    answer_indices = []
    answer1_tokens = []
    answer2_tokens = []

    def __init__(self, question_tokens,ner_tokens,pos_tokens, answer_indices):
        self.question_tokens = question_tokens
        self.answer_indices = answer_indices
        self.ner_tokens = ner_tokens
        self.pos_tokens = pos_tokens
        # self.answer1_tokens = answer1_tokens
        # self.answer2_tokens = answer2_tokens

    def get_question_tokens(self):
        return self.question_tokens

    def set_question_tokens(self, tokens):
        self.question_tokens = tokens

    def set_answer_tokens(self, ans_1, ans_2):
        self.answer1_tokens = ans_1
        self.answer2_tokens = ans_2

    def get_answer_tokens(self):
        return (self.answer1_tokens, self.answer2_tokens)

class Span_Data_Point:
    question_tokens = []
    answer_tokens = []
    context_tokens = []
    span_indices = []
    sentence_bleu = []


    def __init__(self, question_tokens, context_tokens, span_indices, sentence_bleu,answer_tokens=None, gold_sentence_index= -1):
        self.question_tokens = question_tokens
        self.context_tokens = context_tokens
        self.answer_tokens = answer_tokens
        self.span_indices = span_indices
        self.sentence_bleu = sentence_bleu
        self.gold_sentence_index = gold_sentence_index

class Span_Data_Point_Elmo:

    def __init__(self,id, question_tokens,question_embed,  context_tokens, span_indices,  answer_tokens=None):
        self.id = id
        self.question_tokens = question_tokens
        self.question_embed = question_embed
        self.context_tokens = context_tokens
        self.answer_tokens = answer_tokens
        self.span_indices = span_indices



class Data_Point:
    question_tokens = []
    context_tokens = []
    candidates = []
    answer_indices = []
    metrics=[]
    ner_for_candidates = []
    pos_for_candidates = []
    ner_for_question = []
    pos_for_question = []
    def __init__(self, question_tokens,answer_indices, candidates,metrics,
                 ner_for_question,pos_for_question,ner_for_candidates,pos_for_candidates,
                 context_tokens):
        self.question_tokens = question_tokens
        self.context_tokens = context_tokens
        self.answer_indices =answer_indices
        self.candidates = candidates
        self.metrics = metrics

        self.ner_for_question = ner_for_question
        self.pos_for_question = pos_for_question
        self.ner_for_candidates = ner_for_candidates
        self.pos_for_candidates = pos_for_candidates

class Elmo_Data_Point:

    def __init__(self, question_tokens,question_embed,
                 answer_indices, context_tokens, context_embed, candidates, candidates_embed, doc_id):
        self.question_tokens = question_tokens
        self.question_embed = question_embed
        self.context_tokens = context_tokens
        self.context_embed = context_embed
        self.answer_indices =answer_indices
        self.candidates = candidates
        self.candidates_embed = candidates_embed
        self.doc_id = doc_id







