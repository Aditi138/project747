class Document:
    def __init__(self, id, set, kind, document_tokens, queries, entity_dictionary,other_dictionary):
        self.document_id = id
        self.kind = kind
        self.set = set
        self.document_tokens = document_tokens
        self.queries = queries
        self.entity_dictionary = entity_dictionary
        self.other_dictionary = other_dictionary

class Query:
    question_tokens = []
    answer1_tokens = []
    answer2_tokens = []

    def __init__(self, question_tokens, answer1_tokens, answer2_tokens):
        self.question_tokens = question_tokens
        self.answer1_tokens = answer1_tokens
        self.answer2_tokens = answer2_tokens

    def get_question_tokens(self):
        return self.question_tokens

    def set_question_tokens(self, tokens):
        self.question_tokens = tokens

    def set_answer_tokens(self, ans_1, ans_2):
        self.answer1_tokens = ans_1
        self.answer2_tokens = ans_2

    def get_answer_tokens(self):
        return (self.answer1_tokens, self.answer2_tokens)



