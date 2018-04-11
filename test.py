try:
    import cPickle as pickle
except:
    import pickle


def convert_document(document):
    document.answers=[]
    for query in document.queries:
        query.answer1 = len(document.answers)
        document.answers.append(query.answer1_tokens)
        query.answer2 = len(document.answers)
        document.answers.append(query.answer2_tokens)

test_path="/home/michiel/Dropbox/CMU/747/project/narrativeqa-master/pickle/small_summaries.pickle"

with open(test_path,"rb") as file:
    summaries=pickle.load(file, encoding='utf-8')

for summary in summaries:
    convert_document(summary)

# Summaries loaded in final form above here somehow
####################







# load data
# extract questions and answers
# create embedding
# create model
# train model
# evaluate model



