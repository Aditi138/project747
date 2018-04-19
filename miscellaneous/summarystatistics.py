import codecs
from csv import reader
import sys
import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')
summary_path="../narrativeqa-master/third_party/wikipedia/summaries.csv"

summaries=[]
with codecs.open(summary_path, "r", encoding='utf-8', errors='replace') as fin:
    first = True
    for line in reader(fin):
        if first:
            first=False
            continue
        summary_tokens = line[2]
        summaries.append(summary_tokens)
print("Loaded summaries")

summary_lengths=[len(summary) for summary in summaries]
print("Number of summaries: {}".format(len(summaries)))
print("Average length of summaries: {}".format(np.mean(summary_lengths)))
print("Minimum length of summaries: {}".format(np.min(summary_lengths)))
print("Maximum length of summaries: {}".format(np.max(summary_lengths)))
print("tenth percentile of summaries: {}".format(np.percentile(summary_lengths,10)))
print("ninetieth percentile of summaries: {}".format(np.percentile(summary_lengths,90)))