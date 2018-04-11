import spacy
import sys
from dataloaders.dataloader import DataLoader
from allennlp.modules.elmo import Elmo
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper
import torch
import numpy as np

a = torch.autograd.Variable(torch.LongTensor(np.reshape(ELMoCharacterMapper.convert_word_to_char_ids("hello"), [1, 1, 50])))

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file= "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo_instance=Elmo(options_file, weight_file, 1)

print(elmo_instance(a))


# path="/home/michiel/Dropbox/CMU/747/project/narrativeqa-master/pickle/small.pickle"
# testloader = DataLoader("hello")
# documents=testloader.load_documents(path)
# print(documents[0].queries[0].question_tokens)