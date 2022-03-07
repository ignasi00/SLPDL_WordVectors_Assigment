
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.stats import entropy
import torch

from data_loading import Vocabulary
from word_analyser_model import WordVectors


DATASET_VERSION = 'ca-100'
CBOW_VOCABULARY_ROOT = f'./data/text-preprocessing/data/{DATASET_VERSION}'
CBOW_VECTORS_ROOT = f'./model_parameters/{DATASET_VERSION}'


plt.rcParams['figure.figsize'] = [9.5, 6]

dict_ = f'{CBOW_VOCABULARY_ROOT}/ca.wiki.train.tokens.nopunct.dic'
counter = pickle.load(open(dict_, 'rb'))
words, values = zip(*counter.most_common(5000))

print('Most frequent Catalan words')
print(words[:10])
print(values[:10])

h = entropy(values)

print(f'Word entropy: {h:5.2f}, Perplexity: {np.exp(h):5.0f}')
print(f'Probability of the most frequent word: {values[0]/sum(values):2.3f}')

_ = plt.plot(values[:50], 'g', 2*values[0]/np.arange(2,52), 'r')
_ = plt.loglog(values)
#plt.show()

benford = Counter(int(str(item[1])[0]) for item in counter.most_common(5000))

print(benford)

percentage = np.array(list(benford.values()), dtype=float)
percentage /= percentage.sum()

_ = plt.bar(list(benford.keys()), percentage*100)

modelname = f'{CBOW_VECTORS_ROOT}/{DATASET_VERSION}.pt'
state_dict = torch.load(modelname, map_location=torch.device('cpu'))

state_dict.keys()

input_word_vectors = state_dict['emb.weight'].numpy()
output_word_vectors = state_dict['lin.weight'].numpy()

token_vocab = Vocabulary()
token_vocab.load(f'{CBOW_VOCABULARY_ROOT}/ca.wiki.vocab')

model1 = WordVectors(input_word_vectors, token_vocab)
model2 = WordVectors(output_word_vectors, token_vocab)

print("\n---- INPUT  EMBEDDING ----")
print("català is similar to: ")
print( model1.most_similar('català') )
print("França is to francès as Polònia is to: ")
print( model1.analogy('França', 'francès', 'Polònia', keep_all=True))
print("França is to francès as Polònia is to: ")
print( model1.analogy('França', 'francès', 'Polònia', keep_all=False))

print("\n---- OUTPUT EMBEDDING ----\n")
print("català is similar to: ")
print( model2.most_similar('català') )
print("França is to francès as Polònia is to: ")
print( model2.analogy('França', 'francès', 'Polònia', keep_all=True))
print("França is to francès as Polònia is to: ")
print( model2.analogy('França', 'francès', 'Polònia', keep_all=False))
