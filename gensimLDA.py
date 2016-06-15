# Reads a list-of-words corpus from $1
# Reads the number of topics from $2

import sys
from gensim import corpora, models
corpus = corpora.LowCorpus(sys.argv[1])
dictionary = corpora.Dictionary.from_corpus(corpus)
lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=int(sys.argv[2])) 

