# Reads a list-of-words corpus from $1
# Reads the number of topics from $2

input_file = sys.argv[1]
input_file_base = input_file.rsplit('.', 1)[0]
output_file = input_file_base + '.ldavb'

number_topics = int(sys.argv[2])

import sys
from gensim import corpora, models
corpus = corpora.LowCorpus(input_file)
dictionary = corpora.Dictionary.from_corpus(corpus)
lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=number_topics) 
lda.save(output_file)

