# Takes list-of-words corpus as $1
# Takes the gensim (pickled) vb LDA model as $2
# The pyLDAvis html output as $3

import gensim
import pyLDAvis
import pyLDAvis.gensim as gensimvis
from gensim import corpora, models, similarities
import sys

input_corpus = sys.argv[1]
input_model = sys.argv[2]
output_file = sys.argv[3]

corpus = corpora.LowCorpus(input_corpus)
dictionary = corpora.Dictionary.from_corpus(corpus)
lda_model = models.LdaModel.load(input_model)

vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis_data, output_file)
