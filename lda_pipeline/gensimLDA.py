
from gensim import corpora, models

class GensimLDA:

    def __init__(self):
        self.corpus = False
        self.lda = False

    def load_corpus(self, infile):
        self.corpus = corpora.TextCorpus(infile)

    def run_lda(self, nt=50):
        self.lda = models.LdaModel(corpus=self.corpus, id2word=self.corpus.dictionary, num_topics = nt)

    def load_lda(self, infile):
        pass

    def save_lda(self, outfile):
        self.lda.save(outfile)

    def write_document_topics(self, outdir):
        for i, doc in enumerate(self.corpus.get_texts()):
            with open(outdir + '/document_' + str(i) + '.txt', 'w', encoding = 'utf-8') as doc_out:
                topics = self.lda.get_document_topics(bow=self.corpus.dictionary.doc2bow(doc), minimum_probability=0.1)
                for topic in topics:
                    doc_out.write('### Topic ' + str(topic[0]) + ', Probability: ' + str(topic[1]) + '\n---------------------\n')
                    for word in self.lda.show_topic(topic[0], topn=15):
                        doc_out.write(word[0] + '\t' + str(word[1]) + '\n')
                    doc_out.write('\n')

    def write_topics(self, outfile, num_topics=50, num_words=15):
        with open(outfile, 'w', encoding = 'utf-8') as topics_out:
            topics = self.lda.show_topics(num_topics=num_topics,  num_words=num_words)
            for topic in topics:
                topics_out.write('***Topic ' + str(topic[0]) + '\n------------------------\n' + topic[1] + '\n\n')

