
from gensim import corpora, models
import re

class GensimLDA:

    def __init__(self, num_topics = 50):
        self.nt = num_topics
        self.corpus = False
        self.lda = False
        self.dict = False

    def load_corpus(self, infile):
        self.corpus = corpora.MmCorpus(infile)

    def load_dict(self, infile):
        self.dict = corpora.Dictionary.load(infile)

    def run_lda(self):
        self.lda = models.LdaModel(corpus=self.corpus, id2word=self.dict, num_topics = self.nt, iterations=500000)

    def tfidf_weight(self):
        self.corpus = models.TfidfModel(self.corpus, normalize=True)

    def load_lda(self, infile):
        pass

    def save_lda(self, outfile):
        self.lda.save(outfile)

    def return_document_topics(self, document):
        with open(document,'r',encoding='utf-8') as infile:
            document_text = infile.read().split()
        topics = self.lda.get_document_topics(bow=self.dict.doc2bow(document_text), minimum_probability=0.1)
        topics_str = []
        for topic in topics:
            topic_index = str(topic[0])
            topic_probability = str(topic[1])
            topic_words = []
            for word in self.lda.show_topic(topic[0], topn=40):
                topic_words.append((word[0],word[1]))
            topic_words = ', '.join(['-'.join([str(x) for x in tw]) for tw in topic_words])
            topic_str = '\n'.join([topic_index, topic_probability, topic_words])
            topics_str.append(topic_str)
        topics_str = '\n\n-----------------------\n\n'.join(topics_str)
        return topics_str

    def return_topics(self, num_words=40):
        topics = self.lda.show_topics(num_topics=self.nt,num_words=num_words)
        topics_str = []
        for topic in topics:
            topics_str.append(str(topic[0]) + '\n' + topic[1])
        topics_str = '\n\n-----------------------\n\n'.join(topics_str)
        return topics_str

class GensimCorpus:

    def __init__(self, lines=False):
        self.lines=lines # list of lists

    def load_lines(self, lines):
        self.lines=lines

    def save_corpus(self, corpusfile, dictfile):
        dictionary = corpora.Dictionary(self.lines)
        corpus = [dictionary.doc2bow(line) for line in self.lines]
        dictionary.save(dictfile)
        corpora.MmCorpus.serialize(corpusfile, corpus)

    def strip_stopwords(self, stopwordfile):
        with open(stopwordfile,'r',encoding='utf-8') as infile:
            stopwords = infile.read().split('\n')
        newlines = [[word for word in line if word not in stopwords] for line in self.lines]
        self.lines = newlines

    def strip_shortwords(self, lowest_wordlength):
        newlines = []
        for line in self.lines:
            newlines.append([word for word in line if len(word)>=lowest_wordlength])
        self.lines = newlines

    def strip_numbers(self):
        number = re.compile(r'(\.?\d+(\.|,)?(\d+)?)+')
        newlines = []
        for line in self.lines:
            newlines.append([word for word in line if not (number.match(word))])
        self.lines = newlines

    def tolower(self):
        newlines = [[word.lower() for word in line] for line in self.lines]
        self.lines = newlines
