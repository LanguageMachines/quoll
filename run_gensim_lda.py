

import glob
import os
from luigi import Parameter, IntParameter
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, TargetInfo, InputSlot
from luiginlp.util import replaceextension
from gensim import corpora, models

class Featuredir2CorpusTask(Task):

    in_featuredir = InputSlot()

    def out_corpus(self):
        return self.outputfrominput(inputformat='featuredir', stripextension='.featuredir', addextension='.corpus.txt')

    def run(self):

        inputfiles = [filename for filename in glob.glob(self.in_featuredir().path + '/*.features.txt')]
        docs = []
        for inputfile in inputfiles:
            with open(inputfile, 'r', encoding='utf-8') as infile:
                file_str = infile.read()
                words = file_str.strip()
                docs.append(words)
        with open(self.out_corpus().path, 'w', encoding='utf-8') as outfile:
            outfile.write('\n'.join(docs))

class Featuredir2corpusComponent(StandardWorkflowComponent):
    
    def accepts(self):
        return InputFormat(self, format_id='featuredir', extension='.featuredir', directory=True)

    def autosetup(self):
        return Featuredir2CorpusTask

class RunGensimLDATask(Task):

    num_topics = IntParameter(default = 100)

    in_corpus = InputSlot()

    def out_topics(self):
        return self.outputfrominput(inputformat='corpus', stripextension='.corpus.txt', addextension='.gensimlda')    

    def run(self):
        os.mkdir(self.out_topics().path)
        corpus = corpora.TextCorpus(self.in_corpus().path)
        lda = models.LdaModel(corpus=corpus, id2word=corpus.dictionary, num_topics=self.num_topics)
        for i, doc in enumerate(corpus.get_texts()):
            with open(self.out_topics().path + '/document_' + str(i) + '.txt', 'w', encoding = 'utf-8') as doc_out:
                topics = lda.get_document_topics(bow=corpus.dictionary.doc2bow(doc), minimum_probability=0.1)
                for topic in topics:
                    doc_out.write('### Topic ' + str(topic[0]) + ', Probability: ' + str(topic[1]) + '\n---------------------\n')
                    for word in lda.show_topic(topic[0], topn=15):
                        doc_out.write(word[0] + '\t' + str(word[1]) + '\n')
                    doc_out.write('\n')
        with open(self.out_topics().path + '/topics.txt', 'w', encoding = 'utf-8') as topics_out:
            topics = lda.show_topics(num_topics=self.num_topics,  num_words = 15)
            for topic in topics:
                topics_out.write('***Topic ' + str(topic[0]) + '\n------------------------\n' + topic[1] + '\n\n')            
        lda.save(self.out_topics().path + '/model.pcl')

@registercomponent
class RunGensimLDAComponent(StandardWorkflowComponent):
    
    num_topics = IntParameter(default=100)

    def accepts(self):
        return InputComponent(self, Featuredir2corpusComponent)

    def autosetup(self):
        return RunGensimLDATask

#    def setup(self, workflow, input_feeds):
#        gensimtask = workflow.new_task('RunGensimLDATask', RunGensimLDATask, autopass=True)
#        gensimtask.in_corpus = input_feeds['corpus']
#        gensimtask.num_topics = num_topics

        #print('INPUTFEEDS', input_feeds['corpus'])
        
        return gensimtask
