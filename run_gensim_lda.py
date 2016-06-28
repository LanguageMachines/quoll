

import glob
import os
from luigi import Parameter, IntParameter
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, TargetInfo, InputSlot
from luiginlp.util import replaceextension
import gensimLDA
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

    num_topics = IntParameter(default = 50)

    in_corpus = InputSlot()

    def out_topics(self):
        return self.outputfrominput(inputformat='corpus', stripextension='.corpus.txt', addextension='.gensimlda')    

    def run(self):
        GLDA = gensimLDA.GensimLDA()
        GLDA.run_lda(self.in_corpus().path, num_topics=self.num_topics)
        os.mkdir(self.out_topics().path)
        GLDA.write_document_topics(self.out_topics().path)
        GLDA.write_topics(self.out_topics().path + '/topics.txt', num_topics = self.num_topics)
        GLDA.save_lda(self.out_topics().path + '/model.pcl')
        
@registercomponent
class RunGensimLDAComponent(StandardWorkflowComponent):
    
    num_topics = IntParameter(default=50)

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
