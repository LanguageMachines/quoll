

import glob
import os
from luigi import Parameter, IntParameter
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, TargetInfo, InputSlot
from luiginlp.util import replaceextension
from quoll.lda_pipeline.modules.featuredir2corpus import Featuredir2corpusComponent
import gensimLDA
from gensim import corpora, models

class Task_Writedoctopics(Task):

    in_corpusfile = InputSlot()

    doctopics=Parameter()

    def out_doctopics(self):
        return self.outputfrominput(inputformat='corpusfile', stripextension='.corpus.txt', addextension='.topics.txt')

    def run(self):
        with open(self.out_doctopics().path,'w',encoding='utf-8') as outfile:
            outfile.write(self.doctopics)

class Component_Writedoctopics(StandardWorkflowComponent):

    doctopics=Parameter()

    def accepts(self):
        return InputFormat(self, format_id='corpusfile', extension='.corpus.txt')

    def autosetup(self):
        return Task_Writedoctopics

class RunGensimLDATask(Task):

    num_topics = IntParameter(default = 50)

    in_corpus = InputSlot()

    def out_topics(self):
        return self.outputfrominput(inputformat='corpus', stripextension='.corpusdir', addextension='.'+str(self.num_topics)+'topics.outputdir')

    def run(self):
        self.setup_output_dir(self.out_topics().path)
        GLDA = gensimLDA.GensimLDA(self.num_topics)
        GLDA.load_corpus(self.in_corpus().path + '/corpus.mm')
        GLDA.load_dict(self.in_corpus().path + '/corpus.dict')
        GLDA.run_lda()
        GLDA.save_lda(self.out_topics().path + '/model.pcl')
        topics = GLDA.return_topics()
        with open(self.out_topics().path + '/topics.txt','w',encoding='utf-8') as outfile:
            outfile.write(topics)

        documents = [filename for filename in glob.glob(self.in_corpus().path + '/*.corpus.txt')]
        yield [ Component_Writedoctopics(inputfile=inputfile,outputdir=self.out_topics().path,doctopics = GLDA.return_document_topics(inputfile)) for inputfile in documents ]


@registercomponent
class RunGensimLDAComponent(StandardWorkflowComponent):

    num_topics = IntParameter(default=50)

    def accepts(self):
        return InputFormat(self, format_id='corpus', extension='.corpusdir')

    def autosetup(self):
        return RunGensimLDATask
