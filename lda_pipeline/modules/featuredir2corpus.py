
from luigi import Parameter, IntParameter, BoolParameter
from luiginlp.engine import Task, StandardWorkflowComponent, registercomponent, InputSlot, InputComponent, InputFormat

from functions.gensimLDA import GensimCorpus

import glob

class Task_Features2corpusfile(Task):
    
    in_features = InputSlot()

    bow=Parameter()
        
    def out_corpusfile(self):
        return self.outputfrominput(inputformat='features', stripextension='.features.txt', addextension='.corpus.txt')
                            
    def run(self):
        with open(self.out_corpusfile().path,'w',encoding='utf-8') as outfile:
            outfile.write(self.bow)
                                                                                                
class Component_Features2corpusfile(StandardWorkflowComponent):

    bow=Parameter()

    def accepts(self):
        return InputFormat(self, format_id='features', extension='.features.txt')
    
    def autosetup(self):
        return Task_Features2corpusfile
    
class Featuredir2CorpusTask(Task):

    in_featuredir = InputSlot()

    strip_stopwords = Parameter()
    strip_shortwords = IntParameter()
    strip_numbers = BoolParameter()

    def out_corpusdir(self):
        return self.outputfrominput(inputformat='featuredir', stripextension='.featuredir', addextension='corpusdir')

    def out_corpus(self):
        return self.outputfrominput(inputformat='featuredir', stripextension='.featuredir', addextension='corpusdir/corpus.mm')
        
    def out_dictionary(self):
        return self.outputfrominput(inputformat='featuredir', stripextension='.featuredir', addextension='corpusdir/corpus.dict')
            
    def run(self):
        self.setup_output_dir(self.out_corpusdir().path)

        inputfiles = [filename for filename in glob.glob(self.in_featuredir().path + '/*.features.txt')]
        docs = []
        for inputfile in inputfiles:
            with open(inputfile, 'r', encoding='utf-8') as infile:
                file_str = infile.read()
                words = file_str.strip().split()
                docs.append(words)
    
        gc = GensimCorpus(docs)
        if self.strip_stopwords:
            gc.strip_stopwords(self.strip_stopwords)
        if self.strip_shortwords:
            gc.strip_shortwords(self.strip_shortwords)
        if self.strip_numbers:
            gc.strip_numbers()
        gc.tolower()

        gc.save_corpus(self.out_corpus().path, self.out_dictionary().path)
        
        yield [ Component_Features2corpusfile(inputfile=inputfile,outputdir=self.out_corpusdir().path,bow=' '.join(gc.lines[i])) for i,inputfile in enumerate(inputfiles) ]
        

class Featuredir2corpusComponent(StandardWorkflowComponent):

    strip_stopwords = Parameter(default=False)
    strip_shortwords = IntParameter(default=False)
    strip_numbers = BoolParameter(default=False)

    def accepts(self):
        return InputFormat(self, format_id='featuredir', extension='.featuredir', directory=True)

    def autosetup(self):
        return Featuredir2CorpusTask
