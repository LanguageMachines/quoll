
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter
from luiginlp.modules.ucto import Ucto_dir

from quoll.classification_pipeline.functions import featurizer
from quoll.classification_pipeline.modules.tokenize_instances import Tokenize

import numpy
from os import listdir

#################################################################
### Tasks 
#################################################################

# when the input is a file with one tokenized document per line
class Featurize_tokens(Task):

    in_tokenized = InputSlot()

    token_ngrams = Parameter()
    blackfeats = Parameter()
    lowercase = BoolParameter()

    def out_features(self):
        return self.outputfrominput(inputformat='tokenized', stripextension='.tok.txt', addextension='.features.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='tokenized', stripextension='.tok.txt', addextension='.vocabulary.txt')

    def run(self):
        
        # generate dictionary of features
        features = {'token_ngrams':{'n_list':self.token_ngrams.split(), 'blackfeats':self.blackfeats.split()}}
        
        # read in file and put in right format
        with open(self.in_tokenized().path, 'r', encoding = 'utf-8') as file_in:
            documents = file_in.readlines()
            
        if self.lowercase:
            documents = [doc.lower() for doc in documents]

        ft = featurizer.Featurizer(documents, features)
        ft.fit_transform()
        instances, vocabulary = ft.return_instances(['token_ngrams'], )

        numpy.savez(self.out_features().path, data=instances.data, indices=instances.indices, indptr=instances.indptr, shape=instances.shape)

        vocabulary = list(vocabulary)
        with open(self.out_vocabulary().path,'w',encoding='utf-8') as vocab_out:
            vocab_out.write('\n'.join(vocabulary))


# When the input is a directory with tokenized documents
class Featurize_tokdir(Task):

    in_tokdir = InputSlot()

    token_ngrams = Parameter()
    blackfeats = Parameter()
    lowercase = BoolParameter()
        
    def out_features(self):
        return self.outputfrominput(inputformat='tokdir', stripextension='.tok.txtdir', addextension='.features.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='tokdir', stripextension='.tok.txtdir', addextension='.vocabulary.txt')

    def run(self):
        
        # generate dictionary of features
        features = {'token_ngrams':{'n_list':self.token_ngrams.split(), 'blackfeats':self.blackfeats.split()}}
        
        # read in files and put in right format
        documents = []
        for infile in listdir(in_tokdir().path):
            with open(infile, 'r', encoding='utf-8') as f:
                documents.append(f.read().strip()) 
            
        if self.lowercase:
            documents = [doc.lower() for doc in documents]

        ft = featurizer.Featurizer(documents, features)
        ft.fit_transform()
        instances, vocabulary = ft.return_instances(['token_ngrams'], )

        numpy.savez(self.out_features().path, data=instances.data, indices=instances.indices, indptr=instances.indptr, shape=instances.shape)

        vocabulary = list(vocabulary)
        with open(self.out_vocabulary().path,'w',encoding='utf-8') as vocab_out:
            vocab_out.write('\n'.join(vocabulary))

#################################################################
### Component
#################################################################

@registercomponent
class Featurize(StandardWorkflowComponent):
    token_ngrams = Parameter(default='1 2 3')
    blackfeats = Parameter(default=False)
    lowercase = BoolParameter(default=True)    

    tokconfig = Parameter(default=False)
    strip_punctuation = BoolParameter(default=True)
    language = Parameter(default='nl')

    def accepts(self):
        return 
            InputFormat(self, format_id='tokenized', extension='tok.txt'), 
            InputComponent(self, Tokenize, tokconfig=self.tokconfig, strip_punctuation=self.strip_punctuation), 
            InputFormat(self, format_id='toktxtdir', extension='.tok.txtdir', directory=True), 
            InputComponent(self, Ucto_dir, language = self.language)
                    
    def setup(self, workflow, input_feeds):
        if 'tokenized' in input_feeds.keys():
            featurizertask = workflow.new_task('FeaturizerTask_tokens', Featurize_tokens, autopass=True)
            featurizertask.in_tokenized = input_feeds['tokenized']
        elif 'toktxtdir' in input_feeds.keys():
            featurizertask = workflow.new_task('FeaturizerTask_dirtxt', Featurize_tokdir, autopass=True)
            featurizertask.in_tokdir = input_feeds['toktxtdir']
        return featurizertask
