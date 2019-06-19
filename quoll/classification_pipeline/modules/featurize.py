
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.functions import featurizer
from quoll.classification_pipeline.modules.preprocess import Tokenize_document, Tokenize_txtdir, Frog_document, Frog_txtdir

from itertools import groupby 
import numpy
import json
from os import listdir

#################################################################
### Helpers
#################################################################

def keyfunc(s): # to sort any directory with an index system numerically
    try:
        jg = [int(''.join(g)) for k, g in groupby(s, str.isdigit)]
    except:
        jg = [''.join(g) for k, g in groupby(s, str.isdigit)]
    return jg
        
#################################################################
### Tasks 
#################################################################

class FeaturizeDoc(Task):

    in_preprocessed = InputSlot()

    featuretypes = Parameter() # listparameter (within quotes, divided by whitespace); options are tokens, lemmas and pos
    ngrams = Parameter() # listparameter (within quotes, divided by whitespace); applies to all featuretypes
    blackfeats = Parameter() # listparameter (within quotes, divided by whitespace); applies to all featuretypes
    minimum_token_frequency = IntParameter()
    lowercase = BoolParameter() # applies to text tokens only

    def out_features(self):
        return self.outputfrominput(inputformat='preprocessed', stripextension='.preprocessed.json', addextension='.features.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='preprocessed', stripextension='.preprocessed.json', addextension='.vocabulary.txt')

    def run(self):

        # extract featuretypes, ngrams and blackfeats
        featuretypes = self.featuretypes.split()
        ngrams = self.ngrams.split()
        blackfeats = self.blackfeats.split()
        
        # generate dictionary of features
        features = {}
        for featuretype in featuretypes:
            features[featuretype] = {'n_list':ngrams, 'blackfeats':blackfeats, 'mt':self.minimum_token_frequency}
        
        # read in file and put in right format
        with open(self.in_frogged().path, 'r', encoding = 'utf-8') as file_in:
            documents = json.loads(file_in.read())
        
        # set text to lowercase if argument is given
        if self.lowercase:
            new_docs = []
            for document in documents:
                new_doc = {}
                new_doc['raw'] = document['raw'].lower()
                new_doc['processed'] = []
                for sentence in document['processed']:
                    new_sentence = []
                    for token in sentence:
                        new_token = []
                        token['text'] = token['text'].lower()
                        new_token = token
                        new_sentence.append(new_token)
                    new_doc['processed'].append(new_sentence)
                new_docs.append(new_doc)
            documents = new_docs

        # extract features
        ft = featurizer.Featurizer(documents, features)
        ft.fit_transform()
        instances, vocabulary = ft.return_instances(featuretypes)

        # write output
        numpy.savez(self.out_features().path, data=instances.data, indices=instances.indices, indptr=instances.indptr, shape=instances.shape)
        vocabulary = list(vocabulary)
        with open(self.out_vocabulary().path,'w',encoding='utf-8') as vocab_out:
            vocab_out.write('\n'.join(vocabulary))


class FeaturizeDir(Task):

    in_preprocessdir = InputSlot()

    featuretypes = Parameter() # listparameter (within quotes, divided by whitespace); options are tokens, lemmas and pos
    ngrams = Parameter() # listparameter (within quotes, divided by whitespace); applies to all featuretypes
    blackfeats = Parameter() # listparameter (within quotes, divided by whitespace); applies to all featuretypes
    minimum_token_frequency = IntParameter()
    lowercase = BoolParameter() # applies to text tokens only
        
    def out_features(self):
        return self.outputfrominput(inputformat='preprocessdir', stripextension='.preprocessdir', addextension='.features.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='preprocessdir', stripextension='.preprocessdir', addextension='.vocabulary.txt')

    def run(self):
        
        # extract featuretypes, ngrams and blackfeats
        featuretypes = self.featuretypes.split()
        ngrams = self.ngrams.split()
        blackfeats = self.blackfeats.split()

        # generate dictionary of features
        features = {}
        for featuretype in featuretypes:
            features[featuretype] = {'n_list':ngrams, 'blackfeats':blackfeats, 'mt':self.minimum_token_frequency}

        # read in files and put in right format
        documents = []
        for infile in sorted(listdir(self.in_preprocessdir().path),key=keyfunc):            
            with open(self.in_preprocessdir().path + '/' + infile, 'r', encoding = 'utf-8') as file_in:
                document = json.loads(file_in.read())
                # set text to lowercase if argument is given
                if self.lowercase:
                    document['raw'] = document['raw'].lower()
                    for sentence in document['processed']:
                        for token in sentence:
                            token['text'] = token['text'].lower()
                documents.append(document)

        # extract features
        ft = featurizer.Featurizer(documents, features)
        ft.fit_transform()
        instances, vocabulary = ft.return_instances(featuretypes)

        # write output
        numpy.savez(self.out_features().path, data=instances.data, indices=instances.indices, indptr=instances.indptr, shape=instances.shape)
        with open(self.out_vocabulary().path,'w',encoding='utf-8') as vocab_out:
            vocab_out.write('\n'.join(vocabulary))


#################################################################
### Component
#################################################################

@registercomponent
class Featurize(StandardWorkflowComponent):
    ngrams = Parameter(default='1 2 3')
    blackfeats = Parameter(default=False)
    lowercase = BoolParameter()    
    minimum_token_frequency = IntParameter(default=1)
    featuretypes = Parameter(default='tokens')

    tokconfig = Parameter(default=False)
    frogconfig = Parameter(default=False)
    strip_punctuation = BoolParameter(default=True)

    def accepts(self):
        return InputFormat(self,format_id='txtdir',extension='txtdir',directory=True), InputFormat(self,format_id='txt',extension='txt'), InputFormat(self,format_id='preprocessed',extension='preprocessed.json'), InputFormat(self,format_id='preprocessdir',extension='.preprocessdir',directory=True)
                    
    def setup(self, workflow, input_feeds):

        if 'preprocessed' in input_feeds.keys():
            featurizertask = workflow.new_task('DocFeaturizer', FeaturizeDoc, autopass=True, featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
            featurizertask.in_preprocessed = input_feeds['preprocessed']
  
        elif 'preprocessdir' in input_feeds.keys():
            featurizertask = workflow.new_task('DirFeaturizer', FeaturizeDir, autopass=True, featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
            featurizertask.in_preprocessdir = input_feeds['preprocessdir']

        elif 'txt' in input_feeds.keys():
            # could either be frogged or tokenized according to the config that is given as argument
            if self.tokconfig and not self.tokconfig == 'false':
                preprocessor = workflow.new_task('preprocess_tok', Tokenize_document, autopass=True, tokconfig=self.tokconfig, strip_punctuation=self.strip_punctuation)
            elif self.frogconfig:
                preprocessor = workflow.new_task('preprocess_frog', Frog_document, autopass=True, frogconfig=self.frogconfig, strip_punctuation=self.strip_punctuation)
            preprocessor.in_txt = input_feeds['txt']
            featurizertask = workflow.new_task('DocFeaturizer', FeaturizeDoc, autopass=True, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
            featurizertask.in_preprocessed = preprocessor.out_preprocessed

        elif 'txtdir' in input_feeds.keys():
            # could either be frogged or tokenized according to the config that is given as argument
            if self.tokconfig and not self.tokconfig == 'false':
                preprocessor = workflow.new_task('preprocess_dir_tok', Tokenize_txtdir, autopass=True, tokconfig=self.tokconfig, strip_punctuation=self.strip_punctuation)
            elif self.frogconfig:
                preprocessor = workflow.new_task('preprocess_dir_frog', Frog_txtdir, autopass=True, frogconfig=self.frogconfig, strip_punctuation=self.strip_punctuation)
            featurizertask = workflow.new_task('DirFeaturizer', FeaturizeDir, autopass=True, featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
            featurizertask.in_preprocessdir = featurizertask.out_preprocessdir

        return featurizertask
