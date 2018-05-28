
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.functions import featurizer
from quoll.classification_pipeline.modules.preprocess import Tokenize_instances, Tokenize_txtdir, Frog_instances, Frog_txtdir

from itertools import groupby 
import numpy
import json
from os import listdir
from scipy import sparse

#################################################################
### Helpers
#################################################################

def format_tokdoc(docname, lowercase):
    # read in file and put in right format
    with open(docname, 'r', encoding = 'utf-8') as file_in:
        lines = file_in.readlines()

    # lowercase if needed
    if lowercase:
        lines = [line.lower() for line in lines]

    # format documents
    document = []
    for line in lines:
        document.append([{'text':token} for token in line.split()])

    return document

def keyfunc(s): # to sort any directory with an index system numerically
    return [int(''.join(g)) if k else ''.join(g) for k, g in groupby(s, str.isdigit)]

#################################################################
### Tasks 
#################################################################

# when the input is a file with one tokenized document per line
class Tokenized2Features(Task):

    in_tokenized = InputSlot()

    featuretypes = Parameter(default='tokens')
    ngrams = Parameter()
    blackfeats = Parameter()
    minimum_token_frequency = IntParameter()
    lowercase = BoolParameter()

    def out_features(self):
        return self.outputfrominput(inputformat='tokenized', stripextension='.tok.txt', addextension='tokens.n_' + '_'.join(self.ngrams.split()) + '.min' + str(self.minimum_token_frequency) + '.lower_' + self.lowercase.__str__() + '.black_' + '_'.join(self.blackfeats.split()) + '.features.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='tokenized', stripextension='.tok.txt', addextension='.tokens.n_' + '_'.join(self.ngrams.split()) + '.min' + str(self.minimum_token_frequency) + '.lower_' + self.lowercase.__str__() + '.black_' + '_'.join(self.blackfeats.split()) + '.vocabulary.txt')

    def run(self):
        
        # generate dictionary of features
        features = {'tokens':{'n_list':self.ngrams.split(), 'blackfeats':self.blackfeats.split(), 'mt':self.minimum_token_frequency}}
        
        # format lines
        documents = [[doc] for doc in format_tokdoc(self.in_tokenized().path,self.lowercase)]

        # extract features
        ft = featurizer.Featurizer(documents, features)
        ft.fit_transform()
        instances, vocabulary = ft.return_instances(['tokens'])

        # write output
        numpy.savez(self.out_features().path, data=instances.data, indices=instances.indices, indptr=instances.indptr, shape=instances.shape)
        vocabulary = list(vocabulary)
        with open(self.out_vocabulary().path,'w',encoding='utf-8') as vocab_out:
            vocab_out.write('\n'.join(vocabulary))


# When the input is a directory with tokenized documents
class Tokdir2Features(Task):

    in_tokdir = InputSlot()

    featuretypes = Parameter(default='tokens')
    ngrams = Parameter()
    blackfeats = Parameter()
    minimum_token_frequency = IntParameter()
    lowercase = BoolParameter()
        
    def out_features(self):
        return self.outputfrominput(inputformat='tokdir', stripextension='.tok.txtdir', addextension='.tokens.n_' + '_'.join(self.ngrams.split()) + '.min' + str(self.minimum_token_frequency) + '.lower_' + self.lowercase.__str__() + '.black_' + '_'.join(self.blackfeats.split()) + '.features.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='tokdir', stripextension='.tok.txtdir', addextension='.tokens.n_' + '_'.join(self.ngrams.split()) + '.min' + str(self.minimum_token_frequency) + '.lower_' + self.lowercase.__str__() + '.black_' + '_'.join(self.blackfeats.split()) + '.vocabulary.txt')

    def run(self):
        
        # generate dictionary of features
        features = {'tokens':{'n_list':self.ngrams.split(), 'blackfeats':self.blackfeats.split(), 'mt':self.minimum_token_frequency}}
        
        # read in files and put in right format for featurizer
        documents = []
        for infile in sorted(listdir(self.in_tokdir().path),key=keyfunc):
            documents.append(format_tokdoc(self.in_tokdir().path + '/' + infile,self.lowercase))
            
        # extract features
        ft = featurizer.Featurizer(documents, features) # to prevent ngrams across sentences, a featurizer is generated per document
        ft.fit_transform()
        instances, vocabulary = ft.return_instances(['tokens'])

        # write output
        numpy.savez(self.out_features().path, data=instances.data, indices=instances.indices, indptr=instances.indptr, shape=instances.shape)
        with open(self.out_vocabulary().path,'w',encoding='utf-8') as vocab_out:
            vocab_out.write('\n'.join(vocabulary))


# when the input is a file with frogged documents
class Frog2Features(Task):

    in_frogged = InputSlot()

    featuretypes = Parameter() # listparameter (within quotes, divided by whitespace); options are tokens, lemmas and pos
    ngrams = Parameter() # listparameter (within quotes, divided by whitespace); applies to all featuretypes
    blackfeats = Parameter() # listparameter (within quotes, divided by whitespace); applies to all featuretypes
    minimum_token_frequency = IntParameter()
    lowercase = BoolParameter() # applies to text tokens only

    def out_features(self):
        return self.outputfrominput(inputformat='frogged', stripextension='.frog.json', addextension='.' + self.featuretypes.replace(' ','.') + '.n_' + '_'.join(self.ngrams.split()) + '.min' + str(self.minimum_token_frequency) + '.lower_' + self.lowercase.__str__() + '.black_' + '_'.join(self.blackfeats.split()) + '.features.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='frogged', stripextension='.frog.json', addextension='.' + self.featuretypes.replace(' ','.') + '.n_' + '_'.join(self.ngrams.split()) + '.min' + str(self.minimum_token_frequency) + '.lower_' + self.lowercase.__str__() + '.black_' + '_'.join(self.blackfeats.split()) + '.vocabulary.txt')

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
                new_doc = []
                for line in document:
                    new_line = []
                    for token in line:
                        new_token = []
                        token['text'] = token['text'].lower()
                        new_token = token
                        new_line.append(new_token)
                    new_doc.append(new_line)
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


# When the input is a directory with frogged documents
class Frogdir2Features(Task):

    in_frogdir = InputSlot()

    featuretypes = Parameter() # listparameter (within quotes, divided by whitespace); options are tokens, lemmas and pos
    ngrams = Parameter() # listparameter (within quotes, divided by whitespace); applies to all featuretypes
    blackfeats = Parameter() # listparameter (within quotes, divided by whitespace); applies to all featuretypes
    minimum_token_frequency = IntParameter()
    lowercase = BoolParameter() # applies to text tokens only
        
    def out_features(self):
        return self.outputfrominput(inputformat='frogdir', stripextension='.frog.jsondir', addextension='.' + self.featuretypes.replace(' ','.') + '.n_' + '_'.join(self.ngrams.split()) + '.min' + str(self.minimum_token_frequency) + '.lower_' + self.lowercase.__str__() + '.black_' + '_'.join(self.blackfeats.split()) + '.features.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='frogdir', stripextension='.frog.jsondir', addextension='.' + self.featuretypes.replace(' ','.') + '.n_' + '_'.join(self.ngrams.split()) + '.min' + str(self.minimum_token_frequency) + '.lower_' + self.lowercase.__str__() + '.black_' + '_'.join(self.blackfeats.split()) + '.vocabulary.txt')

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
        for infile in sorted(listdir(self.in_frogdir().path),key=keyfunc):            
            with open(self.in_frogdir().path + '/' + infile, 'r', encoding = 'utf-8') as file_in:
                document = sum(json.loads(file_in.read()), []) # to convert output, which is represented as  multiple documents, to one document    
                # set text to lowercase if argument is given
                if self.lowercase:
                    for sentence in document:
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
        return InputFormat(self, format_id='tokenized', extension='tok.txt'), InputFormat(self, format_id='frogged', extension='frog.json'), InputFormat(self, format_id='txt', extension='txt'), InputFormat(self, format_id='toktxtdir', extension='.tok.txtdir', directory=True), InputFormat(self, format_id='frogjsondir', extension='.frog.jsondir', directory=True), InputFormat(self, format_id='txtdir', extension='txtdir',directory=True)
                    
    def setup(self, workflow, input_feeds):

        if 'tokenized' in input_feeds.keys():
            featurizertask = workflow.new_task('FeaturizerTask_tokens', Tokenized2Features, autopass=True, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
            featurizertask.in_tokenized = input_feeds['tokenized']

        elif 'frogged' in input_feeds.keys():
            featurizertask = workflow.new_task('FeaturizerTask_frogged', Frog2Features, autopass=True, featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
            featurizertask.in_frogged = input_feeds['frogged']            

        elif 'toktxtdir' in input_feeds.keys():
            featurizertask = workflow.new_task('FeaturizerTask_tokdir', Tokdir2Features, autopass=True, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
            featurizertask.in_tokdir = input_feeds['toktxtdir']

        elif 'frogjsondir' in input_feeds.keys():
            featurizertask = workflow.new_task('FeaturizerTask_frogdir', Frogdir2Features, autopass=True, featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
            featurizertask.in_frogdir = input_feeds['frogjsondir']

        elif 'txt' in input_feeds.keys():
            # could either be frogged or tokenized according to the config that is given as argument
            if self.tokconfig:
                tokenizer = workflow.new_task('tokenize_instances', Tokenize_instances, autopass=True, tokconfig=self.tokconfig, strip_punctuation=self.strip_punctuation)
                tokenizer.in_txt = input_feeds['txt']
                featurizertask = workflow.new_task('FeaturizerTask_txt', Tokenized2Features, autopass=True, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
                featurizertask.in_tokenized = tokenizer.out_tokenized
            elif self.frogconfig:
                frogger = workflow.new_task('frog_instances', Frog_instances, autopass=True, frogconfig=self.frogconfig, strip_punctuation=self.strip_punctuation)
                frogger.in_txt = input_feeds['txt']
                featurizertask = workflow.new_task('FeaturizerTask_txt', Frog2Features, autopass=True, featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
                featurizertask.in_frogged = frogger.out_frogged

        elif 'txtdir' in input_feeds.keys():
            # could either be frogged or tokenized according to the config that is given as argument
            if self.tokconfig:
                tokenizer = workflow.new_task('tokenize_dir', Tokenize_txtdir, autopass=True, tokconfig=self.tokconfig, strip_punctuation=self.strip_punctuation)
                tokenizer.in_txtdir = input_feeds['txtdir']
                featurizertask = workflow.new_task('FeaturizerTask_txtdir', Tokdir2Features, autopass=True, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
                featurizertask.in_tokdir = tokenizer.out_toktxtdir

            elif self.frogconfig:
                frogger = workflow.new_task('frog_dir', Frog_txtdir, autopass=True, frogconfig=self.frogconfig, strip_punctuation=self.strip_punctuation)
                frogger.in_txtdir = input_feeds['txtdir']
                featurizertask = workflow.new_task('FeaturizerTask_txtdir', Frogdir2Features, autopass=True, featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
                featurizertask.in_frogdir = frogger.out_frogjsondir

        return featurizertask
