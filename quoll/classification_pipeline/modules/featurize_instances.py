
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter

from quoll.classification_pipeline.functions import featurizer
from quoll.classification_pipeline.modules.tokenize_instances import Tokenize_instances, Tokenize_txtdir
from quoll.classification_pipeline.modules.frog_instances import Frog_instances, Frog_txtdir

import numpy
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
        document.append([{'text':token} for token in line.strip()])

    return document

#################################################################
### Tasks 
#################################################################

# when the input is a file with one tokenized document per line
class Tokenized2Features(Task):

    in_tokenized = InputSlot()

    ngrams = Parameter()
    blackfeats = Parameter()
    minimum_token_frequency = IntParameter()
    lowercase = BoolParameter()

    def out_features(self):
        return self.outputfrominput(inputformat='tokenized', stripextension='.tok.txt', addextension='.features.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='tokenized', stripextension='.tok.txt', addextension='.vocabulary.txt')

    def run(self):
        
        # generate dictionary of features
        features = {'tokens':{'n_list':self.ngrams.split(), 'blackfeats':self.blackfeats.split(), 'mt':self.minimum_token_frequency}}
        
        # format lines
        documents = format_tokdoc(self.in_tokenized().path,self.lowercase)

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
        return self.outputfrominput(inputformat='tokdir', stripextension='.tok.txtdir', addextension='.features.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='tokdir', stripextension='.tok.txtdir', addextension='.vocabulary.txt')

    def run(self):
        
        # generate dictionary of features
        features = {'tokens':{'n_list':self.ngrams.split(), 'blackfeats':self.blackfeats.split()}}
        
        # read in files and put in right format for featurizer
        documents = []
        for infile in listdir(self.in_tokdir().path):
            documents.append(format_tokdoc(self.in_tokdir().path + '/' + infile,self.lowercase))
            
        # extract features
        ft = featurizer.Featurizer(document, features) # to prevent ngrams across sentences, a featurizer is generated per document
        ft.fit_transform()
        instances, vocabulary = ft.return_instances(['tokens'])

        # # combine featurized documents
        # overall_vocabulary = list(set(sum([list(fd[1]) for fd in featurized_documents],[])))
        # feature_index = dict(zip(overall_vocabulary,list(range(len(overall_vocabulary)))))
        # nd = len(featurized_documents)
        # aligned_docs = []
        # for j,fd in enumerate(featurized_documents):
        #     doc = fd[0]
        #     vocab = fd[1]
        #     row = [0] * len(overall_vocabulary)
        #     for i,feature in enumerate(vocab):
        #         row[feature_index[feature]] = doc[0,i]
        #     aligned_docs.append(row)
        # instances = sparse.csr_matrix(aligned_docs)

        # write output
        numpy.savez(self.out_features().path, data=instances.data, indices=instances.indices, indptr=instances.indptr, shape=instances.shape)
        vocabulary = overall_vocabulary
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
        return self.outputfrominput(inputformat='frogged', stripextension='.frog.json', addextension='.features.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='frogged', stripextension='.frog.json', addextension='.vocabulary.txt')

    def run(self):

        # extract featuretypes, ngrams and blackfeats
        featuretypes = self.featuretypes.split()
        ngrams = self.ngrams.split()
        blackfeats = self.blackfeats.split()
        
        # generate dictionary of features
        features = {}
        for featuretype in featuretypes:
            features[featuretype] = {'n_list':ngrams, 'blackfeats':blackfeats}
        
        # read in file and put in right format
        with open(self.in_frogged().path, 'r', encoding = 'utf-8') as file_in:
            documents = json.loads(file_in.read())
        
        # set text to lowercase if argument is given
        if self.lowercase:
            documents = [[[token['text'].lower() for token in line] for line in document] for document in documents]

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
        return self.outputfrominput(inputformat='frogdir', stripextension='.frog.txtdir', addextension='.features.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='frogdir', stripextension='.frog.txtdir', addextension='.vocabulary.txt')

    def run(self):
        
        # extract featuretypes, ngrams and blackfeats
        featuretypes = self.featuretypes.split()
        ngrams = self.ngrams.split()
        blackfeats = self.blackfeats.split()

        # generate dictionary of features
        features = {}
        for featuretype in featuretypes:
            features[featuretype] = {'n_list':ngrams, 'blackfeats':blackfeats}

        # read in files and put in right format
        documents = []
        for infile in listdir(self.in_frogdir().path):
            with open(self.in_frogdir().path + '/' + infile, 'r', encoding = 'utf-8') as file_in:
                document = json.loads(file_in.read())    
                # set text to lowercase if argument is given
                if self.lowercase:
                    document = [[token['text'].lower() for token in line] for line in document]
                documents.append(document)

        # extract features
        ft = featurizer.Featurizer(documents, features)
        ft.fit_transform()
        instances, vocabulary = ft.return_instances(featuretypes)

        # # combine featurized documents
        # overall_vocabulary = list(set(sum([list(fd[1]) for fd in featurized_documents],[])))
        # feature_index = dict(zip(overall_vocabulary,list(range(len(overall_vocabulary)))))
        # nd = len(featurized_documents)
        # aligned_docs = []
        # for j,fd in enumerate(featurized_documents):
        #     doc = fd[0]
        #     vocab = fd[1]
        #     row = [0] * len(overall_vocabulary)
        #     for i,feature in enumerate(vocab):
        #         row[feature_index[feature]] = doc[0,i]
        #     aligned_docs.append(row)
        # instances = sparse.csr_matrix(aligned_docs)

        # write output
        numpy.savez(self.out_features().path, data=instances.data, indices=instances.indices, indptr=instances.indptr, shape=instances.shape)
        vocabulary = overall_vocabulary
        with open(self.out_vocabulary().path,'w',encoding='utf-8') as vocab_out:
            vocab_out.write('\n'.join(vocabulary))


#################################################################
### Component
#################################################################

@registercomponent
class Featurize(StandardWorkflowComponent):
    ngrams = Parameter(default='1 2 3')
    blackfeats = Parameter(default=False)
    lowercase = BoolParameter(default=True)    
    minimum_token_frequency = IntParameter(default=1)

    featuretypes = Parameter(default=False)
    tokconfig = Parameter(default=False)
    frogconfig = Parameter(default=False)
    strip_punctuation = BoolParameter(default=True)

    def accepts(self):
        return InputFormat(self, format_id='tokenized', extension='tok.txt'), InputFormat(self, format_id='frogged', extension='frog.json'), InputFormat(self, format_id='txt', extension='txt'), InputFormat(self, format_id='toktxtdir', extension='.tok.txtdir', directory=True), InputFormat(self, format_id='frogtxtdir', extension='.frog.txtdir', directory=True), InputFormat(self, format_id='txtdir', extension='txtdir',directory=True)
                    
    def setup(self, workflow, input_feeds):
        if 'tokenized' in input_feeds.keys():
            featurizertask = workflow.new_task('FeaturizerTask_tokens', Tokenized2Features, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, autopass=True)
            featurizertask.in_tokenized = input_feeds['tokenized']

        elif 'frogged' in input_feeds.keys():
            featurizertask = workflow.new_task('FeaturizerTask_frogged', Frog2Features, featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, autopass=True)
            featurizertask.in_frogged = input_feeds['frogged']            

        elif 'txt' in input_feeds.keys():
            # could either be frogged or tokenized according to the config that is given as argument
            if self.tokconfig:
                tokenizer = workflow.new_task('tokenize_instances', Tokenize_instances, autopass=True, tokconfig=self.tokconfig, strip_punctuation=self.strip_punctuation)
                tokenizer.in_txt = input_feeds['txt']
                featurizertask = workflow.new_task('FeaturizerTask_txt', Tokenized2Features, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, autopass=True)
                featurizertask.in_tokenized = tokenizer.out_tokenized
            elif self.frogconfig:
                frogger = workflow.new_task('frog_instances', Frog_instances, autopass=True, frogconfig=self.frogconfig, strip_punctuation=self.strip_punctuation)
                frogger.in_txt = input_feeds['txt']
                featurizertask = workflow.new_task('FeaturizerTask_txt', Frog2Features, featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, autopass=True)
                featurizertask.in_frogged = frogger.out_frogged

        elif 'toktxtdir' in input_feeds.keys():
            featurizertask = workflow.new_task('FeaturizerTask_tokdir', Tokdir2Features, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, autopass=True)
            featurizertask.in_tokdir = input_feeds['toktxtdir']

        elif 'frogtxtdir' in input_feeds.keys():
            featurizertask = workflow.new_task('FeaturizerTask_frogdir', Frogdir2Features, featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, autopass=True)
            featurizertask.in_frogdir = input_feeds['frogtxtdir']

        elif 'txtdir' in input_feeds.keys():
            # could either be frogged or tokenized according to the config that is given as argument
            if self.tokconfig:
                tokenizer = workflow.new_task('tokenize_dir', Tokenize_txtdir, autopass=True, tokconfig=self.tokconfig, strip_punctuation=self.strip_punctuation)
                tokenizer.in_txtdir = input_feeds['txtdir']
                featurizertask = workflow.new_task('FeaturizerTask_txtdir', Tokdir2Features, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, autopass=True)
                featurizertask.in_tokdir = tokenizer.out_toktxtdir

            elif self.frogconfig:
                frogger = workflow.new_task('frog_dir', Frog_txtdir, autopass=True, frogconfig=self.tokconfig, strip_punctuation=self.strip_punctuation)
                frogger.in_txtdir = input_feeds['txtdir']
                featurizertask = workflow.new_task('FeaturizerTask_txtdir', Frogdir2Features, featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, autopass=True)
                featurizertask.in_frogdir = frogger.out_frogjsondir

        return featurizertask
