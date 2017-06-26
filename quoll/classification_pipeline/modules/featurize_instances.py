
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter

from quoll.classification_pipeline.functions import featurizer
from quoll.classification_pipeline.modules.tokenize_instances import Tokenize_instances, Tokenize_txtdir

import numpy
from os import listdir
from scipy import sparse

#################################################################
### Tasks 
#################################################################

# when the input is a file with one tokenized document per line
class Featurize_tokens(Task):

    in_tokenized = InputSlot()

    ngrams = Parameter()
    blackfeats = Parameter()
    lowercase = BoolParameter()

    def out_features(self):
        return self.outputfrominput(inputformat='tokenized', stripextension='.tok.txt', addextension='.features.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='tokenized', stripextension='.tok.txt', addextension='.vocabulary.txt')

    def run(self):
        
        # generate dictionary of features
        features = {'token_ngrams':{'n_list':self.ngrams.split(), 'blackfeats':self.blackfeats.split()}}
        
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

    ngrams = Parameter()
    blackfeats = Parameter()
    lowercase = BoolParameter()
        
    def out_features(self):
        return self.outputfrominput(inputformat='tokdir', stripextension='.tok.txtdir', addextension='.features.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='tokdir', stripextension='.tok.txtdir', addextension='.vocabulary.txt')

    def run(self):
        
        # generate dictionary of features
        features = {'token_ngrams':{'n_list':self.ngrams.split(), 'blackfeats':self.blackfeats.split()}}
        
        # read in files and extract features
        featurized_documents = []
        for infile in listdir(self.in_tokdir().path):
            with open(self.in_tokdir().path + '/' + infile, 'r', encoding='utf-8') as f:
                document = f.read()
                if self.lowercase:
                    document = document.lower()
                ft = featurizer.Featurizer(document, features)
                ft.fit_transform()
                sentences, vocabulary = ft.return_instances(['token_ngrams'], )
                featurized_documents.append((sentences.sum(axis=0), vocabulary))

        # combine featurized documents
        overall_vocabulary = list(set(sum([list(fd[1]) for fd in featurized_documents],[])))
        feature_index = dict(zip(overall_vocabulary,list(range(len(overall_vocabulary)))))
        nd = len(featurized_documents)
        aligned_docs = []
        for j,fd in enumerate(featurized_documents):
            doc = fd[0]
            vocab = fd[1]
            row = [0] * len(overall_vocabulary)
            for i,feature in enumerate(vocab):
                row[feature_index[feature]] = doc[0,i]
            aligned_docs.append(row)
        instances = sparse.csr_matrix(aligned_docs)

        numpy.savez(self.out_features().path, data=instances.data, indices=instances.indices, indptr=instances.indptr, shape=instances.shape)

        vocabulary = overall_vocabulary
        with open(self.out_vocabulary().path,'w',encoding='utf-8') as vocab_out:
            vocab_out.write('\n'.join(vocabulary))


# When the input is a directory with tokenized documents
class Featurize_frogdir(Task):

    in_tokdir = InputSlot()

    ngrams = Parameter()
    blackfeats = Parameter()
    lowercase = BoolParameter()
        
    def out_features(self):
        return self.outputfrominput(inputformat='tokdir', stripextension='.tok.txtdir', addextension='.features.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='tokdir', stripextension='.tok.txtdir', addextension='.vocabulary.txt')

    def run(self):
        
        # generate dictionary of features
        features = {'pos_ngrams':{'n_list':self.ngrams.split(), 'blackfeats':self.blackfeats.split()}}
        
        # read in files and extract features
        featurized_documents = []
        for infile in listdir(self.in_tokdir().path):
            with open(self.in_tokdir().path + '/' + infile, 'r', encoding='utf-8') as f:
                document = f.read()
                if self.lowercase:
                    document = document.lower()
                ft = featurizer.Featurizer(document, features)
                ft.fit_transform()
                sentences, vocabulary = ft.return_instances(['token_ngrams'], )
                featurized_documents.append((sentences.sum(axis=0), vocabulary))

        # combine featurized documents
        overall_vocabulary = list(set(sum([list(fd[1]) for fd in featurized_documents],[])))
        feature_index = dict(zip(overall_vocabulary,list(range(len(overall_vocabulary)))))
        nd = len(featurized_documents)
        aligned_docs = []
        for j,fd in enumerate(featurized_documents):
            doc = fd[0]
            vocab = fd[1]
            row = [0] * len(overall_vocabulary)
            for i,feature in enumerate(vocab):
                row[feature_index[feature]] = doc[0,i]
            aligned_docs.append(row)
        instances = sparse.csr_matrix(aligned_docs)

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

    tokconfig = Parameter(default=False)
    frogconfig = Parameter(default=False)
    strip_punctuation = BoolParameter(default=True)

    def accepts(self):
        return InputFormat(self, format_id='tokenized', extension='tok.txt'), InputFormat(self, format_id='txt', extension='txt'), InputFormat(self, format_id='toktxtdir', extension='.tok.txtdir', directory=True), InputFormat(self, format_id='txtdir', extension='txtdir',directory=True)
                    
    def setup(self, workflow, input_feeds):
        if 'tokenized' in input_feeds.keys():
            featurizertask = workflow.new_task('FeaturizerTask_tokens', Featurize_tokens, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, autopass=True)
            featurizertask.in_tokenized = input_feeds['tokenized']

        elif 'txt' in input_feeds.keys():
            tokenizer = workflow.new_task('tokenize_instances', Tokenize_instances, autopass=True, tokconfig=self.tokconfig, strip_punctuation=self.strip_punctuation)
            tokenizer.in_txt = input_feeds['txt']

            featurizertask = workflow.new_task('FeaturizerTask_txt', Featurize_tokens, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, autopass=True)
            featurizertask.in_tokenized = tokenizer.out_tokenized

        elif 'toktxtdir' in input_feeds.keys():
            featurizertask = workflow.new_task('FeaturizerTask_dirtok', Featurize_tokdir, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, autopass=True)
            featurizertask.in_tokdir = input_feeds['toktxtdir']

        elif 'txtdir' in input_feeds.keys():
            tokenizer = workflow.new_task('tokenize_dir', Tokenize_txtdir, autopass=True, tokconfig=self.tokconfig, strip_punctuation=self.strip_punctuation)
            tokenizer.in_txtdir = input_feeds['txtdir']

            featurizertask = workflow.new_task('FeaturizerTask_dirtxt', Featurize_tokdir, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, autopass=True)
            featurizertask.in_tokdir = tokenizer.out_toktxtdir

        return featurizertask
