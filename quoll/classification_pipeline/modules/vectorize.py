
import numpy
from scipy import sparse
import pickle
import os

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.functions import vectorizer
from quoll.classification_pipeline.modules.preprocess import Tokenize_instances, Tokenize_txtdir, Frog_instances, Frog_txtdir
from quoll.classification_pipeline.modules.featurize import Tokenized2Features, Tokdir2Features, Frog2Features, Frogdir2Features, Featurize

#################################################################
### Tasks #######################################################
#################################################################

class Balance(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()

    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.balanced.features.npz')

    def out_labels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.balanced.labels')

    def in_vocabulary(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.vocabulary.txt')   

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.balanced.vocabulary.txt')   
    
    def run(self):

        # assert that vocabulary file exists (not checked in component)
        assert os.path.exists(self.in_vocabulary().path), 'Vocabulary file not found, make sure the file exists and/or change vocabulary path name to ' + self.in_vocabulary().path 
        
        # load vocabulary
        with open(self.in_vocabulary().path,'r',encoding='utf-8') as infile:
            vocabulary = infile.read().strip().split('\n')
        
        # load featurized traininstances
        loader = numpy.load(self.in_train().path)
        featurized_traininstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load trainlabels
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')
           
        # balance instances by label frequency
        featurized_traininstances_balanced, trainlabels_balanced = vectorizer.balance_data(featurized_traininstances, trainlabels)

        # write traininstances to file
        numpy.savez(self.out_train().path, data=featurized_traininstances_balanced.data, indices=featurized_traininstances_balanced.indices, indptr=featurized_traininstances_balanced.indptr, shape=featurized_traininstances_balanced.shape)

        # write trainlabels to file
        with open(self.out_labels().path, 'w', encoding='utf-8') as l_out:
            l_out.write('\n'.join(trainlabels_balanced))

        # write vocabulary to file
        with open(self.out_vocabulary().path, 'w', encoding='utf-8') as v_out:
            v_out.write('\n'.join(vocabulary))
        
class FitVectorizer(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    
    weight = Parameter()
    prune = IntParameter()

    def in_vocabulary(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.vocabulary.txt')   
     
    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz')

    def out_featureweights(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.weight_' + self.weight + '.prune_' + str(self.prune) + '.featureweights.txt')

    def out_featureselection(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.weight_' + self.weight + '.prune_' + str(self.prune) + '.featureselection.txt')

    def run(self):

        weight_functions = {
                            'frequency':[vectorizer.return_document_frequency, False], 
                            'binary':[vectorizer.return_document_frequency, vectorizer.return_binary_vectors], 
                            'tfidf':[vectorizer.return_idf, vectorizer.return_tfidf_vectors]
                            }

        # assert that vocabulary file exists (not checked in component)
        assert os.path.exists(self.in_vocabulary().path), 'Vocabulary file not found, make sure the file exists and/or change vocabulary path name to ' + self.in_vocabulary().path 

        # load vocabulary
        with open(self.in_vocabulary().path,'r',encoding='utf-8') as infile:
            vocabulary = infile.read().strip().split('\n')
        
        # load featurized instances
        loader = numpy.load(self.in_train().path)
        featurized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load trainlabels
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')

        # calculate feature_weight
        featureweights = weight_functions[self.weight][0](featurized_instances, trainlabels)

        # vectorize instances
        if weight_functions[self.weight][1]:
            trainvectors = weight_functions[self.weight][1](featurized_instances, featureweights)
        else:
            trainvectors = featurized_instances

        # prune features
        featureselection = vectorizer.return_featureselection(featureweights, self.prune)

        # compress vectors
        trainvectors = vectorizer.compress_vectors(trainvectors, featureselection)

        # write instances to file
        numpy.savez(self.out_train().path, data=trainvectors.data, indices=trainvectors.indices, indptr=trainvectors.indptr, shape=trainvectors.shape)

        # write feature weights to file
        with open(self.out_featureweights().path, 'w', encoding = 'utf-8') as w_out:
            outlist = []
            for key, value in sorted(featureweights.items(), key = lambda k: k[0]):
                outlist.append('\t'.join([str(key), str(value)]))
            w_out.write('\n'.join(outlist))

        # write top features to file
        with open(self.out_featureselection().path, 'w', encoding = 'utf-8') as t_out:
            outlist = []
            for index in featureselection:
                outlist.append('\t'.join([vocabulary[index], str(featureweights[index])]))
            t_out.write('\n'.join(outlist))

class ApplyVectorizer(Task):

    in_test = InputSlot()
    in_featureselection = InputSlot()

    weight = Parameter()
    prune = Parameter()

    def in_vocabulary(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.vocabulary.txt')

    def out_test(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz')

    def run(self):

        weight_functions = {
                            'frequency':False, 
                            'binary':vectorizer.return_binary_vectors, 
                            'tfidf':vectorizer.return_tfidf_vectors 
                            }

        # assert that vocabulary file exists (not checked in component)
        assert os.path.exists(self.in_vocabulary().path), 'Vocabulary file not found, make sure the file exists and/or change vocabulary path name to ' + self.in_vocabulary().path 
        
        # load featurized instances
        loader = numpy.load(self.in_test().path)
        featurized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        
        # print next steps (tends to take a while)
        print('Loaded test instances, Shape:',featurized_instances.shape,', now loading topfeatures...')
 
        # load featureselection
        with open(self.in_featureselection().path,'r',encoding='utf-8') as infile:
            lines = infile.read().split('\n')
            featureselection = [line.split('\t') for line in lines]
        featureselection_vocabulary = [x[0] for x in featureselection]
        featureweights = dict([(i, float(feature[1])) for i, feature in enumerate(featureselection)])
        print('Loaded',len(featureselection_vocabulary),'selected features, now loading sourcevocabulary...')

        # align the test instances to the top features to get the right indices
        with open(self.in_vocabulary().path,'r',encoding='utf-8') as infile:
            vocabulary = infile.read().strip().split('\n')
        print('Loaded sourcevocabulary of size',len(vocabulary),', now aligning testvectors...')
        testvectors = vectorizer.align_vectors(featurized_instances, featureselection_vocabulary, vocabulary)
        print('Done. Shape of testvectors:',testvectors.shape,', now setting feature weights...')

        # set feature weights
        if weight_functions[self.weight]:
            testvectors = weight_functions[self.weight](testvectors, featureweights)
        print('Done. Writing instances to file...')

        # write instances to file
        numpy.savez(self.out_test().path, data=testvectors.data, indices=testvectors.indices, indptr=testvectors.indptr, shape=testvectors.shape)

class VectorizeTxt(Task):

    in_txt = InputSlot()

    delimiter = Parameter()
    normalize = BoolParameter()
    selection = Parameter()

    def out_vectors(self):
        return self.outputfrominput(inputformat='txt', stripextension='.txt', addextension='.normalize_' + self.normalize.__str__() + '.vectors.npz')

    def run(self):

        # load instances
        loader = docreader.Docreader()
        instances = loader.parse_txt(self.in_txt().path,delimiter=self.delimiter)
        instances_float = [[0.0 if feature == 'NA' else 0.0 if feature == '#NULL!' else float(feature.replace(',','.')) for feature in instance] for instance in instances]
        instances_sparse = sparse.csr_matrix(instances_float)

        # normalize features
        if self.normalize:
            instances_sparse = vectorizer.normalize_features(instances_sparse)
 
        # transform features to selection
        if self.selection:
            with open(self.selection,'r',encoding='utf-8') as file_in:
                featureselection = file_in.read().strip().split()
            instances_sparse = vectorizer.compress_vectors(instances_sparse,featureselection)

        # write instances to file
        numpy.savez(self.out_vectors().path, data=instances_sparse.data, indices=instances_sparse.indices, indptr=instances_sparse.indptr, shape=instances_sparse.shape)

class VectorizeCsv(Task):

    in_csv = InputSlot()

    delimiter = Parameter()
    normalize = BoolParameter()
    selection = Parameter()

    def out_vectors(self):
        return self.outputfrominput(inputformat='csv', stripextension='.csv', addextension='.normalize_' + self.normalize.__str__() + '.vectors.npz')

    def run(self):

        # load instances
        loader = docreader.Docreader()
        instances = loader.parse_csv(self.in_csv().path,delimiter=self.delimiter)
        instances_float = [[0.0 if feature == 'NA' else 0.0 if feature == '#NULL!' else float(feature.replace(',','.')) for feature in instance] for instance in instances]
        instances_sparse = sparse.csr_matrix(instances_float)

        # normalize features
        if self.normalize:
            instances_sparse = vectorizer.normalize_features(instances_sparse)

        # transform features to selection
        if self.selection:
            with open(self.selection,'r',encoding='utf-8') as file_in:
                featureselection = file_in.read().strip().split()
            instances_sparse = vectorizer.compress_vectors(instances_sparse,featureselection)

        # write instances to file
        numpy.savez(self.out_vectors().path, data=instances_sparse.data, indices=instances_sparse.indices, indptr=instances_sparse.indptr, shape=instances_sparse.shape)

#################################################################
### Component ###################################################
#################################################################

@registercomponent
class Vectorize(WorkflowComponent):
    
    instances = Parameter()
    labels = Parameter()

    # vectorizer parameters
    weight = Parameter(default = 'frequency') # options: frequency, binary, tfidf
    prune = IntParameter(default = 5000) # after ranking the topfeatures in the training set, based on frequency or idf weighting
    balance = BoolParameter()
    normalize = BoolParameter()
    delimiter = Parameter(default=' ')
    selection = Parameter(default=False) # option to make selection based on integers

    # featurizer parameters
    ngrams = Parameter(default='1 2 3')
    blackfeats = Parameter(default=False)
    lowercase = BoolParameter()    
    minimum_token_frequency = IntParameter(default=1)
    featuretypes = Parameter(default='tokens')

    # ucto / frog parameters
    tokconfig = Parameter(default=False)
    frogconfig = Parameter(default=False)
    strip_punctuation = BoolParameter()

    def accepts(self):
        return  [ 
            ( InputComponent(self,Featurize,inputfile=self.instances,ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation), InputFormat(self, format_id='featurized',extension='.features.npz',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ),
            ( InputFormat(self, format_id='featurized_csv',extension='.features.csv',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), 
            ( InputFormat(self, format_id='featurized_txt',extension='.features.txt',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') )
            # ( InputFormat(self, format_id='tokenized',extension='.tok.txt',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), 
            # ( InputFormat(self, format_id='tokdir',extension='.tok.txtdir',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), 
            # ( InputFormat(self, format_id='frogged',extension='.frog.json',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), 
            # ( InputFormat(self, format_id='frogdir',extension='.frog.jsondir',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), 
            # ( InputFormat(self, format_id='txt',extension='.txt',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), 
            # ( InputFormat(self, format_id='txtdir',extension='.txtdir',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ) 
            ]
    
    def setup(self, workflow, input_feeds):

        if 'featurized_csv' in input_feeds.keys():
            vectorizer = workflow.new_task('vectorizer_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter,normalize=self.normalize,selection=self.selection)
            vectorizer.in_csv = input_feeds['featurized_csv']
        
        elif 'featurized_txt' in input_feeds.keys():
            vectorizer = workflow.new_task('vectorizer_txt',VectorizeTxt,autopass=True,delimiter=self.delimiter,normalize=self.normalize,selection=self.selection)
            vectorizer.in_txt = input_feeds['featurized_txt']

        else:

            labels = input_feeds['labels']
            
            if 'featurized' in input_feeds.keys():
                instances = input_feeds['featurized']
            
            else: # earlier stage
                instances =  = workflow.new_component('featurize',Featurize)

            if self.balance:
                balancetask = workflow.new_task('BalanceTask',Balance,autopass=True)
                balancetask.in_train = instances
                balancetask.in_trainlabels = labels
                instances = balancetask.out_train
                labels = balancetask.out_labels
                                
            vectorizer = workflow.new_task('vectorizer',FitVectorizer,autopass=True,weight=self.weight,prune=self.prune)
            vectorizer.in_train = instances
            vectorizer.in_trainlabels = labels

        return vectorizer

                # if 'tokenized' in input_feeds.keys():
                #     featurizer = workflow.new_task('featurizer_tokens',Tokenized2Features,autopass=True,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                #     featurizer.in_tokenized = input_feeds['tokenized']           

                # elif 'tokdir' in input_feeds.keys():
                #     featurizer = workflow.new_task('featurizer_tokdir',Tokdir2Features,autopass=True,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                #     featurizer.in_tokdir = inputfeeds['tokdir']            

                # elif 'frogged' in input_feeds.keys():
                #     featurizer = workflow.new_task('featurizer_frogged',Frog2Features,autopass=True,featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
                #     featurizer.in_frogged = input_feeds['frogged']

                # elif 'frogdir' in input_feeds.keys():
                #     featurizer = workflow.new_task('featurizer_frogdir',Frogdir2Features,autopass=True,featuretypes=self.featuretypes,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                #     featurizer.in_frogdir = input_feeds['frogdir']

                # elif 'txt' in input_feeds.keys():
                #     # could either be frogged or tokenized according to the config that is given as argument
                #     if self.tokconfig:
                #         tokenizer = workflow.new_task('tokenizer_txt',Tokenize_instances,autopass=True,tokconfig=self.tokconfig,strip_punctuation=self.strip_punctuation)
                #         tokenizer.in_txt = input_feeds['txt']
                #         featurizer = workflow.new_task('featurizer_toktxt',Tokenized2Features,autopass=True,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                #         featurizer.in_tokenized = tokenizer.out_tokenized

                #     elif self.frogconfig:
                #         frogger = workflow.new_task('frogger_txt',Frog_instances,autopass=True,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                #         frogger.in_txt = input_feeds['txt']
                #         featurizer = workflow.new_task('frogger_txt', Frog2Features, featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency, autopass=True)
                #         featurizer.in_frogged = frogger.out_frogged

                # elif 'txtdir' in input_feeds.keys():
                #     # could either be frogged or tokenized according to the config that is given as argument
                #     if self.tokconfig:
                #         tokenizer = workflow.new_task('tokenizer_txtdir',Tokenize_txtdir,autopass=True,tokconfig=self.tokconfig,strip_punctuation=self.strip_punctuation)
                #         tokenizer.in_txtdir = input_feeds['txtdir']
                #         featurizer = workflow.new_task('featurizer_toktxtdir',Tokdir2Features,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                #         featurizer.in_tokdir = tokenizer.out_toktxtdir                

                #     elif self.frogconfig:
                #         frogger = workflow.new_task('frogger_txtdir',Frog_txtdir,autopass=True,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                #         frogger.in_txtdir = input_feeds['txtdir']
                #         featurizer = workflow.new_task('featurizer_frogtxtdir',Frogdir2Features,autopass=True,featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
                #         featurizer.in_frogdir = frogger.out_frogjsondir

                #instances = featurizer.out_features



@registercomponent
class VectorizeTest(WorkflowComponent):
    
    instances = Parameter()
    featureselection = Parameter()

    # vectorizer parameters
    weight = Parameter(default = 'frequency') # options: frequency, binary, tfidf
    prune = IntParameter(default = 5000) # after ranking the featureselection in the training set, based on frequency or idf weighting
    normalize = BoolParameter()
    delimiter = Parameter(default=' ')

    # featurizer parameters
    ngrams = Parameter(default='1 2 3')
    blackfeats = Parameter(default=False)
    lowercase = BoolParameter()    
    minimum_token_frequency = IntParameter(default=1)
    featuretypes = Parameter(default='tokens')

    # ucto / frog parameters
    tokconfig = Parameter(default=False)
    frogconfig = Parameter(default=False)
    strip_punctuation = BoolParameter()

    def accepts(self):
        return  [ 
            ( InputFormat(self, format_id='featurized',extension='.features.npz',inputparameter='instances'), InputFormat(self, format_id='featureselection',extension='.featureselection.txt',inputparameter='featureselection') ), 
            ( InputFormat(self, format_id='featurized_csv',extension='.features.csv',inputparameter='instances'), InputFormat(self, format_id='featureselection',extension='.featureselection.txt',inputparameter='featureselection') ), 
            ( InputFormat(self, format_id='featurized_txt',extension='.features.txt',inputparameter='instances'), InputFormat(self, format_id='featureselection',extension='.featureselection.txt',inputparameter='featureselection') ), 
            ( InputFormat(self, format_id='tokenized',extension='.tok.txt',inputparameter='instances'), InputFormat(self, format_id='featureselection',extension='.featureselection.txt',inputparameter='featureselection') ), 
            ( InputFormat(self, format_id='tokdir',extension='.tok.txtdir',inputparameter='instances'), InputFormat(self, format_id='featureselection',extension='.featureselection.txt',inputparameter='featureselection') ), 
            ( InputFormat(self, format_id='frogged',extension='.frog.json',inputparameter='instances'), InputFormat(self, format_id='featureselection',extension='.featureselection.txt',inputparameter='featureselection') ), 
            ( InputFormat(self, format_id='frogdir',extension='.frog.jsondir',inputparameter='instances'), InputFormat(self, format_id='featureselection',extension='.featureselection.txt',inputparameter='featureselection') ), 
            ( InputFormat(self, format_id='txt',extension='.txt',inputparameter='instances'), InputFormat(self, format_id='featureselection',extension='.featureselection.txt',inputparameter='featureselection') ), 
            ( InputFormat(self, format_id='txtdir',extension='.txtdir',inputparameter='instances'), InputFormat(self, format_id='featureselection',extension='.featureselection.txt',inputparameter='featureselection') )
            ]

    def setup(self, workflow, input_feeds):

        featureselection = input_feeds['featureselection']

        if 'featurized_csv' in input_feeds.keys():
            testvectorizer = workflow.new_task('vectorizer_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter,normalize=self.normalize,selection=featureselection)
            testvectorizer.in_csv = input_feeds['featurized_csv']
        
        elif 'featurized_txt' in input_feeds.keys():
            testvectorizer = workflow.new_task('vectorizer_txt',VectorizeTxt,autopass=True,delimiter=self.delimiter,normalize=self.normalize,selection=featureselection)
            testvectorizer.in_txt = input_feeds['featurized_txt']

        else:

            if 'featurized' in input_feeds.keys():
                testinstances = input_feeds['featurized']
            
            else: # earlier stage
                if 'tokenized' in input_feeds.keys():
                    featurizer = workflow.new_task('featurizer_tokens',Tokenized2Features,autopass=True,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                    featurizer.in_tokenized = input_feeds['tokenized']           

                elif 'tokdir' in input_feeds.keys():
                    featurizer = workflow.new_task('featurizer_tokdir',Tokdir2Features,autopass=True,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                    featurizer.in_tokdir = inputfeeds['tokdir']            

                elif 'frogged' in input_feeds.keys():
                    featurizer = workflow.new_task('featurizer_frogged',Frog2Features,autopass=True,featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
                    featurizer.in_frogged = input_feeds['frogged']

                elif 'frogdir' in input_feeds.keys():
                    featurizer = workflow.new_task('featurizer_frogdir',Frogdir2Features,autopass=True,featuretypes=self.featuretypes,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                    featurizer.in_frogdir = input_feeds['frogdir']

                elif 'txt' in input_feeds.keys():
                    # could either be frogged or tokenized according to the config that is given as argument
                    if self.tokconfig:
                        tokenizer = workflow.new_task('tokenizer_txt',Tokenize_instances,autopass=True,tokconfig=self.tokconfig,strip_punctuation=self.strip_punctuation)
                        tokenizer.in_txt = input_feeds['txt']
                        featurizer = workflow.new_task('featurizer_toktxt',Tokenized2Features,autopass=True,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                        featurizer.in_tokenized = tokenizer.out_tokenized

                    elif self.frogconfig:
                        frogger = workflow.new_task('frogger_txt',Frog_instances,autopass=True,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                        frogger.in_txt = input_feeds['txt']
                        featurizer = workflow.new_task('frogger_txt', Frog2Features, featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency, autopass=True)
                        featurizer.in_frogged = frogger.out_frogged

                elif 'txtdir' in input_feeds.keys():
                    # could either be frogged or tokenized according to the config that is given as argument
                    if self.tokconfig:
                        tokenizer = workflow.new_task('tokenizer_txtdir',Tokenize_txtdir,autopass=True,tokconfig=self.tokconfig,strip_punctuation=self.strip_punctuation)
                        tokenizer.in_txtdir = input_feeds['txtdir']
                        featurizer = workflow.new_task('featurizer_toktxtdir',Tokdir2Features,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                        featurizer.in_tokdir = tokenizer.out_toktxtdir                

                    elif self.frogconfig:
                        frogger = workflow.new_task('frogger_txtdir',Frog_txtdir,autopass=True,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                        frogger.in_txtdir = input_feeds['txtdir']
                        featurizer = workflow.new_task('featurizer_frogtxtdir',Frogdir2Features,autopass=True,featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
                        featurizer.in_frogdir = frogger.out_frogjsondir

                testinstances = featurizer.out_features
                
            testvectorizer = workflow.new_task('testvectorizer',ApplyVectorizer,autopass=True,weight=self.weight,prune=self.prune)
            testvectorizer.in_test = test_instances
            testvectorizer.in_featureselection = featureselection

        return testvectorizer


@registercomponent
class VectorizeTrainTest(WorkflowComponent):
    
    traininstances = Parameter()
    trainlabels = Parameter()
    testinstances = Parameter()

    # vectorizer parameters
    weight = Parameter(default = 'frequency') # options: frequency, binary, tfidf
    prune = IntParameter(default = 5000) # after ranking the topfeatures in the training set, based on frequency or idf weighting
    balance = BoolParameter()
    normalize = BoolParameter()
    delimiter = Parameter(default=' ')
    selection = Parameter(default=False) # option to make selection based on integers

    # featurizer parameters
    ngrams = Parameter(default='1 2 3')
    blackfeats = Parameter(default=False)
    lowercase = BoolParameter()    
    minimum_token_frequency = IntParameter(default=1)
    featuretypes = Parameter(default='tokens')

    # ucto / frog parameters
    tokconfig = Parameter(default=False)
    frogconfig = Parameter(default=False)
    strip_punctuation = BoolParameter()

    def accepts(self):
        return  [ 
            ( InputFormat(self, format_id='featurized_train',extension='.features.npz',inputparameter='traininstances'), InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels'), InputFormat(self, format_id='featurized_test',extension='.features.npz',inputparameter='testinstances') ), 
            ( InputFormat(self, format_id='featurized_train_csv',extension='.features.csv',inputparameter='traininstances'), InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels'), InputFormat(self, format_id='featurized_test_csv',extension='.features.csv',inputparameter='testinstances') ), 
            ( InputFormat(self, format_id='featurized_train_txt',extension='.features.txt',inputparameter='traininstances'), InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels'), InputFormat(self, format_id='featurized_test_txt',extension='.features.txt',inputparameter='testinstances') ), 
            ( InputFormat(self, format_id='tokenized_train',extension='.tok.txt',inputparameter='traininstances'), InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels'), InputFormat(self, format_id='tokenized_test',extension='.tok.txt',inputparameter='testinstances') ), 
            ( InputFormat(self, format_id='tokdir_train',extension='.tok.txtdir',inputparameter='traininstances'), InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels'), InputFormat(self, format_id='tokdir_test',extension='.tok.txtdir',inputparameter='testinstances') ), 
            ( InputFormat(self, format_id='frogged_train',extension='.frog.json',inputparameter='traininstances'), InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels'), InputFormat(self, format_id='frogged_test',extension='.frog.json',inputparameter='testinstances') ), 
            ( InputFormat(self, format_id='frogdir_train',extension='.frog.jsondir',inputparameter='traininstances'), InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels'), InputFormat(self, format_id='frogdir_test',extension='.frog.jsondir',inputparameter='testinstances') ), 
            ( InputFormat(self, format_id='txt_train',extension='.txt',inputparameter='traininstances'), InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels'), InputFormat(self, format_id='txt_test',extension='.txt',inputparameter='testinstances') ), 
            ( InputFormat(self, format_id='txtdir_train',extension='.txtdir',inputparameter='traininstances'), InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels'), InputFormat(self, format_id='txtdir_test',extension='.txtdir',inputparameter='testinstances') ) 
            ]
    
    def setup(self, workflow, input_feeds):

        ######################
        ### Training phase ###
        ######################

        if 'featurized_train_csv' in input_feeds.keys():
            trainvectorizer = workflow.new_task('train_vectorizer_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter,normalize=self.normalize,selection=self.selection)
            trainvectorizer.in_csv = input_feeds['featurized_train_csv']
        
        elif 'featurized_train_txt' in input_feeds.keys():
            trainvectorizer = workflow.new_task('train_vectorizer_txt',VectorizeTxt,autopass=True,delimiter=self.delimiter,normalize=self.normalize,selection=self.selection)
            trainvectorizer.in_txt = input_feeds['featurized_train_txt']

        else:

            labels = input_feeds['labels_train']
            
            if 'featurized_train' in input_feeds.keys():
                traininstances = input_feeds['featurized_train']
            
            else: # earlier stage
                if 'tokenized_train' in input_feeds.keys():
                    trainfeaturizer = workflow.new_task('trainfeaturizer_tokens',Tokenized2Features,autopass=True,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                    trainfeaturizer.in_tokenized = input_feeds['tokenized_train']           

                elif 'tokdir_train' in input_feeds.keys():
                    trainfeaturizer = workflow.new_task('trainfeaturizer_tokdir',Tokdir2Features,autopass=True,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                    trainfeaturizer.in_tokdir = inputfeeds['tokdir_train']            

                elif 'frogged_train' in input_feeds.keys():
                    trainfeaturizer = workflow.new_task('trainfeaturizer_frogged',Frog2Features,autopass=True,featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
                    trainfeaturizer.in_frogged = input_feeds['frogged_train']

                elif 'frogdir_train' in input_feeds.keys():
                    trainfeaturizer = workflow.new_task('trainfeaturizer_frogdir',Frogdir2Features,autopass=True,featuretypes=self.featuretypes,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                    trainfeaturizer.in_frogdir = input_feeds['frogdir_train']

                elif 'txt_train' in input_feeds.keys():
                    # could either be frogged or tokenized according to the config that is given as argument
                    if self.tokconfig:
                        traintokenizer = workflow.new_task('traintokenizer_txt',Tokenize_instances,autopass=True,tokconfig=self.tokconfig,strip_punctuation=self.strip_punctuation)
                        traintokenizer.in_txt = input_feeds['txt_train']
                        trainfeaturizer = workflow.new_task('trainfeaturizer_toktxt',Tokenized2Features,autopass=True,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                        trainfeaturizer.in_tokenized = traintokenizer.out_tokenized

                    elif self.frogconfig:
                        trainfrogger = workflow.new_task('trainfrogger_txt',Frog_instances,autopass=True,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                        trainfrogger.in_txt = input_feeds['txt_train']
                        trainfeaturizer = workflow.new_task('trainfrogger_txt', Frog2Features, featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency, autopass=True)
                        trainfeaturizer.in_frogged = trainfrogger.out_frogged

                elif 'txtdir_train' in input_feeds.keys():
                    # could either be frogged or tokenized according to the config that is given as argument
                    if self.tokconfig:
                        traintokenizer = workflow.new_task('traintokenizer_txtdir',Tokenize_txtdir,autopass=True,tokconfig=self.tokconfig,strip_punctuation=self.strip_punctuation)
                        traintokenizer.in_txtdir = input_feeds['txtdir_train']
                        trainfeaturizer = workflow.new_task('trainfeaturizer_toktxtdir',Tokdir2Features,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                        trainfeaturizer.in_tokdir = traintokenizer.out_toktxtdir                

                    elif self.frogconfig:
                        trainfrogger = workflow.new_task('trainfrogger_txtdir',Frog_txtdir,autopass=True,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                        trainfrogger.in_txtdir = input_feeds['txtdir_train']
                        trainfeaturizer = workflow.new_task('trainfeaturizer_frogtxtdir',Frogdir2Features,autopass=True,featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
                        trainfeaturizer.in_frogdir = trainfrogger.out_frogjsondir

                traininstances = trainfeaturizer.out_features

            if self.balance:
                balancetask = workflow.new_task('BalanceTask',Balance,autopass=True)
                balancetask.in_train = traininstances
                balancetask.in_trainlabels = labels
                traininstances = balancetask.out_train
                labels = balancetask.out_labels
                                
            trainvectorizer = workflow.new_task('vectorizer',FitVectorizer,autopass=True,weight=self.weight,prune=self.prune)
            trainvectorizer.in_train = traininstances
            trainvectorizer.in_trainlabels = labels

        ######################
        ### Testing phase ####
        ######################

        if 'featurized_csv' in input_feeds.keys():
            testvectorizer = workflow.new_task('vectorizer_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter,normalize=self.normalize,selection=self.selection)
            testvectorizer.in_csv = input_feeds['featurized_test_csv']
        
        elif 'featurized_txt' in input_feeds.keys():
            testvectorizer = workflow.new_task('vectorizer_txt',VectorizeTxt,autopass=True,delimiter=self.delimiter,normalize=self.normalize,selection=self.selection)
            testvectorizer.in_txt = input_feeds['featurized_test_txt']

        else:

            if 'featurized_test' in input_feeds.keys():
                testinstances = input_feeds['featurized']
            
            else: # earlier stage
                if 'tokenized_test' in input_feeds.keys():
                    testfeaturizer = workflow.new_task('testfeaturizer_tokens',Tokenized2Features,autopass=True,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                    testfeaturizer.in_tokenized = input_feeds['tokenized_test']           

                elif 'tokdir_test' in input_feeds.keys():
                    testfeaturizer = workflow.new_task('testfeaturizer_tokdir',Tokdir2Features,autopass=True,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                    testfeaturizer.in_tokdir = inputfeeds['tokdir_test']            

                elif 'frogged_test' in input_feeds.keys():
                    testfeaturizer = workflow.new_task('testfeaturizer_frogged',Frog2Features,autopass=True,featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
                    testfeaturizer.in_frogged = input_feeds['frogged_test']

                elif 'frogdir_test' in input_feeds.keys():
                    testfeaturizer = workflow.new_task('testfeaturizer_frogdir',Frogdir2Features,autopass=True,featuretypes=self.featuretypes,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                    testfeaturizer.in_frogdir = input_feeds['frogdir_test']

                elif 'txt_test' in input_feeds.keys():
                    # could either be frogged or tokenized according to the config that is given as argument
                    if self.tokconfig:
                        testtokenizer = workflow.new_task('testtokenizer_txt',Tokenize_instances,autopass=True,tokconfig=self.tokconfig,strip_punctuation=self.strip_punctuation)
                        testtokenizer.in_txt = input_feeds['txt_test']
                        testfeaturizer = workflow.new_task('testfeaturizer_toktxt',Tokenized2Features,autopass=True,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                        testfeaturizer.in_tokenized = testtokenizer.out_tokenized

                    elif self.frogconfig:
                        testfrogger = workflow.new_task('testfrogger_txt',Frog_instances,autopass=True,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                        testfrogger.in_txt = input_feeds['txt_test']
                        testfeaturizer = workflow.new_task('testfrogger_txt', Frog2Features, featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency, autopass=True)
                        testfeaturizer.in_frogged = testfrogger.out_frogged

                elif 'txtdir_test' in input_feeds.keys():
                    # could either be frogged or tokenized according to the config that is given as argument
                    if self.tokconfig:
                        testtokenizer = workflow.new_task('tokenizer_txtdir',Tokenize_txtdir,autopass=True,tokconfig=self.tokconfig,strip_punctuation=self.strip_punctuation)
                        testtokenizer.in_txtdir = input_feeds['txtdir_test']
                        testfeaturizer = workflow.new_task('featurizer_toktxtdir',Tokdir2Features,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                        testfeaturizer.in_tokdir = testtokenizer.out_toktxtdir                

                    elif self.frogconfig:
                        testfrogger = workflow.new_task('frogger_txtdir',Frog_txtdir,autopass=True,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                        testfrogger.in_txtdir = input_feeds['txtdir_test']
                        testfeaturizer = workflow.new_task('featurizer_frogtxtdir',Frogdir2Features,autopass=True,featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency)
                        testfeaturizer.in_frogdir = testfrogger.out_frogjsondir

                testinstances = testfeaturizer.out_features
                
            testvectorizer = workflow.new_task('testvectorizer',ApplyVectorizer,autopass=True,weight=self.weight,prune=self.prune)
            testvectorizer.in_test = testinstances
            testvectorizer.in_featureselection = trainvectorizer.out_featureselection

        return testvectorizer
