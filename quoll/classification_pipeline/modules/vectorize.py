
import numpy
from scipy import sparse
import pickle
import os

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.functions import vectorizer
from quoll.classification_pipeline.modules.preprocess import Tokenize_instances, Tokenize_txtdir, Frog_instances, Frog_txtdir
from quoll.classification_pipeline.modules.featurize import Tokenized2Features, Tokdir2Features, Frog2Features, Frogdir2Features

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
        
class VectorizeTrain(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    
    weight = Parameter()
    prune = IntParameter()

    def in_vocabulary(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.vocabulary.txt')   
     
    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz')

    def out_featureweights(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.weight_' + self.weight + '.prune_' + str(self.prune) + '.feature_weights.txt')

    def out_topfeatures(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.weight_' + self.weight + '.prune_' + str(self.prune) + '.topfeatures.txt')

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
        feature_weights = weight_functions[self.weight][0](featurized_instances, trainlabels)

        # vectorize instances
        if weight_functions[self.weight][1]:
            trainvectors = weight_functions[self.weight][1](featurized_instances, feature_weights)
        else:
            trainvectors = featurized_instances

        # prune features
        topfeatures = vectorizer.return_top_features(feature_weights, self.prune)

        # compress vectors
        trainvectors = vectorizer.compress_vectors(trainvectors, topfeatures)

        # write instances to file
        numpy.savez(self.out_train().path, data=trainvectors.data, indices=trainvectors.indices, indptr=trainvectors.indptr, shape=trainvectors.shape)

        # write feature weights to file
        with open(self.out_featureweights().path, 'w', encoding = 'utf-8') as w_out:
            outlist = []
            for key, value in sorted(feature_weights.items(), key = lambda k: k[0]):
                outlist.append('\t'.join([str(key), str(value)]))
            w_out.write('\n'.join(outlist))

        # write top features to file
        with open(self.out_topfeatures().path, 'w', encoding = 'utf-8') as t_out:
            outlist = []
            for index in topfeatures:
                outlist.append('\t'.join([vocabulary[index], str(feature_weights[index])]))
            t_out.write('\n'.join(outlist))

class VectorizeTest(Task):

    in_test = InputSlot()
    in_testvocabulary = InputSlot()
    in_topfeatures = InputSlot()

    weight = Parameter()
    prune = Parameter()

    def out_test(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz')

    def run(self):

        weight_functions = {
                            'frequency':False, 
                            'binary':vectorizer.return_binary_vectors, 
                            'tfidf':vectorizer.return_tfidf_vectors, 
                            'infogain':vectorizer.return_infogain_vectors
                            }

        # load featurized instances
        loader = numpy.load(self.in_test().path)
        featurized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        
        # print next steps (tends to take a while)
        print('Loaded test instances, Shape:',featurized_instances.shape,', now loading topfeatures...')
 
        # load top features
        with open(self.in_topfeatures().path,'r',encoding='utf-8') as infile:
            lines = infile.read().split('\n')
            topfeatures = [line.split('\t') for line in lines]
        topfeatures_vocabulary = [x[0] for x in topfeatures]
        feature_weights = dict([(i, float(feature[1])) for i, feature in enumerate(topfeatures)])
        print('Loaded',len(topfeatures_vocabulary),'topfeatures, now loading sourcevocabulary...')

        # align the test instances to the top features to get the right indices
        with open(self.in_testvocabulary().path,'r',encoding='utf-8') as infile:
            sourcevocabulary = infile.read().split('\n')
        print('Loaded sourcevocabulary of size',len(sourcevocabulary),', now aligning testvectors...')
        testvectors = vectorizer.align_vectors(featurized_instances, topfeatures_vocabulary, sourcevocabulary)
        print('Done. Shape of testvectors:',testvectors.shape,', now setting feature weights...')

        # set feature weights
        if weight_functions[self.weight]:
            testvectors = weight_functions[self.weight](testvectors, feature_weights)
        print('Done. Writing instances to file...')

        # write instances to file
        numpy.savez(self.out_test().path, data=testvectors.data, indices=testvectors.indices, indptr=testvectors.indptr, shape=testvectors.shape)

class VectorizeTxt(Task):

    in_txt = InputSlot()

    delimiter = Parameter()
    normalize = BoolParameter()

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

        # write instances to file
        numpy.savez(self.out_vectors().path, data=instances_sparse.data, indices=instances_sparse.indices, indptr=instances_sparse.indptr, shape=instances_sparse.shape)

class VectorizeCsv(Task):

    in_csv = InputSlot()

    delimiter = Parameter()
    normalize = BoolParameter()

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

    # featurizer parameters
    ngrams = Parameter(default='1 2 3')
    blackfeats = Parameter(default=False)
    lowercase = BoolParameter()    
    minimum_token_frequency = IntParameter(default=1)
    featuretypes = Parameter(default='tokens')

    # ucto / frog parameters
    tokconfig = Parameter(default=False)
    frogconfig = Parameter(default=False)
    strip_punctuation = BoolParameter(default=True)

    def accepts(self):
        return  [ ( InputFormat(self, format_id='featurized',extension='.features.npz',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), ( InputFormat(self, format_id='featurized_csv',extension='.features.csv',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), ( InputFormat(self, format_id='featurized_txt',extension='.features.txt',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), ( InputFormat(self, format_id='tokenized',extension='.tok.txt',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), ( InputFormat(self, format_id='tokdir',extension='.tok.txtdir',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), ( InputFormat(self, format_id='frogged',extension='.frog.json',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), ( InputFormat(self, format_id='frogdir',extension='.frog.jsondir',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), ( InputFormat(self, format_id='txt',extension='.txt',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), ( InputFormat(self, format_id='txtdir',extension='.txtdir',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ) ]
    
    def setup(self, workflow, input_feeds):

        if 'featurized_csv' in input_feeds.keys():
            vectorizer = workflow.new_task('vectorizer_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter,normalize=self.normalize)
            vectorizer.in_csv = input_feeds['featurized_csv']
        
        elif 'featurized_txt' in input_feeds.keys():
            vectorizer = workflow.new_task('vectorizer_txt',VectorizeTxt,autopass=True,delimiter=self.delimiter,normalize=self.normalize)
            vectorizer.in_txt = input_feeds['featurized_txt']

        else:

            labels = input_feeds['labels']
            
            if 'featurized' in input_feeds.keys():
                instances = input_feeds['featurized']
            
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

                instances = featurizer.out_features

            if self.balance:
                balancetask = workflow.new_task('BalanceTask',Balance,autopass=True)
                balancetask.in_train = instances
                balancetask.in_trainlabels = labels
                instances = balancetask.out_train
                labels = balancetask.out_labels
                                
            vectorizer = workflow.new_task('vectorizer',VectorizeTrain,autopass=True,weight=self.weight,prune=self.prune)
            vectorizer.in_train = instances
            vectorizer.in_trainlabels = labels

        return vectorizer

@registercomponent
class VectorizeTest(WorkflowComponent):
    
    instances = Parameter()

    # vectorizer parameters
    topfeatures = Parameter(default=False)
    weight = Parameter(default = 'frequency') # options: frequency, binary, tfidf
    prune = IntParameter(default = 5000) # after ranking the topfeatures in the training set, based on frequency or idf weighting
    balance = BoolParameter()
    pca = BoolParameter()
    normalize = BoolParameter()
    delimiter = Parameter(default=' ')

    # featurizer parameters
    ngrams = Parameter(default='1 2 3')
    blackfeats = Parameter(default=False)
    lowercase = BoolParameter()    
    minimum_token_frequency = IntParameter(default=1)
    featuretypes = Parameter(default=False)

    # ucto / frog parameters
    tokconfig = Parameter(default=False)
    frogconfig = Parameter(default=False)
    strip_punctuation = BoolParameter(default=True)

    def accepts(self):
        return  [ ( InputFormat(self, format_id='featurized',extension='.features.npz',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), ( InputFormat(self, format_id='featurized_csv',extension='.features.csv',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), ( InputFormat(self, format_id='featurized_txt',extension='.features.txt',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), ( InputFormat(self, format_id='tokenized',extension='.tok.txt',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), ( InputFormat(self, format_id='tokdir',extension='.tok.txtdir',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), ( InputFormat(self, format_id='frogged',extension='.frog.json',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), ( InputFormat(self, format_id='frogdir',extension='.frog.jsondir',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), ( InputFormat(self, format_id='txt',extension='.txt',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ), ( InputFormat(self, format_id='txtdir',extension='.txtdir',inputparameter='instances'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels') ) ]

    def setup(self, workflow, input_feeds):

        if 'featurized_csv' in input_feeds.keys():
            test_vectorizer = workflow.new_task('vectorizer_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter,normalize=self.normalize)
            test_vectorizer.in_csv = input_feeds['featurized_csv']
        
        elif 'featurized_txt' in input_feeds.keys():
            test_vectorizer = workflow.new_task('vectorizer_txt',VectorizeTxt,autopass=True,delimiter=self.delimiter,normalize=self.normalize)
            test_vectorizer.in_txt = input_feeds['featurized_txt']

        else:
            
            if 'featurized' in input_feeds.keys():
                test_instances = input_feeds['featurized']
            
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

                test_instances = featurizer.out_features

            if self.pca:
                pca_test = workflow.new_task('PCA_test',PCATest,autopass=True)
                pca_test.in_pca = pca_train.out_pca
                pca_test.in_vectors = test_instances
                test_instances = pca_test.out_vectors
                
            test_vectorizer = workflow.new_task('testvectorizer',VectorizeTest,autopass=True,weight=self.weight,prune=self.prune)
            test_vectorizer.in_test = test_instances
            test_vectorizer.in_topfeatures = input_feeds['topfeatures']

        return test_vectorizer
