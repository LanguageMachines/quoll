
import numpy
from scipy import sparse
import pickle

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.functions import vectorizer
from quoll.classification_pipeline.modules.tokenize_instances import Tokenize_instances, Tokenize_txtdir
from quoll.classification_pipeline.modules.frog_instances import Frog_instances, Frog_txtdir
from quoll.classification_pipeline.modules.featurize_instances import Tokenized2Features, Tokdir2Features, Frog2Features, Frogdir2Features

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

    def run(self):

        # load featurized traininstances
        loader = numpy.load(self.in_train().path)
        featurized_traininstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load featurized testinstances
        loader = numpy.load(self.in_test().path)
        featurized_testinstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

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

class PCATrain(Task):

    in_train = InputSlot()

    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.pca.vectors.npz')

    def out_pca(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.pca.model.pkl')
    
    def out_vocabulary(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.pca.vocabulary.txt')
    
    def run(self):

        # load vectorized traininstances
        loader = numpy.load(self.in_train().path)
        vectorized_traininstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape']).toarray()

        # reduce dimensions using sklearn PCA
        vectorized_traininstances_pca, pca_model, pca_vocab = vectorizer.train_pca(vectorized_traininstances)

        # write traininstances to file
        vectorized_traininstances_pca = sparse.csr_matrix(vectorized_traininstances_pca)
        numpy.savez(self.out_train().path, data=vectorized_traininstances_pca.data, indices=vectorized_traininstances_pca.indices, indptr=vectorized_traininstances_pca.indptr, shape=vectorized_traininstances_pca.shape)
        
        # write pca model to file
        with open(self.out_pca().path, 'wb') as fid:
            pickle.dump(pca_model, fid)
        
        # write pca components to file
        with open(self.out_vocabulary().path, 'w', encoding='utf-8') as v_out:
            for feats in pca_vocab:
                v_out.write(' '.join([str(x) for x in feats]) + '\n')

class PCATest(Task): # TODO: PCA test function

    in_vectors = InputSlot()
    in_pca = InputSlot()

    def out_vectors(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.pca.vectors.npz')

    def run(self):

        # load vectors
        loader = numpy.load(self.in_vectors().path)
        vectorized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape']).toarray()

        # load pca

        # reduce dimensions using sklearn PCA
        vectorized_instances_pca = vectorizer.test_pca(vectorized_instances)

        # write instances to file
        vectorized_instances_pca = sparse.csr_matrix(vectorized_instances_pca)
        numpy.savez(self.out_vectors().path, data=vectorized_instances_pca.data, indices=vectorized_instances_pca.indices, indptr=vectorized_instances_pca.indptr, shape=vectorized_instances_pca.shape)
        
class VectorizeTrain(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_vocabulary = InputSlot()

    weight = Parameter()
    prune = IntParameter()

    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.vectors.npz')

    def out_featureweights(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.feature_weights.txt')

    def out_topfeatures(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.topfeatures.txt')

    def run(self):

        weight_functions = {
                            'frequency':[vectorizer.return_document_frequency, False], 
                            'binary':[vectorizer.return_document_frequency, vectorizer.return_binary_vectors], 
                            'tfidf':[vectorizer.return_idf, vectorizer.return_tfidf_vectors]
                            }

        # load featurized instances
        loader = numpy.load(self.in_train().path)
        featurized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load trainlabels
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')

        # load vocabulary
        with open(self.in_vocabulary().path,'r',encoding='utf-8') as infile:
            vocabulary = infile.read().strip().split('\n')

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

    def out_test(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.vectors.npz')

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
        return self.outputfrominput(inputformat='txt', stripextension='.txt', addextension='.vectors.npz')

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
        return self.outputfrominput(inputformat='csv', stripextension='.csv', addextension='.vectors.npz')

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
    
    train_instances = Parameter()
    train_labels = Parameter()
    test_instances = Parameter()
    train_vocabulary = Parameter()
    test_vocabulary = Parameter()

    # vectorizer parameters
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

    # ucto / frog parameters
    featuretypes = Parameter(default=False)
    tokconfig = Parameter(default=False)
    frogconfig = Parameter(default=False)
    strip_punctuation = BoolParameter(default=True)

    def accepts(self):
        return [ ( 
                InputFormat(self, format_id='featurized_train',extension='.features.npz',inputparameter='train_instances'),
                InputFormat(self, format_id='featurized_test',extension='.features.npz',inputparameter='test_instances'),
                InputFormat(self, format_id='featurized_train_csv',extension='.csv',inputparameter='train_instances'),
                InputFormat(self, format_id='featurized_test_csv',extension='.csv',inputparameter='test_instances'),
                InputFormat(self, format_id='featurized_train_txt',extension='.txt',inputparameter='train_instances'),
                InputFormat(self, format_id='featurized_test_txt',extension='.txt',inputparameter='test_instances'),
                InputFormat(self, format_id='tokenized_train',extension='.tok.txt',inputparameter='train_instances'),
                InputFormat(self, format_id='tokenized_test',extension='.tok.txt',inputparameter='test_instances'),
                InputFormat(self, format_id='tokdir_train',extension='.tok.txtdir',inputparameter='train_instances'),
                InputFormat(self, format_id='tokdir_test',extension='.tok.txtdir',inputparameter='test_instances'),
                InputFormat(self, format_id='frogged_train',extension='.frog.json',inputparameter='train_instances'),
                InputFormat(self, format_id='frogged_test',extension='.frog.json',inputparameter='test_instances'),
                InputFormat(self, format_id='frogdir_train',extension='.frog.jsondir',inputparameter='train_instances'),
                InputFormat(self, format_id='frogdir_test',extension='.frog.jsondir',inputparameter='test_instances'),
                InputFormat(self, format_id='txt_train',extension='.txt',inputparameter='train_instances'),
                InputFormat(self, format_id='txt_test',extension='.txt',inputparameter='test_instances'),
                InputFormat(self, format_id='txtdir_train',extension='.txtdir',inputparameter='train_instances'),
                InputFormat(self, format_id='txtdir_test',extension='.txtdir',inputparameter='test_instances'),
                InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='train_labels'),
                InputFormat(self, format_id='vocabulary_train',extension='.txt',inputparameter='train_vocabulary'),
                InputFormat(self, format_id='vocabulary_test',extension='.txt',inputparameter='test_vocabulary')
                ) ]

    def setup(self, workflow, input_feeds):

        ########################################################
        ### Train phase #########################################
        ########################################################

        if 'featurized_train_csv' in input_feeds.keys():
            trainvectorizer = workflow.new_task('trainvectorizer_csv',autopass=True,VectorizeCsv,delimiter=self.delimiter,normalize=self.normalize)
            trainvectorizer.in_csv = input_feeds['featurized_train_csv']
        
        elif 'featurized_train_txt' in input_feeds.keys():
            trainvectorizer = workflow.new_task('trainvectorizer_txt',autopass=True,VectorizeTxt,delimiter=self.delimiter,normalize=self.normalize)
            trainvectorizer.in_txt = input_feeds['featurized_train_txt']

        else:

            train_labels = input_feeds['labels_train']
            
            if 'featurized_train' in input_feeds.keys():
                train_instances = input_feeds['featurized_train']
                train_vocabulary = input_feeds['vocabulary_train']
            
            else: # earlier stage
                if 'tokenized_train' in input_feeds.keys():
                    trainfeaturizer = workflow.new_task('trainfeaturizer_tokens',autopass=True,Tokenized2Features,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
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
                        traintokenizer = workflow.new_task('traintokenizer_txt',TokenizeTask,autopass=True,tokconfig=self.tokconfig,strip_punctuation=self.strip_punctuation)
                        traintokenizer.in_txt = input_feeds['txt_train']
                        trainfeaturizer = workflow.new_task('trainfeaturizer_toktxt',Tokenized2Features,autopass=True,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                        trainfeaturizer.in_tokenized = traintokenizer.out_tokenized

                    elif self.frogconfig:
                        trainfrogger = workflow.new_task('trainfrogger_txt',Frog_instances,autopass=True,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                        trainfrogger.in_txt = input_feeds['txt']
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
                        trainfeaturizer = workflow.new_task('trainfeaturizer_frogtxtdir',Frogdir2Features, featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency, autopass=True)
                        trainfeaturizer.in_frogdir = trainfrogger.out_frogjsondir

                train_instances = trainfeaturizer.out_features
                train_vocabulary = trainfeaturizer.out_vocabulary

                if self.balance:
                    balancetask = workflow.new_task('BalanceTask',Balance,autopass=True)
                    balancetask.in_train = train_instances
                    balancetask.in_trainlabels = train_labels
                    train_instances = balancetask.out_train
                    train_labels = balancetask.out_labels
                
                if self.pca:
                    pca_train = workflow.new_task('PCA_train',PCATrain,autopass=True):
                    pca_train.in_train = train_instances
                    train_instances = pca_train.out_train
                    train_vocabulary = pca_train.out_vocabulary
                
                train_vectorizer = workflow.new_task('trainvectorizer',VectorizeTrain,autopass=True,weight=self.weight,prune=self.prune)
                train_vectorizer.in_train = train_instances
                train_vectorizer.in_trainlabels = train_labels
                train_vectorizer.in_vocabulary = train_vocabulary

        ########################################################
        ### Test phase #########################################
        ########################################################

        test_instances = False
        test_vocabulary = False

        if 'featurized_test_csv' in input_feeds.keys():
            test_instances = 'csv'
            testvectorizer = workflow.new_task('trainvectorizer_csv',autopass=True,VectorizeCsv,delimiter=self.delimiter,normalize=self.normalize)
            testvectorizer.in_csv = input_feeds['featurized_test_csv']
        
        elif 'featurized_test_txt'  in input_feeds.keys():
            test_instances = 'txt'
            testvectorizer = workflow.new_task('testvectorizer_txt',autopass=True,VectorizeTxt,delimiter=self.delimiter,normalize=self.normalize)
            testvectorizer.in_txt = input_feeds['featurized_test_txt']      

        else:

            if 'featurized_test' in input_feeds.keys():
                test_instances = input_feeds['featurized_test']
                test_vocabulary = input_feeds['vocabulary_test']
            
            elif 'tokenized_test' in input_feeds.keys() or 'tokdir_test' in input_feeds.keys() or 'frogged_test' in input_feeds.keys() or 'frogdir_test' in input_feeds.keys() or 'txt_test' in input_feeds.keys() or 'txtdir_test' in input_feeds.keys(): # earlier stage
                if 'tokenized_test' in input_feeds.keys():
                    testfeaturizer = workflow.new_task('testfeaturizer_tokens',autopass=True,Tokenized2Features,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
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
                        testtokenizer = workflow.new_task('testtokenizer_txt',TokenizeTask,autopass=True,tokconfig=self.tokconfig,strip_punctuation=self.strip_punctuation)
                        testtokenizer.in_txt = input_feeds['txt_test']
                        testfeaturizer = workflow.new_task('testfeaturizer_toktxt',Tokenized2Features,autopass=True,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                        testfeaturizer.in_tokenized = testtokenizer.out_tokenized

                    elif self.frogconfig:
                        testfrogger = workflow.new_task('testfrogger_txt',Frog_instances,autopass=True,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                        testfrogger.in_txt = input_feeds['txt']
                        testfeaturizer = workflow.new_task('testfeaturizer_frogtxt',Frog2Features,autopass=True,featuretypes=self.featuretypes,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,autopass=True)
                        testfeaturizer.in_frogged = testfrogger.out_frogged

                elif 'txtdir_test' in input_feeds.keys():
                    # could either be frogged or tokenized according to the config that is given as argument
                    if self.tokconfig:
                        testtokenizer = workflow.new_task('testtokenizer_txtdir',Tokenize_txtdir,autopass=True,tokconfig=self.tokconfig,strip_punctuation=self.strip_punctuation)
                        testtokenizer.in_txtdir = input_feeds['txtdir_test']
                        testfeaturizer = workflow.new_task('testfeaturizer_toktxtdir',Tokdir2Features,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency)
                        testfeaturizer.in_tokdir = testtokenizer.out_toktxtdir              

                    elif self.frogconfig:
                        testfrogger = workflow.new_task('testfrogger_txtdir',Frog_txtdir,autopass=True,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                        testfrogger.in_txtdir = input_feeds['txtdir_test']
                        testfeaturizer = workflow.new_task('testfeaturizer_frogtxtdir',Frogdir2Features, featuretypes=self.featuretypes, ngrams=self.ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase, minimum_token_frequency=self.minimum_token_frequency, autopass=True)
                        testfeaturizer.in_frogdir = testfrogger.out_frogjsondir

                test_instances = testfeaturizer.out_features
                test_vocabulary = testfeaturizer.out_vocabulary

        if test_instances:

            if not test_instances in ['txt','csv']:

                if self.pca:
                    pca_test = workflow.new_task('PCA_test',PCATest,autopass=True):
                    pca_test.in_pca = pca_train.out_pca
                    pca_test.in_vectors = test_instances
                    test_instances = pca_test.out_vectors
                    test_vocabulary = train_vocabulary
                
                test_vectorizer = workflow.new_task('testvectorizer',VectorizeTest,autopass=True,weight=self.weight)
                test_vectorizer.in_test = test_instances
                test_vectorizer.in_testvocabulary = test_vocabulary
                test_vectorizer.in_topfeatures = train_vectorizer.out_topfeatures

            return test_vectorizer

        else:

            return train_vectorizer
