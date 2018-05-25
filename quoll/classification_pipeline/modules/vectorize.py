
import numpy
from scipy import sparse
import os
import itertools
import pickle

from luiginlp.engine import Task, WorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.functions import vectorizer, docreader
from quoll.classification_pipeline.modules.featurize import Featurize

#################################################################
### Tasks #######################################################
#################################################################

class Balance(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()

    def in_vocabulary(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.vocabulary.txt')   

    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.balanced.features.npz')

    def out_labels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.balanced.labels')

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
        featureselection = vectorizer.return_featureselection(weight_functions['frequency'][0](featurized_instances, trainlabels), self.prune)

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
    in_train = InputSlot()

    weight = Parameter()
    prune = Parameter()
    balance = BoolParameter()
    
    def in_vocabulary(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.vocabulary.txt')

    def in_featureselection(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.featureselection.txt')

    def out_test(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.balance else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz')

    def out_featureselection(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.featureselection.txt' if self.balance else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.featureselection.txt')

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

        # write featureselection to file
        with open(self.out_featureselection().path, 'w', encoding = 'utf-8') as t_out:
            t_out.write('\n'.join(lines))
        
class VectorizeCsv(Task):

    in_csv = InputSlot()

    delimiter = Parameter()  
    
    def out_vectors(self):
        return self.outputfrominput(inputformat='csv', stripextension='.csv', addextension='.vectors.npz')

    def run(self):

        # load instances
        loader = docreader.Docreader()
        instances = loader.parse_csv(self.in_csv().path,delim=self.delimiter)
        instances_float = [[0.0 if feature == 'NA' else 0.0 if feature == '#NULL!' else float(feature.replace(',','.')) for feature in instance] for instance in instances]
        instances_sparse = sparse.csr_matrix(instances_float)

        # write instances to file
        numpy.savez(self.out_vectors().path, data=instances_sparse.data, indices=instances_sparse.indices, indptr=instances_sparse.indptr, shape=instances_sparse.shape)

class FitTransformScale(Task):

    in_vectors = InputSlot()

    def in_featureselection(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.featureselection.txt')       
    
    def out_vectors(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.scaled.vectors.npz')

    def out_scaler(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.scaler.pkl')

    def out_featureselection(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.scaled.featureselection.txt')       
    
    def run(self):

        # read vectors
        loader = numpy.load(self.in_vectors().path)
        vectors = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # read vocabulary
        with open(self.in_featureselection().path,'r',encoding='utf-8') as file_in:
            featureselection = file_in.read().strip().split('\n')
        
        # scale vectors
        scaler = vectorizer.fit_scale(vectors)
        scaled_vectors = vectorizer.scale_vectors(vectors,scaler)

        # write vectors
        numpy.savez(self.out_vectors().path, data=scaled_vectors.data, indices=scaled_vectors.indices, indptr=scaled_vectors.indptr, shape=scaled_vectors.shape)

        # write scaler
        with open(self.out_scaler().path, 'wb') as fid:
            pickle.dump(scaler, fid)

        # write vocabulary
        with open(self.out_featureselection().path,'w',encoding='utf-8') as out:
            out.write('\n'.join(featureselection))


class TransformScale(Task):

    in_vectors = InputSlot()
    in_scaler = InputSlot()

    def in_featureselection(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.featureselection.txt')       
    
    def out_vectors(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.scaled.vectors.npz')

    def out_featureselection(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.scaled.featureselection.txt')       

    
    def run(self):

        # read vectors
        loader = numpy.load(self.in_vectors().path)
        vectors = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # read vocabulary
        with open(self.in_featureselection().path,'r',encoding='utf-8') as file_in:
            featureselection = file_in.read().strip().split('\n')

        # read scaler
        with open(self.in_scaler().path, 'rb') as fid:
            scaler = pickle.load(fid)

        # scale vectors
        scaled_vectors = vectorizer.scale_vectors(vectors,scaler)

        # write vectors
        numpy.savez(self.out_vectors().path, data=scaled_vectors.data, indices=scaled_vectors.indices, indptr=scaled_vectors.indptr, shape=scaled_vectors.shape)

        # write vocabulary
        with open(self.out_featureselection().path,'w',encoding='utf-8') as out:
            out.write('\n'.join(featureselection))


class Combine(Task):

    in_vectors = InputSlot()
    in_vectors_append = InputSlot()

    normalize = Parameter()

    def in_vocabulary(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.featureselection.txt')

    def in_vocabulary_append(self):
        return self.outputfrominput(inputformat='vectors_append', stripextension='.vectors.npz', addextension='.featureselection.txt')   

    def out_featureselection(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.' + self.in_vectors_append().path.split('.')[-3] + '.featureselection.txt')   

    def out_combined(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.' + self.in_vectors_append().path.split('.')[-3] + '.vectors.npz')
    
    def run(self):

        # assert that vocabulary file exists (not checked in component)
        assert os.path.exists(self.in_vocabulary().path), 'Vocabulary file not found, make sure the file exists and/or change vocabulary path name to ' + self.in_vocabulary().path 
        
        # assert that vocabulary_append file exists (not checked in component)
        assert os.path.exists(self.in_vocabulary_append().path), 'Second vocabulary file not found, make sure the file exists and/or change vocabulary path name to ' + self.in_vocabulary_append().path 

        # load vocabulary
        with open(self.in_vocabulary().path,'r',encoding='utf-8') as infile:
            vocabulary = infile.read().strip().split('\n')
        
        # load vocabulary append
        with open(self.in_vocabulary_append().path,'r',encoding='utf-8') as infile:
            vocabulary_append = infile.read().strip().split('\n')

        # load vectors
        loader = numpy.load(self.in_vectors().path)
        vectors = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        V = vectors.toarray()
        
        # load vectors append
        loader = numpy.load(self.in_vectors_append().path)
        vectors_append = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        VA = vectors_append.toarray()
        
        # combine vocabularies
        vocabulary_combined = vocabulary + vocabulary_append

        # combine vectors
        vectors_combined = sparse.hstack([vectors,vectors_append]).tocsr()

        # scale
        if self.scale:
            instances_sparse = vectorizer.normalize_features(instances_sparse)

        # write vocabulary to file
        with open(self.out_featureselection().path, 'w', encoding='utf-8') as v_out:
            v_out.write('\n'.join(vocabulary_combined))

        # write combined vectors to file
        numpy.savez(self.out_combined().path, data=vectors_combined.data, indices=vectors_combined.indices, indptr=vectors_combined.indptr, shape=vectors_combined.shape)


class VectorizeFoldreporter(Task):

    in_predictions = InputSlot()
    in_bins = InputSlot()

    def out_vectors(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.bow.vectors.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.bow.featureselection.txt')
    
    def run(self):

        # open bin indices
        dr = docreader.Docreader()
        bins_str = dr.parse_csv(self.in_bins().path)
        indices = sum([[int(x) for x in bin] for bin in bins_str],[])

        # load predictions
        with open(self.in_predictions().path,'r',encoding='utf-8') as file_in:
            predictions = file_in.read().strip().split('\n')

        # generate prediction dict (to convert names to numbrers)
        predictiondict = {}
        for i,pred in enumerate(list(set(predictions))):
            predictiondict[pred] = 0

        # initialize vectorcolumn
        vectors = [[0]] * len(indices)

        # for each prediction
        for i,prediction in enumerate(predictions):
            index = indices[i]
            vectors[index][0] = predictiondict[prediction]
        vectors_csr = sparse.csr_matrix(vectors)

        # write output
        numpy.savez(self.out_vectors().path, data=vectors_csr.data, indices=vectors_csr.indices, indptr=vectors_csr.indptr, shape=vectors_csr.shape)

        with open(self.out_vocabulary().path,'w',encoding='utf-8') as out:
            out.write('BOW')
        
class VectorizePredictions(Task):

    in_predictions = InputSlot()

    def out_vectors(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.bow.vectors.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.bow.featureselection.txt')
    
    def run(self):

        # load predictions
        with open(self.in_predictions().path,'r',encoding='utf-8') as file_in:
            predictions = file_in.read().strip().split('\n')

        # generate prediction dict (to convert names to numbrers)
        predictiondict = {}
        for i,pred in enumerate(list(set(predictions))):
            predictiondict[pred] = 0

        # initialize vectorcolumn
        vectors = []

        # for each prediction
        for prediction in predictions:
            vectors.append([predictiondict[prediction]])
        vectors_csr = sparse.csr_matrix(vectors)

        # write output
        numpy.savez(self.out_vectors().path, data=vectors_csr.data, indices=vectors_csr.indices, indptr=vectors_csr.indptr, shape=vectors_csr.shape)

        with open(self.out_vocabulary().path,'w',encoding='utf-8') as out:
            out.write('BOW')


class FeaturizeTask(Task):

    in_pre_featurized = InputSlot()
    
    ngrams = Parameter()
    blackfeats = Parameter()
    lowercase = BoolParameter()
    minimum_token_frequency = IntParameter()
    featuretypes = Parameter()

    tokconfig = Parameter()
    frogconfig = Parameter()
    strip_punctuation = BoolParameter()

    def out_featurized(self):
        return self.outputfrominput(inputformat='pre_featurized', stripextension='.' + self.in_pre_featurized().task.extension, addextension='tokens.n_' + '_'.join(self.ngrams.split()) + '.min' + str(self.minimum_token_frequency) + '.lower_' + self.lowercase.__str__() + '.black_' + '_'.join(self.blackfeats.split()) + '.features.npz')
    
    def run(self):
        
        yield Featurize(inputfile=self.in_pre_featurized().path,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                
#################################################################
### Component ###################################################
#################################################################

@registercomponent
class Vectorize(WorkflowComponent):
    
    traininstances = Parameter()
    trainlabels = Parameter(default='xxx.xxx') # not obligatory, dummy extension to enable a pass
    testinstances = Parameter(default='xxx.xxx') # not obligatory, dummy extension to enable a pass
    
    # vectorizer parameters
    weight = Parameter(default = 'frequency') # options: frequency, binary, tfidf
    prune = IntParameter(default = 5000) # after ranking the topfeatures in the training set, based on frequency or idf weighting
    balance = BoolParameter()
    delimiter = Parameter(default=',')
    scale = BoolParameter()

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

        return [tuple(x) for x in numpy.array(numpy.meshgrid(*
            [
                (   
                InputFormat(self, format_id='vectorized_train',extension='.vectors.npz',inputparameter='traininstances'),
                InputFormat(self, format_id='featurized_train',extension='.features.npz',inputparameter='traininstances'),
                InputFormat(self, format_id='featurized_train_csv',extension='.csv',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txt',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txtdir',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.json',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.jsondir',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.txt',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.txtdir',inputparameter='traininstances')
                ),
                (
                InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels')
                ),
                (
                InputFormat(self, format_id='featurized_test',extension='.features.npz',inputparameter='testinstances'),
                InputFormat(self, format_id='featurized_test_csv',extension='.csv',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_featurized_test',extension='.tok.txt',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_featurized_test',extension='.tok.txtdir',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_featurized_test',extension='.frog.json',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_featurized_test',extension='.frog.jsondir',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_featurized_test',extension='.txt',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_featurized_test',extension='.txtdir',inputparameter='testinstances')
                ),
            ]
            )).T.reshape(-1,3)]  
    
    def setup(self, workflow, input_feeds):
        
        ######################
        ### Training phase ###
        ######################
        
        if 'featurized_train_csv' in input_feeds.keys():
            trainvectorizer_csv = workflow.new_task('train_vectorizer_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter)
            trainvectorizer_csv.in_csv = input_feeds['featurized_train_csv']

            if self.scale:
                trainvectorizer = workflow.new_task('scale_trainvectors',FitTransformScale,autopass=True)
                trainvectorizer.in_vectors = trainvectorizer_csv.out_vectors

            else:
                trainvectorizer = trainvectorizer_csv

        else:

            if 'vectorized_train' not in input_feeds.keys():

                labels = input_feeds['labels_train']
            
                if 'featurized_train' in input_feeds.keys():
                    traininstances = input_feeds['featurized_train']
            
                else: # pre_featurized
                    trainfeaturizer = workflow.new_task('featurize_train',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                    trainfeaturizer.in_pre_featurized = input_feeds['pre_featurized_train']

                    traininstances = trainfeaturizer.out_featurized

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

        if len(list(set(['featurized_test_csv','featurized_test_txt','featurized_test','pre_featurized_test']) & set(list(input_feeds.keys())))) > 0:
        
            if 'featurized_test_csv' in input_feeds.keys():
                testvectorizer_csv = workflow.new_task('vectorizer_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter)
                testvectorizer_csv.in_csv = input_feeds['featurized_test_csv']

                if self.scale:
                    testvectorizer = workflow.new_task('scale_testvectors',TransformScale,autopass=True)
                    testvectorizer.in_vectors = testvectorizer_csv.out_vectors
                    testvectorizer.in_scaler = trainvectorizer.out_scaler

                else:
                    testvectorizer = testvectorizer_csv

                return testvectorizer, trainvectorizer
        
            else:
                
                if 'featurized_test' in input_feeds.keys():
                    testinstances = input_feeds['featurized_test']
            
                else: # pre_featurized
                    testfeaturizer = workflow.new_task('featurize_test',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                    testfeaturizer.in_pre_featurized = input_feeds['pre_featurized_test']

                    testinstances = testfeaturizer.out_featurized
                    
                if 'vectorized_train' in input_feeds.keys():
                    trainvectors = input_feeds['vectorized_train']
 
                else:
                    trainvectors = trainvectorizer.out_train

                testvectorizer = workflow.new_task('testvectorizer',ApplyVectorizer,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
                testvectorizer.in_test = testinstances
                testvectorizer.in_train = trainvectors
                
                return testvectorizer
        
        else: # only train

            return trainvectorizer
