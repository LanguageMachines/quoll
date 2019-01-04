
import numpy
from scipy import sparse
import os
import itertools
import pickle

from luiginlp.engine import Task, WorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.functions import vectorizer, featselector, quoll_helpers, docreader
from quoll.classification_pipeline.modules.featurize import Featurize

#################################################################
### Tasks #######################################################
#################################################################

class FitVectorizer(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()

    vectorize_parameters = Parameter()

    def in_vocabulary(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.vocabulary.txt')   
     
    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.vectors.npz')

    def out_labels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.vectors.labels')

    def out_featureweights(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.featureweights.txt')

    def out_featureselection(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.featureselection.txt')

    def run(self):

        kwargs = quoll_helpers.decode_task_input(['vectorize'],[self.vectorize_parameters])

        weight_functions = {
            'frequency':[vectorizer.return_document_frequency, False], 
            'binary':[vectorizer.return_document_frequency, vectorizer.return_binary_vectors], 
            'tfidf':[vectorizer.return_idf, vectorizer.return_tfidf_vectors]
        }

        selection_functions = {
            'fcbf':featselector.FCBF(),
            'mrmr_linear':featselector.MRMRLinear()
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
        featureweights = weight_functions[kwargs['weight']][0](featurized_instances, trainlabels)

        # vectorize instances
        if weight_functions[kwargs['weight']][1]:
            trainvectors = weight_functions[kwargs['weight']][1](featurized_instances, featureweights)
        else:
            trainvectors = featurized_instances

        # prune features
        featureselection = vectorizer.return_featureselection(weight_functions['frequency'][0](featurized_instances, trainlabels), kwargs['prune'])

        # compress vectors
        trainvectors = vectorizer.compress_vectors(trainvectors, featureselection)

        # select features
        if kwargs['select']:
            selectionclass = selection_functions[kwargs['selector']]
            trainvectors, weights_selected_features, indices_selected_features = selectionclass.fit_transform(trainvectors, trainlabels, kwargs['select_threshold'])    
            featureselection = numpy.array(featureselection)[indices_selected_features].tolist()
            featureweights = dict(zip(featureselection,weights_selected_features))
            
        # balance instances by label frequency
        if kwargs['balance']:
            trainvectors, trainlabels = vectorizer.balance_data(trainvectors, trainlabels)

        # write instances to file
        numpy.savez(self.out_train().path, data=trainvectors.data, indices=trainvectors.indices, indptr=trainvectors.indptr, shape=trainvectors.shape)

        # write trainlabels to file
        with open(self.out_labels().path, 'w', encoding='utf-8') as l_out:
            l_out.write('\n'.join(trainlabels))

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

    def in_vocabulary(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.vocabulary.txt')

    def in_featureselection(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.featureselection.txt')

    def out_test(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.vectors.npz')

    def out_featureselection(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.featureselection.txt')

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
        try:
            featureweights = dict([(i, float(feature[1])) for i, feature in enumerate(featureselection)])
        except:
            featureweights = False
        print('Loaded',len(featureselection_vocabulary),'selected features, now loading sourcevocabulary...')

        # align the test instances to the top features to get the right indices
        with open(self.in_vocabulary().path,'r',encoding='utf-8') as infile:
            vocabulary = infile.read().strip().split('\n')
        print('Loaded sourcevocabulary of size',len(vocabulary),', now aligning testvectors...')
        testvectors = vectorizer.align_vectors(featurized_instances, featureselection_vocabulary, vocabulary)
        print('Done. Shape of testvectors:',testvectors.shape,', now setting feature weights...')

        # set feature weights
        if featureweights:
            if weight_functions[self.weight]:
                testvectors = weight_functions[self.weight](testvectors, featureweights)
            print('Done. Writing instances to file...')

        # write instances to file
        numpy.savez(self.out_test().path, data=testvectors.data, indices=testvectors.indices, indptr=testvectors.indptr, shape=testvectors.shape)

        # write featureselection to file
        with open(self.out_featureselection().path, 'w', encoding = 'utf-8') as t_out:
            t_out.write('\n'.join(lines))
        
class TransformCsv(Task):

    in_csv = InputSlot()

    delimiter = Parameter()
    
    def out_features(self):
        return self.outputfrominput(inputformat='csv', stripextension='.csv', addextension='.features.npz')

    def run(self):

        # load instances
        loader = docreader.Docreader()
        instances = loader.parse_csv(self.in_csv().path,delim=self.delimiter)
        instances_float = [[0.0 if feature == 'NA' else 0.0 if feature == '#NULL!' else float(feature.replace(',','.')) for feature in instance] for instance in instances]
        instances_sparse = sparse.csr_matrix(instances_float)

        # write instances to file
        numpy.savez(self.out_features().path, data=instances_sparse.data, indices=instances_sparse.indices, indptr=instances_sparse.indptr, shape=instances_sparse.shape)

class Combine(Task):

    in_vectors = InputSlot()
    in_vectors_append = InputSlot()
    
    def in_vocabulary(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.featureselection.txt')

    def in_vocabulary_append(self):
        return self.outputfrominput(inputformat='vectors_append', stripextension='.vectors.npz', addextension='.featureselection.txt')   

    def out_featureselection(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.combined.featureselection.txt')   

    def out_combined(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.combined.vectors.npz')
    
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
        
        # load vectors append
        loader = numpy.load(self.in_vectors_append().path)
        vectors_append = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        
        # combine vocabularies
        vocabulary_combined = vocabulary + vocabulary_append

        # combine vectors
        vectors_combined = sparse.hstack([vectors,vectors_append]).tocsr()

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

        # generate prediction dict (to convert names to numbers)
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


class VectorizeFoldreporterProbs(Task):

    in_full_predictions = InputSlot()
    in_bins = InputSlot()

    include_labels = Parameter()

    def out_vectors(self):
        return self.outputfrominput(inputformat='full_predictions', stripextension='.full_predictions.txt', addextension='.bow.vectors.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='full_predictions', stripextension='.full_predictions.txt', addextension='.bow.featureselection.txt')
    
    def run(self):

        # open bin indices
        dr = docreader.Docreader()
        bins_str = dr.parse_csv(self.in_bins().path)
        indices = sum([[int(x) for x in bin] for bin in bins_str],[])

        # load full predictions
        with open(self.in_full_predictions().path) as infile:
            lines = [line.split('\t') for line in infile.read().strip().split('\n')]
        label_order = lines[0]
        full_predictions = lines[1:]

        # extract labels to include
        if self.include_labels == 'all':
            included_labels = label_order
        else:
            included_labels = self.include_labels.split()
            if len(list(set(included_labels) & set(label_order))) != len(included_labels):
                print('Included labels for bow features (',included_labels,') does not match predicted labels (',label_order,'), exiting program.')
                quit() 

        # initialize vectorcolumn
        vector = [0] * len(included_labels)
        vectors = [vector] * len(indices)

        # for each prediction
        for i,full_prediction in enumerate(full_predictions):
            index = indices[i]
            for j,label in enumerate(included_labels):
                label_index = label_order.index(label)
                vectors[index][j] = float(full_prediction[label_index])
        vectors_csr = sparse.csr_matrix(vectors)

        # write output
        numpy.savez(self.out_vectors().path, data=vectors_csr.data, indices=vectors_csr.indices, indptr=vectors_csr.indptr, shape=vectors_csr.shape)

        with open(self.out_vocabulary().path,'w',encoding='utf-8') as out:
            for label in included_labels:
                out.write(label + '_prediction_prob')
            
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

        # generate prediction dict (to convert names to numbers)
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

class VectorizePredictionsProbs(Task):
 
    in_full_predictions = InputSlot()

    include_labels = Parameter()

    def out_vectors(self):
        return self.outputfrominput(inputformat='full_predictions', stripextension='.full_predictions.txt', addextension='.bow.vectors.npz')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='full_predictions', stripextension='.full_predictions.txt', addextension='.bow.featureselection.txt')
    
    def run(self):

        # load full predictions
        with open(self.in_full_predictions().path) as infile:
            lines = [line.split('\t') for line in infile.read().strip().split('\n')]
        label_order = lines[0]
        full_predictions = lines[1:]

        # extract labels to include
        if self.include_labels == 'all':
            included_labels = label_order
        else:
            included_labels = self.include_labels.split()
            if len(list(set(included_labels) & set(label_order))) != len(included_labels):
                print('Included labels for bow features (',included_labels,') does not match predicted labels (',label_order,'), exiting program.')
                quit() 

        # initialize vectorcolumn
        vector = [0] * len(included_labels)
        vectors = [vector] * len(full_predictions)

        # for each prediction
        for i,full_prediction in enumerate(full_predictions):
            for j,label in enumerate(included_labels):
                label_index = label_order.index(label)
                vectors[i][j] = float(full_prediction[label_index])
        vectors_csr = sparse.csr_matrix(vectors)

        # write output
        numpy.savez(self.out_vectors().path, data=vectors_csr.data, indices=vectors_csr.indices, indptr=vectors_csr.indptr, shape=vectors_csr.shape)

        with open(self.out_vocabulary().path,'w',encoding='utf-8') as out:
            for label in included_labels:
                out.write(label + '_prediction_prob')

class FeaturizeTask(Task):

    in_pre_featurized = InputSlot()
    
    featurize_parameters = Parameter()
    preprocess_parameters = Parameter()

    def out_featurized(self):
        return self.outputfrominput(inputformat='pre_featurized', stripextension='.' + self.in_pre_featurized().task.extension, addextension='.features.npz')
    
    def run(self):

        if self.complete(): # necessary as it will not complete otherwise
            return True
        kwargs = quoll_helpers.decode_task_input(['featurize','preprocess'],[self.featurize_parameters,self.preprocess_parameters])
        yield Featurize(inputfile=self.in_pre_featurized().path,**kwargs)
                
#################################################################
### Component ###################################################
#################################################################

@registercomponent
class Vectorize(WorkflowComponent):
    
    train = Parameter()
    trainlabels = Parameter(default='xxx.xxx') # not obligatory, dummy extension to enable a pass
    test = Parameter(default='xxx.xxx') # not obligatory, dummy extension to enable a pass
    
    # vectorizer parameters
    weight = Parameter(default = 'frequency') # options: frequency, binary, tfidf
    prune = IntParameter(default = 5000) # after ranking the topfeatures in the training set, based on frequency or idf weighting
    delimiter = Parameter(default=',')
    select = BoolParameter()
    selector = Parameter(default=False)
    select_threshold = Parameter(default=False)
    balance = BoolParameter()

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
                InputFormat(self, format_id='vectorized_train',extension='.vectors.npz',inputparameter='train'),
                InputFormat(self, format_id='featurized_train',extension='.features.npz',inputparameter='train'),
                InputFormat(self, format_id='featurized_train_csv',extension='.csv',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txt',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txtdir',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.json',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.jsondir',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.txt',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.txtdir',inputparameter='train')
                ),
                (
                InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels')
                ),
                (
                InputFormat(self, format_id='vectorized_test',extension='.vectors.npz',inputparameter='test'),
                InputFormat(self, format_id='featurized_test',extension='.features.npz',inputparameter='test'),
                InputFormat(self, format_id='featurized_test_csv',extension='.csv',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.tok.txt',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.tok.txtdir',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.frog.json',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.frog.jsondir',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.txt',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.txtdir',inputparameter='test')
                ),
            ]
            )).T.reshape(-1,3)]  
    
    def setup(self, workflow, input_feeds):
        
        task_args = quoll_helpers.prepare_task_input(['preprocess','featurize','vectorize'],workflow.param_kwargs)

        ######################
        ### Training phase ###
        ######################

        labels = False if not 'labels_train' in input_feeds.keys() else input_feeds['labels_train']
        
        if 'vectorized_train' in input_feeds.keys():
            traininstances = input_feeds['vectorized_train']

        else: 

            if 'featurized_train_csv' in input_feeds.keys():
                traincsvtransformer = workflow.new_task('train_transformer_csv',TransformCsv,autopass=True,delimiter=self.delimiter)
                traincsvtransformer.in_csv = input_feeds['featurized_train_csv']
                trainfeatures = traincsvtransformer.out_features
            elif 'featurized_train' in input_feeds.keys():
                trainfeatures = input_feeds['featurized_train']
            else: # pre_featurized
                trainfeaturizer = workflow.new_task('featurize_train',FeaturizeTask,autopass=False,preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'])
                trainfeaturizer.in_pre_featurized = input_feeds['pre_featurized_train']

                trainfeatures = trainfeaturizer.out_featurized

            # assert that labels are inputted, in order to run the vectorizer
            assert labels, 'Cannot run vectorizer without trainlabels as inputfile...' 
                
            trainvectorizer = workflow.new_task('vectorizer',FitVectorizer,autopass=True,vectorize_parameters=task_args['vectorize'])
            trainvectorizer.in_train = trainfeatures
            trainvectorizer.in_trainlabels = labels

            traininstances = trainvectorizer.out_train
            labels = trainvectorizer.out_labels

        ######################
        ### Testing phase ####
        ######################

        if set(['vectorized_test','featurized_test_csv','featurized_test_txt','featurized_test','pre_featurized_test']) & set(list(input_feeds.keys())):
                
            if 'vectorized_test' in input_feeds.keys():
                testinstances=input_feeds['vectorized_test']

            else:
            
                if 'featurized_test_csv' in input_feeds.keys():
                    testcsvtransformer = workflow.new_task('test_transformer_csv',TransformCsv,autopass=True,delimiter=self.delimiter)
                    testcsvtransformer.in_csv = input_feeds['featurized_test_csv']
                    testfeatures = testcsvtransformer.out_features
                elif 'featurized_test' in input_feeds.keys():
                    testfeatures = input_feeds['featurized_test']
                else: # pre_featurized
                    testfeaturizer = workflow.new_task('featurize_test',FeaturizeTask,autopass=False,preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'])
                    testfeaturizer.in_pre_featurized = input_feeds['pre_featurized_test']

                    testfeatures = testfeaturizer.out_featurized
                    
                testvectorizer = workflow.new_task('testvectorizer',ApplyVectorizer,autopass=True,weight=self.weight)
                testvectorizer.in_test = testfeatures
                testvectorizer.in_train = traininstances
                
            return testvectorizer
        
        else: # only train

            return trainvectorizer
