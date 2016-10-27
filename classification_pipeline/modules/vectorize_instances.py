
import numpy
from scipy import sparse
from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

import vectorizer

class Vectorize_traininstances(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_vocabulary = InputSlot()

    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()

    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.vectors.npz')

    def out_featureweights(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.feature_weights.txt')

    def out_topfeatures(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.topfeatures.txt')

    def out_labels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.vectorlabels')
    def run(self):
        
        weight_functions = {'frequency':[vectorizer.return_document_frequency, False], 'binary':[vectorizer.return_document_frequency, vectorizer.return_binary_vectors], 'tfidf':[vectorizer.return_idf, vectorizer.return_tfidf_vectors], 'infogain':[vectorizer.return_infogain, vectorizer.return_infogain_vectors]}

        # load featurized instances 
        loader = numpy.load(self.in_train().path)
        featurized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load trainlabels
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().split('\n')

        # load vocabulary
        with open(self.in_vocabulary().path,'r',encoding='utf-8') as infile:
            vocabulary = infile.read().split('\n')

        # balance instances by label frequency
        if self.balance:
            featurized_instances, trainlabels = vectorizer.balance_data(featurized_instances, trainlabels)

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

        # write labels to file
        with open(self.out_labels().path, 'w', encoding='utf-8') as l_out:
            l_out.write('\n'.join(trainlabels))

class Vectorize_testinstances(Task):

    in_test = InputSlot()
    in_sourcevocabulary = InputSlot()
    in_topfeatures = InputSlot()

    weight = Parameter(default='frequency')

    def out_test(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.vectors.npz')

    def run(self):

        weight_functions = {'frequency':False, 'binary':vectorizer.return_binary_vectors, 'tfidf':vectorizer.return_tfidf_vectors, 'infogain':vectorizer.return_infogain_vectors}

        # load featurized instances
        loader = numpy.load(self.in_test().path)
        featurized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
      
        # load top features
        with open(self.in_topfeatures().path,'r',encoding='utf-8') as infile:
            lines = infile.read().split('\n')
            topfeatures = [line.split('\t') for line in lines]
        topfeatures_vocabulary = [x[0] for x in topfeatures]
        feature_weights = dict([(i, float(feature[1])) for i, feature in enumerate(topfeatures)])
           
        # align the test instances to the top features to get the right indices
        with open(self.in_sourcevocabulary().path,'r',encoding='utf-8') as infile:
                sourcevocabulary = infile.read().split('\n')
        testvectors = vectorizer.align_vectors(featurized_instances, topfeatures_vocabulary, sourcevocabulary)

        # set feature weights
        if weight_functions[self.weight]:
            testvectors = weight_functions[self.weight](testvectors, feature_weights)

        # write instances to file
        numpy.savez(self.out_test().path, data=testvectors.data, indices=testvectors.indices, indptr=testvectors.indptr, shape=testvectors.shape)

@registercomponent
class Vectorize(WorkflowComponent):
 
    trainfile = Parameter()
    trainlabels_file = Parameter()
    trainvocabulary = Parameter()
   
    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter(default=False)
    
    def accepts(self):
        return [ ( InputFormat(self, format_id='train', extension='.features.npz', inputparameter='trainfile'), InputFormat(self, format_id='trainlabels', extension='.labels',inputparameter='trainlabels_file'), InputFormat(self, format_id='vocabulary', extension='.txt',inputparameter='trainvocabulary') ) ]

    def setup(self, workflow, input_feeds):

        train_vectors = workflow.new_task('vectorize_traininstances', Vectorize_traininstances, autopass=True, weight=self.weight, prune=self.prune, balance=self.balance)
        train_vectors.in_train = input_feeds['train']
        train_vectors.in_trainlabels = input_feeds['trainlabels']
        train_vectors.in_vocabulary = input_feeds['vocabulary']
        return train_vectors

@registercomponent
class Vectorize_traintest(WorkflowComponent):
    
    trainfile = Parameter()
    trainlabels_file = Parameter()
    testfile = Parameter()
    trainvocabulary = Parameter()
    testvocabulary = Parameter()
    weight = Parameter(default = 'frequency')
    prune = IntParameter(default = 5000)
    balance = BoolParameter(default = False)

    def accepts(self):
        return [ ( InputFormat(self, format_id='train', extension='.features.npz',inputparameter='trainfile'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels_file'), InputFormat(self, format_id='test', extension='.features.npz',inputparameter='testfile'), InputFormat(self, format_id='vocabulary', extension='.txt',inputparameter='trainvocabulary'), InputFormat(self, format_id='sourcevocabulary', extension='.txt',inputparameter='testvocabulary')) ]
                                
    def setup(self, workflow, input_feeds):

        train_vectors = workflow.new_task('vectorize_traininstances', Vectorize_traininstances, autopass=True, weight=self.weight, prune=self.prune, balance=self.balance)
        train_vectors.in_train = input_feeds['train']
        train_vectors.in_trainlabels = input_feeds['trainlabels']
        train_vectors.in_vocabulary = input_feeds['vocabulary']

        test_vectors = workflow.new_task('vectorize_testinstances', Vectorize_testinstances, autopass=True, weight=self.weight)
        test_vectors.in_test = input_feeds['test']
        test_vectors.in_sourcevocabulary = input_feeds['sourcevocabulary']
        test_vectors.in_topfeatures = train_vectors.out_topfeatures        

        return test_vectors
