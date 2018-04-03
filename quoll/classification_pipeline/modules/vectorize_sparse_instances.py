
import numpy
from scipy import sparse
import pickle
from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.functions import vectorizer

class Balance_instances(Task):

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

class TrainApply_PCA(Task):

    in_train = InputSlot()
    in_test = InputSlot()

    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.pca.vectors.npz')

    def out_test(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.pca.vectors.npz')

    def out_pca(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.pca.model.pkl')
    
    def out_vocab(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.pca.vocabulary.txt')
    
    def run(self):

        # load vectorized traininstances
        loader = numpy.load(self.in_train().path)
        vectorized_traininstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape']).toarray()

        # load vectorized testinstances
        loader = numpy.load(self.in_test().path)
        vectorized_testinstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape']).toarray()

        # reduce dimensions using sklearn PCA
        vectorized_traininstances_pca, vectorized_testinstances_pca, pca_model, pca_vocab = vectorizer.trainapply_pca(vectorized_traininstances, vectorized_testinstances)

        # write traininstances to file
        vectorized_traininstances_pca = sparse.csr_matrix(vectorized_traininstances_pca)
        numpy.savez(self.out_train().path, data=vectorized_traininstances_pca.data, indices=vectorized_traininstances_pca.indices, indptr=vectorized_traininstances_pca.indptr, shape=vectorized_traininstances_pca.shape)

        # write testinstances to file
        vectorized_testinstances_pca = sparse.csr_matrix(vectorized_testinstances_pca)
        numpy.savez(self.out_test().path, data=vectorized_testinstances_pca.data, indices=vectorized_testinstances_pca.indices, indptr=vectorized_testinstances_pca.indptr, shape=vectorized_testinstances_pca.shape)
        
        # write pca model to file
        with open(self.out_pca().path, 'wb') as fid:
            pickle.dump(pca_model, fid)
        
        # write pca components to file
        with open(self.out_vocab().path, 'w', encoding='utf-8') as v_out:
            for feats in pca_vocab:
                v_out.write(' '.join([str(x) for x in feats]) + '\n')
            
class Vectorize_traininstances(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_vocabulary = InputSlot()

    weight = Parameter()
    prune = IntParameter()
    pca = BoolParameter()

    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.vectors.npz')

    def out_featureweights(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.feature_weights.txt')

    def out_topfeatures(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.topfeatures.txt')

    def out_labels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.vectorized.labels')

    def run(self):

        weight_functions = {'frequency':[vectorizer.return_document_frequency, False], 'binary':[vectorizer.return_document_frequency, vectorizer.return_binary_vectors], 'tfidf':[vectorizer.return_idf, vectorizer.return_tfidf_vectors], 'infogain':[vectorizer.return_infogain, vectorizer.return_infogain_vectors]}

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
        
        print('Loaded test instances, Shape:',featurized_instances.shape,', now loading topfeatures...')
 
        # load top features
        with open(self.in_topfeatures().path,'r',encoding='utf-8') as infile:
            lines = infile.read().split('\n')
            topfeatures = [line.split('\t') for line in lines]
        topfeatures_vocabulary = [x[0] for x in topfeatures]
        feature_weights = dict([(i, float(feature[1])) for i, feature in enumerate(topfeatures)])
        print('Loaded',len(topfeatures_vocabulary),'topfeatures, now loading sourcevocabulary...')

        # align the test instances to the top features to get the right indices
        with open(self.in_sourcevocabulary().path,'r',encoding='utf-8') as infile:
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


class Vectorize_mutually(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_vocabulary = InputSlot()
    in_test = InputSlot()
    in_sourcevocabulary = InputSlot()

    weight = Parameter(default='frequency')
    prune = IntParameter()
    balance = BoolParameter()

    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.vectors.npz')

    def out_test(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.vectors.npz')

    def out_featureweights(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.feature_weights.txt')

    def out_topfeatures(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.topfeatures.txt')

    def out_labels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.vectorized.labels')

    def run(self):

        weight_functions = {'frequency':[vectorizer.return_document_frequency, False], 'binary':[vectorizer.return_document_frequency, vectorizer.return_binary_vectors], 'tfidf':[vectorizer.return_idf, vectorizer.return_tfidf_vectors], 'infogain':[vectorizer.return_infogain, vectorizer.return_infogain_vectors]}

        # load featurized traininstances
        loader = numpy.load(self.in_train().path)
        featurized_traininstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load featurized testinstances
        loader = numpy.load(self.in_test().path)
        featurized_testinstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load trainlabels
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')

        # load vocabulary
        with open(self.in_vocabulary().path,'r',encoding='utf-8') as infile:
            vocabulary = infile.read().strip().split('\n')

        # balance instances by label frequency
        if self.balance:
            featurized_traininstances, trainlabels = vectorizer.balance_data(featurized_traininstances, trainlabels)

        # calculate feature_weight
        feature_weights = weight_functions[self.weight][0](featurized_traininstances, trainlabels)

        # vectorize instances
        if weight_functions[self.weight][1]:
            trainvectors = weight_functions[self.weight][1](featurized_traininstances, feature_weights)
        else:
            trainvectors = featurized_traininstances

        # prune features
        topfeatures = vectorizer.return_top_features(feature_weights, self.prune)
        topfeatures_vocabulary = []
        for index in topfeatures:
            topfeatures_vocabulary.append(vocabulary[index])

        # compress vectors
        compressed_trainvectors = vectorizer.compress_vectors(trainvectors, topfeatures)

        # align train and testvectors
        with open(self.in_sourcevocabulary().path,'r',encoding='utf-8') as infile:
            sourcevocabulary = infile.read().split('\n')
        print('Now aligning train and test vectors, shape compressed trainvectors:',compressed_trainvectors.shape,', number of topfeatures:',len(topfeatures_vocabulary))
        aligned_trainvectors, aligned_testvectors, topfeatures_vocabulary = vectorizer.align_vectors_mutually(compressed_trainvectors, featurized_testinstances, topfeatures_vocabulary, sourcevocabulary)
        print('Done. Trainvectors shape:',aligned_trainvectors.shape,', Testvectors shape:',aligned_testvectors.shape)

        # set feature weights
        if weight_functions[self.weight]:
            aligned_testvectors = weight_functions[self.weight][1](aligned_testvectors, feature_weights)

        # write instances to file
        numpy.savez(self.out_train().path, data=aligned_trainvectors.data, indices=aligned_trainvectors.indices, indptr=aligned_trainvectors.indptr, shape=aligned_trainvectors.shape)

        # write instances to file
        numpy.savez(self.out_test().path, data=aligned_testvectors.data, indices=aligned_testvectors.indices, indptr=aligned_testvectors.indptr, shape=aligned_testvectors.shape)

        # write feature weights to file
        with open(self.out_featureweights().path, 'w', encoding = 'utf-8') as w_out:
            outlist = []
            for key, value in sorted(feature_weights.items(), key = lambda k: k[0]):
                outlist.append('\t'.join([str(key), str(value)]))
            w_out.write('\n'.join(outlist))

        # write top features to file
        with open(self.out_topfeatures().path, 'w', encoding = 'utf-8') as t_out:
            # outlist = []
            # for feature in topfeatures_vocabulary:
            #     outlist.append([feature,  vocabulary[index], str(feature_weights[feature])]))
            t_out.write('\n'.join(topfeatures_vocabulary))

        # write labels to file
        with open(self.out_labels().path, 'w', encoding='utf-8') as l_out:
            l_out.write('\n'.join(trainlabels))


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

@registercomponent
class Vectorize_test(WorkflowComponent):

    testfile = Parameter()
    testvocabulary = Parameter()
    topfeatures = Parameter()

    weight = Parameter(default = 'frequency')

    def accepts(self):
        return [ ( InputFormat(self, format_id='test', extension='.features.npz',inputparameter='testfile'), InputFormat(self, format_id='testvocabulary', extension='.txt',inputparameter='testvocabulary'), InputFormat(self, format_id='topfeatures', extension='.txt',inputparameter='topfeatures')) ]

    def setup(self, workflow, input_feeds):

        test_vectors = workflow.new_task('vectorize_testinstances', Vectorize_testinstances, autopass=True, weight=self.weight)
        test_vectors.in_test = input_feeds['test']
        test_vectors.in_sourcevocabulary = input_feeds['testvocabulary']
        test_vectors.in_topfeatures = input_feeds['topfeatures']

        return test_vectors
