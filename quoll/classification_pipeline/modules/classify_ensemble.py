
import os
import numpy
from scipy import sparse
import pickle
import math
import random
from collections import defaultdict

from luiginlp.engine import Task, WorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.modules.validate import Validate
from quoll.classification_pipeline.modules.report import ClassifyTask, TrainTask
from quoll.classification_pipeline.modules.classify import Classify
from quoll.classification_pipeline.modules.vectorize import FeaturizeTask, TransformCsv, VectorizeFoldreporter, VectorizePredictions

from quoll.classification_pipeline.functions import quoll_helpers, vectorizer, docreader

#################################################################
### Tasks #######################################################
#################################################################

class EnsembleTrainClf(Task):

    in_directory = InputSlot()
    in_instances = InputSlot()
    in_labels = InputSlot()
    in_docs = InputSlot()
    
    classifier = Parameter()
    linear_raw = BoolParameter()
    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()

    def in_vocabulary(self):
        return self.outputfrominput(inputformat='instances', stripextension='.features.npz', addextension='.vocabulary.txt')   

    def in_nominal_labels(self):
        return self.outputfrominput(inputformat='labels', stripextension='.raw.labels' if self.linear_raw else '.labels', addextension='.labels')   
            
    def out_ensemble_clf(self):
        return self.outputfrominput(inputformat='directory', stripextension='.ensemble', addextension='.ensemble/' + str(self.classifier))    

    def out_instances(self):
        return self.outputfrominput(inputformat='directory', stripextension='.ensemble', addextension='.ensemble/' + str(self.classifier) + '/instances.features.npz')        

    def out_labels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.ensemble', addextension='.ensemble/' + str(self.classifier) + '/instances.raw.labels' if (self.linear_raw and self.classifier == 'linreg') else '.ensemble/' + str(self.classifier) + '/instances.labels')

    def out_nominal_labels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.ensemble', addextension='.ensemble/' + str(self.classifier) + '/instances.labels')

    def out_vocabulary(self):
        return self.outputfrominput(inputformat='directory', stripextension='.ensemble', addextension='.ensemble/' + str(self.classifier) + '/instances.vocabulary.txt')

    def out_docs(self):
        return self.outputfrominput(inputformat='directory', stripextension='.ensemble', addextension='.ensemble/' + str(self.classifier) + '/instances.docs.txt')

    def out_bins(self):
        return self.outputfrominput(inputformat='directory', stripextension='.ensemble', addextension='.ensemble/' + str(self.classifier) + '/instances.raw.nfoldcv.bins.csv' if (self.linear_raw and self.classifier == 'linreg') else '.ensemble/' + str(self.classifier) + '/instances.nfoldcv.bins.csv')

    def out_predictions(self):
        return self.outputfrominput(inputformat='directory', stripextension='.ensemble', addextension='.ensemble/' + str(self.classifier) + '/instances.validated.predictions.txt')

    def run(self):

        # print('CLASSIFIER',self.classifier,'LINEAR RAW',self.linear_raw)
        
        if self.complete(): # needed as it will not complete otherwise
            return True
        
        # make ensemble directory
        self.setup_output_dir(self.out_ensemble_clf().path)

        # open instances
        loader = numpy.load(self.in_instances().path)
        instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load vocabulary
        with open(self.in_vocabulary().path,'r',encoding='utf-8') as infile:
            vocabulary = infile.read().strip().split('\n')

        # open labels
        with open(self.in_labels().path,'r',encoding='utf-8') as infile:
            labels = numpy.array(infile.read().strip().split('\n'))

        # open documents
        with open(self.in_docs().path,'r',encoding='utf-8') as infile:
            documents = numpy.array(infile.read().strip().split('\n'))

        # write data
        if self.linear_raw:
            # open labels
            with open(self.in_nominal_labels().path,'r',encoding='utf-8') as infile:
                nominal_labels = numpy.array(infile.read().strip().split('\n'))    
            with open(self.out_nominal_labels().path,'w',encoding='utf-8') as outfile:
                outfile.write('\n'.join(nominal_labels))
            # print('writing nominal labels',nominal_labels,'to',self.out_nominal_labels().path)
        numpy.savez(self.out_instances().path, data=instances.data, indices=instances.indices, indptr=instances.indptr, shape=instances.shape)
        with open(self.out_vocabulary().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(vocabulary))
        if not(self.linear_raw and not self.classifier == 'linreg'): 
            with open(self.out_labels().path,'w',encoding='utf-8') as outfile:
                outfile.write('\n'.join(labels))
        with open(self.out_docs().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(documents))

        print('Gathering train values for classifier',self.classifier)
        kwargs = quoll_helpers.decode_task_input(['ga','classify','vectorize'],[self.ga_parameters,self.classify_parameters,self.vectorize_parameters])
        kwargs['ensemble'] = False
        kwargs['classifier'] = self.classifier
        kwargs['n'] = 5
        # if self.classifier == 'svm':
        #     print('CLASSIFIER',self.classifier,'BOOLEAN RAW',self.linear_raw,'NOMINAL LABELS',nominal_trainlabels)
        #     exit()
        if self.linear_raw and not self.classifier == 'linreg':
                kwargs['linear_raw'] = False
                yield Validate(instances=self.out_instances().path,labels=self.out_nominal_labels().path,docs=self.out_docs().path,**kwargs)
        else:
            yield Validate(instances=self.out_instances().path,labels=self.out_labels().path,docs=self.out_docs().path,**kwargs)


class EnsembleTrainClfs(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()

    ensemble_clfs = Parameter()
    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()

    def in_docs(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.txt')
    
    def out_ensembledir(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.ensemble')
    
    def run(self):

        # make ensemble directory
        self.setup_output_dir(self.out_ensembledir().path)

        # run ensemble classifiers
        for ensemble_clf in self.ensemble_clfs.split():
            yield RunEnsembleTrainClf(directory=self.out_ensembledir().path,instances=self.in_train().path,labels=self.in_trainlabels().path,docs=self.in_docs().path,classifier=ensemble_clf,ga_parameters=self.ga_parameters,classify_parameters=self.classify_parameters,vectorize_parameters=self.vectorize_parameters)

class EnsembleTrainVectorizer(Task):

    in_ensembledir = InputSlot()
    in_labels = InputSlot()

    ensemble_clfs = Parameter()
    balance = BoolParameter()
    linear_raw = BoolParameter()
    
    def in_nominal_labels(self):
        return self.outputfrominput(inputformat='labels', stripextension='.raw.labels' if self.linear_raw else '.labels', addextension='.labels')   

    def out_vectors(self):
        return self.outputfrominput(inputformat='ensembledir', stripextension='.ensemble', addextension='.ensemble.vectors.npz')

    def out_labels(self):
        return self.outputfrominput(inputformat='ensembledir', stripextension='.ensemble', addextension='.ensemble.vectors.labels')

    def out_featurenames(self):
        return self.outputfrominput(inputformat='ensembledir', stripextension='.ensemble', addextension='.ensemble.featureselection.txt')  

    def run(self):
        
        if self.complete(): # necessary as it will not complete otherwise
            return True

        # collect prediction files
        featurenames = []
        all_indices = []
        all_predictions = []
        for clf in self.ensemble_clfs.split():
            featurenames.append(clf)
            prediction_file = self.in_ensembledir().path + '/' + clf + '/instances.validated.predictions.txt'
            bins_file = self.in_ensembledir().path + '/instances.raw.nfoldcv.bins.csv' if self.linear_raw and clf == 'linreg' else self.in_ensembledir().path + '/instances.nfoldcv.bins.csv'        
            # open bin indices
            dr = docreader.Docreader()
            bins_str = dr.parse_csv(binsfile)
            all_indices.append = sum([[int(x) for x in bin] for bin in bins_str],[])
            # load predictions
            with open(prediction_file,'r',encoding='utf-8') as file_in:
                all_predictions.append(file_in.read().strip().split('\n'))

        # generate prediction dict (to convert names to numbers)
        predictiondict = {}
        for i,pred in enumerate(sorted(list(set(sum(all_predictions,[]))))):
            predictiondict[pred] = i

        vectors = []
        # for each set of predictions
        for i,predictions in enumerate(all_predictions):
            # initialize vectorcolumn
            vector = []
            # for each prediction
            indices = all_indices[i]
            for j,prediction in enumerate(predictions):
                index = indices[j]
                vector.append([index,predictiondict[prediction]])
            vector_sorted = sorted(vector,key = lambda k : k[0])
            vector_final = [x[1] for x in vector_sorted]
            vector_csr = sparse.csr_matrix(vector_final.transpose())
            vectors.append(vector_csr)
            
        # combine and write vectors
        ensemblevectors = sparse.hstack(vectors).tocsr()
        with open(self.in_nominal_labels().path,'r',encoding='utf-8') as infile:
            ensemblelabels = infile.read().strip().split('\n')
        if self.balance:
            ensemblevectors, ensemblelabels = vectorizer.balance_data(ensemblevectors, ensemblelabels)
        numpy.savez(self.out_vectors().path, data=ensemblevectors.data, indices=ensemblevectors.indices, indptr=ensemblevectors.indptr, shape=ensemblevectors.shape)
        with open(self.out_labels().path, 'w', encoding='utf-8') as l_out:
            l_out.write('\n'.join(ensemblelabels))
        with open(self.out_featurenames().path,'w',encoding='utf-8') as out:
            out.write('\n'.join(featurenames))


class EnsemblePredictClf(Task):

    in_directory = InputSlot()
    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_test = InputSlot()
    
    classifier = Parameter()
    linear_raw = BoolParameter()
    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()

    def in_trainvocabulary(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.vocabulary.txt')   

    def in_testvocabulary(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.vocabulary.txt')   

    def in_nominal_labels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.raw.labels' if self.linear_raw else '.labels', addextension='.labels')   
            
    def out_ensemble_clf(self):
        return self.outputfrominput(inputformat='directory', stripextension='.ensemble', addextension='.ensemble/' + str(self.classifier))    

    def out_train(self):
        return self.outputfrominput(inputformat='directory', stripextension='.ensemble', addextension='.ensemble/' + str(self.classifier) + '/train.features.npz')        

    def out_trainvocabulary(self):
        return self.outputfrominput(inputformat='directory', stripextension='.ensemble', addextension='.ensemble/' + str(self.classifier) + '/train.vocabulary.txt')

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.ensemble', addextension='.ensemble/' + str(self.classifier) + '/train.raw.labels' if (self.linear_raw and self.classifier == 'linreg') else '.ensemble/' + str(self.classifier) + '/train.labels')

    def out_nominal_labels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.ensemble', addextension='.ensemble/' + str(self.classifier) + '/train.labels')

    def out_test(self):
        return self.outputfrominput(inputformat='directory', stripextension='.ensemble', addextension='.ensemble/' + str(self.classifier) + '/test.features.npz')        

    def out_testvocabulary(self):
        return self.outputfrominput(inputformat='directory', stripextension='.ensemble', addextension='.ensemble/' + str(self.classifier) + '/test.vocabulary.txt')

    def out_predictions(self):
        return self.outputfrominput(inputformat='directory', stripextension='.ensemble', addextension='.ensemble/' + str(self.classifier) + '/test.predictions.txt')

    def run(self):
        
        if self.complete(): # needed as it will not complete otherwise
            return True
        
        # make ensemble directory
        self.setup_output_dir(self.out_ensemble_clf().path)

        # open train
        loader = numpy.load(self.in_train().path)
        train = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load trainvocabulary
        with open(self.in_trainvocabulary().path,'r',encoding='utf-8') as infile:
            trainvocabulary = infile.read().strip().split('\n')

        # open trainlabels
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = numpy.array(infile.read().strip().split('\n'))

        # open test
        loader = numpy.load(self.in_test().path)
        test = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load trainvocabulary
        with open(self.in_testvocabulary().path,'r',encoding='utf-8') as infile:
            testvocabulary = infile.read().strip().split('\n')

        # write data
        if self.linear_raw:
            # open labels
            with open(self.in_nominal_labels().path,'r',encoding='utf-8') as infile:
                nominal_trainlabels = numpy.array(infile.read().strip().split('\n'))        
            with open(self.out_nominal_labels().path,'w',encoding='utf-8') as outfile:
                outfile.write('\n'.join(nominal_trainlabels))
        numpy.savez(self.out_train().path, data=train.data, indices=train.indices, indptr=train.indptr, shape=train.shape)
        with open(self.out_trainvocabulary().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(trainvocabulary))
        if not(self.linear_raw and not self.classifier == 'linreg'): 
            with open(self.out_trainlabels().path,'w',encoding='utf-8') as outfile:
                outfile.write('\n'.join(trainlabels))
        numpy.savez(self.out_test().path, data=test.data, indices=test.indices, indptr=test.indptr, shape=test.shape)
        with open(self.out_testvocabulary().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(testvocabulary))
        print('Gathering prediction values for classifier',self.classifier)
        kwargs = quoll_helpers.decode_task_input(['ga','classify','vectorize'],[self.ga_parameters,self.classify_parameters,self.vectorize_parameters])
        kwargs['ensemble'] = False
        kwargs['classifier'] = self.classifier
        # if self.classifier == 'svm':
        #     print('CLASSIFIER',self.classifier,'BOOLEAN RAW',self.linear_raw,'NOMINAL LABELS',nominal_trainlabels)
        #     exit()
        if self.linear_raw and not self.classifier == 'linreg':
            kwargs['linear_raw'] = False                
            yield Classify(train=self.out_train().path,trainlabels=self.out_nominal_labels().path,test=self.out_test().path,**kwargs)
        else:
            yield Classify(train=self.out_train().path,trainlabels=self.out_trainlabels().path,test=self.out_test().path,**kwargs)

class EnsemblePredictClfs(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_test = InputSlot()

    ensemble_clfs = Parameter()
    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()
    
    def out_ensembledir(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.ensemble')

    def run(self):

        # make ensemble directory
        self.setup_output_dir(self.out_ensembledir().path)

        # run ensemble classifiers
        for ensemble_clf in self.ensemble_clfs.split():
            yield RunEnsemblePredictClf(directory=self.out_ensembledir().path,train=self.in_train().path,trainlabels=self.in_trainlabels().path,test=self.in_test().path,classifier=ensemble_clf,ga_parameters=self.ga_parameters,classify_parameters=self.classify_parameters,vectorize_parameters=self.vectorize_parameters)

class EnsemblePredictVectorizer(Task):

    in_ensembledir = InputSlot()

    ensemble_clfs = Parameter()
    linear_raw = BoolParameter()

    def out_vectors(self):
        return self.outputfrominput(inputformat='ensembledir', stripextension='.ensemble', addextension='.ensemble.vectors.npz')

    def out_featurenames(self):
        return self.outputfrominput(inputformat='ensembledir', stripextension='.ensemble', addextension='.ensemble.featureselection.txt')  

    def run(self):
        
        if self.complete(): # necessary as it will not complete otherwise
            return True

        # collect prediction files
        featurenames = []
        all_indices = []
        all_predictions = []
        for clf in self.ensemble_clfs.split():
            featurenames.append(clf)
            prediction_file = self.in_ensembledir().path + '/' + clf + '/test.translated.predictions.txt' if self.linear_raw and clf == 'linreg' else self.in_ensembledir().path + '/' + clf + '/test.predictions.txt'
            # load predictions
            with open(prediction_file,'r',encoding='utf-8') as file_in:
                all_predictions.append(file_in.read().strip().split('\n'))

        # generate prediction dict (to convert names to numbers)
        predictiondict = {}
        for i,pred in enumerate(sorted(list(set(sum(all_predictions,[]))))):
            predictiondict[pred] = i

        vectors = []
        # for each set of predictions
        for predictions in all_predictions:
            # initialize vectorcolumn
            vector = []
            # for each prediction
            for i,prediction in enumerate(predictions):
                vector.append(predictiondict[prediction])
            vector_csr = sparse.csr_matrix(vector).transpose()
            vectors.append(vector_csr)
            
        # combine and write vectors
        ensemblevectors = sparse.hstack(vectors).tocsr()
        numpy.savez(self.out_vectors().path, data=ensemblevectors.data, indices=ensemblevectors.indices, indptr=ensemblevectors.indptr, shape=ensemblevectors.shape)
        with open(self.out_featurenames().path,'w',encoding='utf-8') as out:
            out.write('\n'.join(featurenames))


class EnsembleTrain(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()

    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()
    featurize_parameters = Parameter()
    preprocess_parameters = Parameter()
    linear_raw = BoolParameter()

    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.'.join(self.in_train().path.split('.')[-2:]) if (self.in_train().path[-3:] == 'npz' or self.in_train().path[-7:-4] == 'tok') else '.' + self.in_train().path.split('.')[-1], addextension='.ensemble.model.pkl')

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='train', stripextension='.'.join(self.in_train().path.split('.')[-2:]) if (self.in_train().path[-3:] == 'npz' or self.in_train().path[-7:-4] == 'tok') else '.' + self.in_train().path.split('.')[-1], addextension='.ensemble.vectors.labels')

    def run(self):

        if self.complete(): # necessary as it will not complete otherwise
            return True

        kwargs = quoll_helpers.decode_task_input(['classify','ga','vectorize','featurize','preprocess'],[self.classify_parameters,self.ga_parameters,self.vectorize_parameters,self.featurize_parameters,self.preprocess_parameters])        
        yield Ensemble(train=self.in_train().path,trainlabels=self.in_trainlabels().path,**kwargs)


class EnsembleTrainTest(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_test = InputSlot()

    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()
    featurize_parameters = Parameter()
    preprocess_parameters = Parameter()
    linear_raw = BoolParameter()

    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.'.join(self.in_test().path.split('.')[-2:]) if (self.in_test().path[-3:] == 'npz' or self.in_test().path[-7:-4] == 'tok') else '.' + self.in_test().path.split('.')[-1], addextension='.ensemble.model.pkl')

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='train', stripextension='.'.join(self.in_test().path.split('.')[-2:]) if (self.in_test().path[-3:] == 'npz' or self.in_test().path[-7:-4] == 'tok') else '.' + self.in_test().path.split('.')[-1], addextension='.ensemble.vectors.labels')

    def out_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.'.join(self.in_test().path.split('.')[-2:]) if (self.in_test().path[-3:] == 'npz' or self.in_test().path[-7:-4] == 'tok') else '.' + self.in_test().path.split('.')[-1], addextension='.ensemble.predictions.txt')        

    def run(self):
        
        if self.complete(): # necessary as it will not complete otherwise
            return True

        kwargs = quoll_helpers.decode_task_input(['classify','ga','vectorize','featurize','preprocess'],[self.classify_parameters,self.ga_parameters,self.vectorize_parameters,self.featurize_parameters,self.preprocess_parameters])        
        yield Ensemble(train=self.in_train().path,trainlabels=self.in_trainlabels().path,test=self.in_test().path,**kwargs)


##################################################################
### Components ###################################################
##################################################################

@registercomponent
class RunEnsembleTrainClf(WorkflowComponent):

    directory = Parameter()
    instances = Parameter()
    labels = Parameter()
    docs = Parameter()

    classifier = Parameter()
    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()

    def accepts(self):
        return [ ( 
            InputFormat(self,format_id='directory',extension='.ensemble',inputparameter='directory'), 
            InputFormat(self,format_id='instances',extension='.features.npz',inputparameter='instances'), 
            InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), 
            InputFormat(self,format_id='docs',extension='.txt',inputparameter='docs'),
        ) ]
 
    def setup(self, workflow, input_feeds):

        kwargs = quoll_helpers.decode_task_input(['classify'],[self.classify_parameters])
        ensemble_train = workflow.new_task(
            'ensemble_train', EnsembleTrainClf, autopass=True,classifier=self.classifier,linear_raw=kwargs['linear_raw'],ga_parameters=self.ga_parameters,classify_parameters=self.classify_parameters,vectorize_parameters=self.vectorize_parameters
        )
        ensemble_train.in_directory = input_feeds['directory']
        ensemble_train.in_instances = input_feeds['instances']
        ensemble_train.in_labels = input_feeds['labels']
        ensemble_train.in_docs = input_feeds['docs']

        return ensemble_train


@registercomponent
class RunEnsemblePredictClf(WorkflowComponent):

    directory = Parameter()
    train = Parameter()
    trainlabels = Parameter()
    test = Parameter()

    classifier = Parameter()
    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()

    def accepts(self):
        return [ ( 
            InputFormat(self,format_id='directory',extension='.ensemble',inputparameter='directory'), 
            InputFormat(self,format_id='train',extension='.features.npz',inputparameter='train'), 
            InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), 
            InputFormat(self,format_id='test',extension='.features.npz',inputparameter='test'),
        ) ]
 
    def setup(self, workflow, input_feeds):

        kwargs = quoll_helpers.decode_task_input(['classify'],[self.classify_parameters])
        ensemble_predict = workflow.new_task(
            'ensemble_predict', EnsemblePredictClf, autopass=True,classifier=self.classifier,linear_raw=kwargs['linear_raw'],ga_parameters=self.ga_parameters,classify_parameters=self.classify_parameters,vectorize_parameters=self.vectorize_parameters
        )
        ensemble_predict.in_directory = input_feeds['directory']
        ensemble_predict.in_train = input_feeds['train']
        ensemble_predict.in_trainlabels = input_feeds['trainlabels']
        ensemble_predict.in_test = input_feeds['test']
        
        return ensemble_predict


@registercomponent
class Ensemble(WorkflowComponent):

    train = Parameter()
    trainlabels = Parameter()
    test = Parameter(default = 'xxx.xxx')

    # featureselection parameters
    ga = BoolParameter()
    num_iterations = IntParameter(default=300)
    population_size = IntParameter(default=100)
    elite = Parameter(default='0.1')
    crossover_probability = Parameter(default='0.9')
    mutation_rate = Parameter(default='0.3')
    tournament_size = IntParameter(default=2)
    n_crossovers = IntParameter(default=1)
    stop_condition = IntParameter(default=5)
    weight_feature_size = Parameter(default='0.0')
    steps = IntParameter(default=1)
    sampling = BoolParameter() # repeated resampling to prevent overfitting
    samplesize = Parameter(default='0.8') # size of trainsample
    
    # classifier parameters
    classifier = Parameter(default='naive_bayes')
    ensemble = Parameter(default=False)
    ordinal = BoolParameter()
    jobs = IntParameter(default=1)
    iterations = IntParameter(default=10)
    scoring = Parameter(default='roc_auc') # optimization metric for grid search
    linear_raw = BoolParameter()
    scale = BoolParameter()
    min_scale = Parameter(default='0')
    max_scale = Parameter(default='1')

    random_clf = Parameter(default='equal')
    
    nb_alpha = Parameter(default='1.0')
    nb_fit_prior = BoolParameter()
    
    svm_c = Parameter(default='1.0')
    svm_kernel = Parameter(default='linear')
    svm_gamma = Parameter(default='0.1')
    svm_degree = Parameter(default='1')
    svm_class_weight = Parameter(default='balanced')

    lr_c = Parameter(default='1.0')
    lr_solver = Parameter(default='liblinear')
    lr_dual = BoolParameter()
    lr_penalty = Parameter(default='l2')
    lr_multiclass = Parameter(default='ovr')
    lr_maxiter = Parameter(default='1000')

    linreg_fit_intercept = Parameter(default='1')
    linreg_normalize = Parameter(default='0')
    linreg_copy_X = Parameter(default='1')

    xg_booster = Parameter(default='gbtree') # choices: ['gbtree', 'gblinear']
    xg_silent = Parameter(default='1') # set to '1' to mute printed info on progress
    xg_learning_rate = Parameter(default='0.1') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_min_child_weight = Parameter(default='1') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_max_depth = Parameter(default='6') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_gamma = Parameter(default='0') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_max_delta_step = Parameter(default='0')
    xg_subsample = Parameter(default='1') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_colsample_bytree = Parameter(default='1.0') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_reg_lambda = Parameter(default='1')
    xg_reg_alpha = Parameter(default='0') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_scale_pos_weight = Parameter('1')
    xg_objective = Parameter(default='binary:logistic') # choices: ['binary:logistic', 'multi:softmax', 'multi:softprob']
    xg_seed = Parameter(default='7')
    xg_n_estimators = Parameter(default='100') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 

    knn_n_neighbors = Parameter(default='3')
    knn_weights = Parameter(default='uniform')
    knn_algorithm = Parameter(default='auto')
    knn_leaf_size = Parameter(default='30')
    knn_metric = Parameter(default='euclidean')
    knn_p = IntParameter(default=2)

    perceptron_alpha = Parameter(default='1.0')

    tree_class_weight = Parameter(default=False)

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
    strip_punctuation = BoolParameter(default=True)
    
    def accepts(self):
        return [tuple(x) for x in numpy.array(numpy.meshgrid(*
            [
                (
                InputFormat(self, format_id='train_csv',extension='.csv',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.features.npz',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.tok.txt',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.tok.txtdir',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.frog.json',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.frog.jsondir',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.txt',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.txtdir',inputparameter='train')
                ),
                (
                InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels')
                ),
                (
                InputFormat(self, format_id='test_csv',extension='.csv',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.features.npz',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.tok.txt',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.tok.txtdir',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.frog.json',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.frog.jsondir',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.txt',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.txtdir',inputparameter='test')
                ),
            ]
            )).T.reshape(-1,3)]
 
    def setup(self, workflow, input_feeds):

        task_args = quoll_helpers.prepare_task_input(['preprocess','featurize','vectorize','classify','ga'],workflow.param_kwargs)
        
        trainlabels = input_feeds['labels_train']

        ######################
        ### featurize ########
        ######################

        if 'train_csv' in input_feeds.keys():
            traincsvtransformer = workflow.new_task('train_transformer_csv',TransformCsv,autopass=True,delimiter=self.delimiter)
            traincsvtransformer.in_csv = input_feeds['train_csv']
            trainfeatures = traincsvtransformer.out_features
        else:
            trainfeaturizer = workflow.new_task('featurize_train',FeaturizeTask,autopass=False,preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'])
            trainfeaturizer.in_pre_featurized = input_feeds['train']
            trainfeatures = trainfeaturizer.out_featurized

        if set(['test','test_csv']) & set(list(input_feeds.keys())):
            
            if 'test_csv' in input_feeds.keys():
                testcsvtransformer = workflow.new_task('test_transformer_csv',TransformCsv,autopass=True,delimiter=self.delimiter)
                testcsvtransformer.in_csv = input_feeds['test_csv']
                testfeatures = testcsvtransformer.out_features
            else:
                testfeaturizer = workflow.new_task('featurize_test',FeaturizeTask,autopass=False,preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'])
                testfeaturizer.in_pre_featurized = input_feeds['test']
                testfeatures = testfeaturizer.out_featurized

        ######################
        ### Training phase ###
        ######################

        ensemble_trainer = workflow.new_task('train_ensemble',EnsembleTrainClfs,autopass=False,ensemble_clfs=self.ensemble,ga_parameters=task_args['ga'],classify_parameters=task_args['classify'],vectorize_parameters=task_args['vectorize'])
        ensemble_trainer.in_train = trainfeatures
        ensemble_trainer.in_trainlabels = trainlabels

        ensemble_train_vectorizer = workflow.new_task('vectorize_ensemble_train',EnsembleTrainVectorizer,autopass=True,ensemble_clfs=self.ensemble,balance=self.balance,linear_raw=self.linear_raw)
        ensemble_train_vectorizer.in_ensembledir = ensemble_trainer.out_ensembledir
        ensemble_train_vectorizer.in_labels = trainlabels

        if set(['test','test_csv']) & set(list(input_feeds.keys())):

            ensemble_predictor = workflow.new_task('predict_ensemble',EnsemblePredictClfs,autopass=True,ensemble_clfs=self.ensemble,ga_parameters=task_args['ga'],classify_parameters=task_args['classify'],vectorize_parameters=task_args['vectorize'])
            ensemble_predictor.in_train = trainfeatures
            ensemble_predictor.in_trainlabels = trainlabels
            ensemble_predictor.in_test = testfeatures

            ensemble_predict_vectorizer = workflow.new_task('vectorize_ensemble_predict',EnsemblePredictVectorizer,autopass=True,ensemble_clfs=self.ensemble)
            ensemble_predict_vectorizer.in_ensembledir = ensemble_predictor.out_ensembledir

            classifier = workflow.new_task('classify',ClassifyTask,autopass=True,
                preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],
                classify_parameters=task_args['classify'],ga_parameters=task_args['ga'],linear_raw=False     
            )
            classifier.in_train = ensemble_train_vectorizer.out_vectors
            classifier.in_test = ensemble_predict_vectorizer.out_vectors
            classifier.in_trainlabels = ensemble_train_vectorizer.out_labels

            return classifier

        else:

            trainer = workflow.new_task('train_ensemble',TrainTask,autopass=True,
                preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],
                classify_parameters=task_args['classify'],ga_parameters=task_args['ga'],linear_raw=False     
            )
            trainer.in_train = ensemble_train_vectorizer.out_vectors
            trainer.in_trainlabels = ensemble_train_vectorizer.out_labels

            return trainer       
