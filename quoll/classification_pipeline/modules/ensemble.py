
import os
import numpy
from scipy import sparse
import pickle
import math
import random
from collections import defaultdict

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.modules.validate import Validate
from quoll.classification_pipeline.modules.classify import Classify, Train, Predict, TranslatePredictions
from quoll.classification_pipeline.modules.vectorize import FeaturizeTask, TransformCsv, PredictionsToVectors

from quoll.classification_pipeline.functions import quoll_helpers, vectorizer

#################################################################
### Tasks #######################################################
#################################################################

class EnsembleTrainTask(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()

    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()
    
    def in_docs(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.txt')

    def in_vocabulary(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.vocabulary.txt')

    def out_ensembledir(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.ensemble')

    def out_vectors(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.ensemble.vectors.npz')

    def out_labels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.ensemble.labels')

    def out_featurenames(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.ensemble.featurenames.txt')

    def run(self):
        
        if self.complete(): # necessary as it will not complete otherwise
            return True

        # make ensemble directory
        self.setup_output_dir(self.out_ensembledir().path)

        # extract ensemble classifiers
        kwargs = quoll_helpers.decode_task_input(['ga','classify'],[self.ga_parameters,self.classify_parameters])
        ensemble_clfs = kwargs['ensemble'].split()
        kwargs['ensemble'] = False
        kwargs['n'] = 5

        vectors = []
        featurenames = []
        # for each ensemble clf
        for ensemble_clf in ensemble_clfs:
            # prepare files
            kwargs['classifier'] = ensemble_clf
            clfdir = self.out_ensembledir().path + '/' + ensemble_clf
            os.mkdir(clfdir)
            instances = clfdir + '/instances.npz'
            labels = clfdir + '/instances.labels'
            docs = clfdir + '/docs.txt'
            vocabulary = clfdir + '/instances.vocabulary.txt'
            os.system('cp ' + self.in_train().path + ' ' + instances)
            os.system('cp ' + self.in_trainlabels().path + ' ' + labels)
            os.system('cp ' + self.in_docs().path + ' ' + docs)
            os.system('cp ' + self.in_vocabulary().path + ' ' + vocabulary)
            yield Validate(instances=instances,labels=labels,docs=docs,**kwargs)
            yield PredictionsToVectors(bins=bins,predictions=clfdir + '/instances.validated.predictions.txt',featurename=ensemble_clf)
            featurenames.append(ensemble_clf)
            loader = numpy.load(clfdir + '/instances.validated.predictions.vectors.npz')
            vectors.append(sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape']))

        # combine and write vectors
        ensemblevectors = sparse.hstack(vectors)
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            ensemblelabels = infile.read().strip().split('\n')
        if kwargs['balance']:
            ensemblevectors, ensemblelabels = vectorizer.balance_data(ensemblevectors, ensemblelabels)
        numpy.savez(self.out_vectors().path, data=ensemblevectors.data, indices=ensemblevectors.indices, indptr=ensemblevectors.indptr, shape=ensemblevectors.shape)
        with open(self.out_labels().path, 'w', encoding='utf-8') as l_out:
            l_out.write('\n'.join(ensemblelabels))

        # combine and write featurenames
        with open(self.out_featurenames().path,'w',encoding='utf-8') as out:
            out.write('\n'.join(featurenames))


class EnsemblePredictTask(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_test = InputSlot()

    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()

    def in_train_vocabulary(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.vocabulary.txt')

    def in_test_vocabulary(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.vocabulary.txt') 

    def out_ensembledir(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.ensemble')

    def out_vectors(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.ensemble.vectors.npz')

    def out_featurenames(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.ensemble.featurenames.txt')

    def run(self):
        
        if self.complete(): # necessary as it will not complete otherwise
            return True

        # make ensemble directory
        self.setup_output_dir(self.out_ensembledir().path)

        # extract ensemble classifiers
        kwargs = quoll_helpers.decode_task_input(['ga','classify','vectorize'],[self.ga_parameters,self.classify_parameters,self.vectorize_parameters])
        ensemble_clfs = kwargs['ensemble'].split()
        kwargs['ensemble'] = False

        vectors = []
        featurenames = []
        # for each ensemble clf
        for ensemble_clf in ensemble_clfs:
            # prepare files
            kwargs['classifier'] = ensemble_clf
            clfdir = self.out_ensembledir().path + '/' + ensemble_clf
            os.mkdir(clfdir)
            train = clfdir + '/train.features.npz'
            trainlabels = clfdir + '/train.labels'
            test = clfdir + '/test.features.npz'
            os.system('cp ' + self.in_train().path + ' ' + train)
            os.system('cp ' + self.in_trainlabels().path + ' ' + labels)
            os.system('cp ' + self.in_test().path + ' ' + test)
            yield Classify(train=self.in_train().path,trainlabels=self.in_trainlabels().path,test=self.in_test().path,**kwargs)
            yield PredictionsToVectors(predictions=clfdir+'/test.predictions.txt')
            featurenames.append(ensemble_clf)
            loader = numpy.load(clfdir + '/test.predictions.vectors.npz')
            vectors.append(sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape']))

        # combine and write vectors
        ensemblevectors = sparse.hstack(vectors)
        numpy.savez(self.out_vectors().path, data=ensemblevectors.data, indices=ensemblevectors.indices, indptr=ensemblevectors.indptr, shape=ensemblevectors.shape)

        # combine and write featurenames
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

    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.ensemble.vectors.npz')

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.ensemble.labels')

    def run(self):
        
        if self.complete(): # necessary as it will not complete otherwise
            return True

        kwargs = quoll_helpers.decode_task_input(['classify','ga','vectorize'],[self.classify_parameters,self.ga_parameters,self.vectorize_parameters,self.featurize_parameters,self.preprocess_parameters])        
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

    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.ensemble.vectors.npz')

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.ensemble.labels')
    
    def out_test(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.ensemble.vectors.npz')

    def out_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.ensemble.predictions.txt')        

    def run(self):

        if self.complete(): # necessary as it will not complete otherwise
            return True

        kwargs = quoll_helpers.decode_task_input(['classify','ga','vectorize'],[self.classify_parameters,self.ga_parameters,self.vectorize_parameters,self.featurize_parameters,self.preprocess_parameters])        
        yield Ensemble(train=self.in_train().path,trainlabels=self.in_trainlabels().path,test=self.in_test().path,**kwargs)


##################################################################
### Components ###################################################
##################################################################

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
                InputFormat(self, format_id='modeled_train',extension ='.model.pkl',inputparameter='train'),
                InputFormat(self, format_id='vectors_train',extension='.vectors.npz',inputparameter='train'),
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
                InputFormat(self, format_id='vectors_test',extension='.vectors.npz',inputparameter='test'),
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

        traininstances = False
        model = False

        ######################
        ### featurize ########
        ######################

        if 'modeled_train' in input_feeds.keys():
            model = input_feeds['modeled_train']
        if 'vectors_train' in input_feeds.keys():
            traininstances = input_feeds['vectors_train']
        else:
            elif 'train_csv' in input_feeds.keys():
                traincsvtransformer = workflow.new_task('train_transformer_csv',TransformCsv,autopass=True,delimiter=self.delimiter)
                traincsvtransformer.in_csv = input_feeds['featurized_train_csv']
                trainfeatures = traincsvtransformer.out_features
            else:
                trainfeaturizer = workflow.new_task('featurize_train',FeaturizeTask,autopass=False,preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'])
                trainfeaturizer.in_pre_featurized = input_feeds['train']
                trainfeatures = trainfeaturizer.out_featurized

        if set(['test','test_csv','vectors_test']) & set(list(input_feeds.keys())):
            
            if 'vectors_test' in input_feeds.keys():
                testinstances = input_feeds['vectors_test']
            elif 'test_csv' in input_feeds.keys():
                testcsvtransformer = workflow.new_task('test_transformer_csv',TransformCsv,autopass=True,delimiter=self.delimiter)
                testcsvtransformer.in_csv = input_feeds['featurized_test_csv']
                testfeatures = testcsvtransformer.out_features
            else:
                testfeaturizer = workflow.new_task('featurize_test',FeaturizeTask,autopass=False,preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'])
                testfeaturizer.in_pre_featurized = input_feeds['test']
                testfeatures = testfeaturizer.out_featurized

        ######################
        ### Training phase ###
        ######################

        if set(['test','test_csv','vectors_test']) & set(list(input_feeds.keys())):
            ensembler = workflow.new_task('ensemble_traintest',EnsembleTrainTest,autopass=True,
                classify_parameters=task_args['classify'],ga_parameters=task_args['ga'],vectorize_parameters=task_args['vectorize'])
            ensembler.in_test = testfeatures
            ensembler.in_train = trainfeatures
            ensembler.in_trainlabels = trainlabels
        else: # only train
            ensembler = workflow.new_task('ensemble_train',EnsembleTrain,autopass=True,
                classify_parameters=task_args['classify'],ga_parameters=task_args['ga'],vectorize_parameters=task_args['vectorize'])
            ensembler.in_train = trainfeatures
            ensembler.in_trainlabels = trainlabels

        trainer = workflow.new_task('train',Train,autopass=True,classifier=self.classifier,classify_parameters=task_args['classify'],ga_parameters=task_args['ga'])
        trainer.in_train = ensembler.out_train
        trainer.in_trainlabels = ensembler.out_labels            

        ######################
        ### Testing phase ####
        ######################

        if set(['test','test_csv','vectors_test']) & set(list(input_feeds.keys())):

            if not model:
                model = trainer.out_model

            predictor = workflow.new_task('predictor',Predict,autopass=True,
                classifier=self.classifier,ordinal=self.ordinal,linear_raw=self.linear_raw,scale=self.scale,ga=self.ga)
            predictor.in_test = ensembler.out_test
            predictor.in_trainlabels = ensembler.out_labels
            predictor.in_model = model

            if self.linear_raw:
                translator = workflow.new_task('predictor',TranslatePredictions,autopass=True)
                translator.in_linear_labels = trainlabels
                translator.in_predictions = predictor.out_predictions
                return translator

            else:
                return predictor

        else:
            return trainer
