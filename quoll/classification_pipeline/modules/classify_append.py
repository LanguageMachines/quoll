
import os
import numpy
from scipy import sparse
import pickle
import math
from collections import defaultdict

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.modules.validate import ValidateTask
from quoll.classification_pipeline.modules.report import ReportFolds
from quoll.classification_pipeline.modules.classify import Train, Predict, VectorizeTrain, VectorizeTrainCombinedTask, VectorizeTrainTest, VectorizeTestCombinedTask
from quoll.classification_pipeline.modules.vectorize import Vectorize, TransformCsv, FeaturizeTask, Combine, VectorizeFoldreporter, VectorizeFoldreporterProbs, VectorizePredictions, VectorizePredictionsProbs

from quoll.classification_pipeline.functions.classifier import *
from quoll.classification_pipeline.functions import quoll_helpers

#################################################################
### Component ###################################################
#################################################################

@registercomponent
class ClassifyAppend(WorkflowComponent):
    
    train = Parameter()
    train_append = Parameter()
    trainlabels = Parameter()
    test = Parameter(default = 'xxx.xxx') # not obligatory, dummy extension to enable a pass
    test_append = Parameter(default = 'xxx.xxx')
    traindocs = Parameter(default = 'xxx.xxx')

    # append parameters
    bow_as_feature = BoolParameter() # to combine bow as separate classification with other features, only relevant in case of train_append
    bow_classifier = Parameter(default='naive_bayes')
    bow_include_labels = Parameter(default='all') # will give prediction probs as feature for each label by default, can specify particular labels (separated by a space) here, only applies when 'bow_prediction_probs' is chosen
    bow_prediction_probs = BoolParameter() # choose to add prediction probabilities

    # fold-parameters
    n = IntParameter(default=10)
    steps = IntParameter(default=1) # useful to increase if close-by instances, for example sets of 2, are dependent
    teststart = IntParameter(default=0) # if part of the instances are only used for training and not for testing (for example because they are less reliable), specify the test indices via teststart and testend
    testend = IntParameter(default=-1)

    # feature selection parameters
    ga = BoolParameter()
    num_iterations = IntParameter(default=300)
    elite = Parameter(default='0.1')
    population_size = IntParameter(default=100)
    crossover_probability = Parameter(default='0.9')
    mutation_rate = Parameter(default='0.3')
    tournament_size = IntParameter(default=2)
    n_crossovers = IntParameter(default=1)
    stop_condition = IntParameter(default=5)
    weight_feature_size = Parameter(default='0.0')
    instance_steps = IntParameter(default=1)
    sampling = BoolParameter()
    samplesize = Parameter(default='0.8')
    
    # classifier parameters
    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter()
    jobs = IntParameter(default=1)
    iterations = IntParameter(default=10)
    scoring = Parameter(default='roc_auc')
    linear_raw = BoolParameter()
    min_scale = Parameter(default='0')
    max_scale = Parameter(default='1')
    scale = BoolParameter()

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
    xg_silent = Parameter(default='0') # set to '1' to mute printed info on progress
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
    xg_seed = IntParameter(default=7)
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
    balance = BoolParameter()
    delimiter = Parameter(default=',')
    select = BoolParameter()
    selector = Parameter(default=False)
    select_threshold = Parameter(default=False)

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
                InputFormat(self, format_id='vectors_train',extension='.vectors.npz',inputparameter='train'),
                InputFormat(self, format_id='pre_vectors_train',extension='.features.npz',inputparameter='train'),
                InputFormat(self, format_id='pre_vectors_train',extension='.csv',inputparameter='train'),
                InputFormat(self, format_id='pre_vectors_train',extension='.tok.txt',inputparameter='train'),
                InputFormat(self, format_id='pre_vectors_train',extension='.tok.txtdir',inputparameter='train'),
                InputFormat(self, format_id='pre_vectors_train',extension='.frog.json',inputparameter='train'),
                InputFormat(self, format_id='pre_vectors_train',extension='.frog.jsondir',inputparameter='train'),
                InputFormat(self, format_id='pre_vectors_train',extension='.txtdir',inputparameter='train'),
                InputFormat(self, format_id='pre_vectors_train_txt',extension='.txt',inputparameter='train')
                ),
                (
                InputFormat(self, format_id='vectors_train_append',extension='.vectors.npz',inputparameter='train_append'),
                InputFormat(self, format_id='vectors_train_append',extension='.csv',inputparameter='train_append'),
                ),
                (
                InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels')
                ),
                (
                InputFormat(self, format_id='vectors_test',extension='.vectors.npz',inputparameter='test'),
                InputFormat(self, format_id='vectors_test',extension='.features.npz',inputparameter='test'),
                InputFormat(self, format_id='vectors_test',extension='.csv',inputparameter='test'),
                InputFormat(self, format_id='vectors_test',extension='.tok.txt',inputparameter='test'),
                InputFormat(self, format_id='vectors_test',extension='.tok.txtdir',inputparameter='test'),
                InputFormat(self, format_id='vectors_test',extension='.frog.json',inputparameter='test'),
                InputFormat(self, format_id='vectors_test',extension='.frog.jsondir',inputparameter='test'),
                InputFormat(self, format_id='vectors_test',extension='.txt',inputparameter='test'),
                InputFormat(self, format_id='vectors_test',extension='.txtdir',inputparameter='test')
                ),
                (
                InputFormat(self, format_id='vectors_test_append',extension='.vectors.npz',inputparameter='test_append'),
                InputFormat(self, format_id='vectors_test_append',extension='.csv',inputparameter='test_append'),
                ),
                InputFormat(self, format_id='docs_train',extension='.txt',inputparameter='traindocs')
            ]
            )).T.reshape(-1,6)]

    def setup(self, workflow, input_feeds):

        task_args = quoll_helpers.prepare_task_input(['preprocess','featurize','vectorize','classify','ga','append','validate'],workflow.param_kwargs)

        #############################
        ### Prepare vectors phase ###
        #############################
        
        # vectorize train - test
        trainlabels = input_feeds['labels_train']

        vectors = False
        docs = False
        if 'vectors_train' in input_feeds.keys():
            traininstances = input_feeds['vectors_train']
            vectors = True
        else:
            if 'pre_vectors_train_txt' in input_feeds.keys():
                docs_train = input_feeds['pre_vectors_train_txt']
                traininstances = input_feeds['pre_vectors_train_txt']
                docs = True
            else:
                traininstances = input_feeds['pre_vectors_train']

        if 'vectors_test' in input_feeds.keys():

            testinstances = input_feeds['vectors_test']
                
            vectorizer = workflow.new_task('vectorize_traintest',VectorizeTrainTest,autopass=True,
                preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize']
            )
            vectorizer.in_train = traininstances
            vectorizer.in_trainlabels = trainlabels
            vectorizer.in_test = testinstances

            testvectors = vectorizer.out_test                
            test = True

        else: # only train

            # vectorize traininstances
            vectorizer = workflow.new_task('vectorize_train',VectorizeTrain,autopass=True,
                preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize']
            )
            vectorizer.in_train = traininstances
            vectorizer.in_trainlabels = trainlabels

            test = False

        trainvectors = vectorizer.out_train
        trainlabels = vectorizer.out_trainlabels        
        
        # vectorize train_append - test_append 
        traininstances_append = input_feeds['vectors_train_append']

        if 'vectors_test_append' in input_feeds.keys():

            testinstances_append = input_feeds['vectors_test_append']
                
            vectorizer_append = workflow.new_task('vectorize_traintest_append',VectorizeTrainTest,autopass=True,
                preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize']
            )
            vectorizer_append.in_train = traininstances_append
            vectorizer_append.in_trainlabels = input_feeds['labels_train']
            vectorizer_append.in_test = testinstances_append

            testvectors_append = vectorizer_append.out_test                

        else: # only train

            # vectorize traininstances
            vectorizer_append = workflow.new_task('vectorize_train_append',VectorizeTrain,autopass=True,
                preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize']
            )
            vectorizer_append.in_train = traininstances_append
            vectorizer_append.in_trainlabels = input_feeds['labels_train']

        trainvectors_append = vectorizer_append.out_train 

        # prepare bag-of-words feature
        if self.bow_as_feature:

            if not docs:
                if 'docs_train' in input_feeds.keys():
                    docs_train = input_feeds['docs_train']
                else:
                    print('No traindocs (extension \'.txt\') inputted, which is needed for bow-as-feature setting, exiting program...')
                    exit()
            
            if vectors:
                print('Bag-of-words as features can only be ran on featurized train instances (ending with \'.features.npz\', exiting programme...')
                exit()

            bow_validator = workflow.new_task('nfold_cv_bow', ValidateTask, autopass=False,
                n=self.n,classifier=self.bow_classifier,preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],classify_parameters=task_args['classify'],ga_parameters=task_args['ga'],validate_parameters=task_args['validate']
            )
            bow_validator.in_instances = traininstances
            bow_validator.in_labels = input_feeds['labels_train']
            bow_validator.in_docs = docs_train

            # prepare bow test vectors
            bow_trainer = workflow.new_task('train_bow',Train,autopass=True,classifier=self.bow_classifier,classify_parameters=task_args['classify'],ga_parameters=task_args['ga'])
            bow_trainer.in_train = trainvectors
            bow_trainer.in_trainlabels = trainlabels

            if test:

                bow_predictor = workflow.new_task('predictor_bow',Predict,autopass=True,
                    classifier=self.bow_classifier,ordinal=self.ordinal,linear_raw=self.linear_raw,scale=self.scale,ga=self.ga)
                bow_predictor.in_test = testvectors
                bow_predictor.in_trainlabels = trainlabels
                bow_predictor.in_model = bow_trainer.out_model

                if self.bow_prediction_probs:
                    prediction_vectorizer = workflow.new_task('vectorize_predictions_probs', VectorizePredictionsProbs, autopass=True, include_labels=self.bow_include_labels)
                    prediction_vectorizer.in_full_predictions = bow_predictor.out_full_predictions
                else:
                    prediction_vectorizer = workflow.new_task('vectorize_predictions', VectorizePredictions, autopass=True)
                    prediction_vectorizer.in_predictions = bow_predictor.out_predictions

                testvectors = prediction_vectorizer.out_vectors

            # prepare bow train vectors
            if self.bow_prediction_probs:
                fold_vectorizer = workflow.new_task('vectorize_foldreporter_probs', VectorizeFoldreporterProbs, autopass=True, include_labels=self.bow_include_labels)
                fold_vectorizer.in_full_predictions = bow_validator.out_full_predictions
                fold_vectorizer.in_bins = bow_validator.out_bins
            else:
                fold_vectorizer = workflow.new_task('vectorize_foldreporter', VectorizeFoldreporter, autopass=True)
                fold_vectorizer.in_predictions = bow_validator.out_predictions
                fold_vectorizer.in_bins = bow_validator.out_bins

            trainvectors = fold_vectorizer.out_vectors

        # combine vectors
        traincombiner = workflow.new_task('vectorize_trainvectors_combined',Combine,autopass=True)
        traincombiner.in_vectors = trainvectors
        traincombiner.in_vectors_append = trainvectors_append

        if test:
            testvector_combiner = workflow.new_task('vectorize_test_combined_vectors',Combine,autopass=True)
            testvector_combiner.in_vectors = testvectors
            testvector_combiner.in_vectors_append = testvectors_append

        #######################
        ### Training phase ####
        #######################

        trainer = workflow.new_task('train',Train,autopass=True,classifier=self.classifier,classify_parameters=task_args['classify'],ga_parameters=task_args['ga'])
        trainer.in_train = traincombiner.out_combined
        trainer.in_trainlabels = trainlabels          

        ######################
        ### Testing phase ####
        ######################

        if test:

            predictor = workflow.new_task('predictor',Predict,autopass=True,
                classifier=self.classifier,ordinal=self.ordinal,linear_raw=self.linear_raw,scale=self.scale,ga=self.ga)
            predictor.in_test = testvector_combiner.out_combined
            predictor.in_trainlabels = trainlabels
            predictor.in_model = trainer.out_model

            if self.linear_raw:
                translator = workflow.new_task('translator',TranslatePredictions,autopass=True)
                translator.in_linear_labels = trainlabels
                translator.in_predictions = predictor.out_predictions
                return translator

            else:
                return predictor

        else:
            return trainer            
