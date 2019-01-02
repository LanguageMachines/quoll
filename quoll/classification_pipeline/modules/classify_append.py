
import os
import numpy
from scipy import sparse
import pickle
import math
from collections import defaultdict

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.modules.validate import MakeBins, Folds
from quoll.classification_pipeline.modules.report import ReportFolds
from quoll.classification_pipeline.modules.classify import Train, Predict, VectorizeTrain, VectorizeTrainCombinedTask, VectorizeTrainTest, VectorizeTestCombinedTask
from quoll.classification_pipeline.modules.vectorize import Vectorize, TransformCsv, FeaturizeTask, Combine, VectorizeFoldreporter, VectorizeFoldreporterProbs, VectorizePredictions, VectorizePredictionsProbs

from quoll.classification_pipeline.functions.classifier import *

#################################################################
### Component ###################################################
#################################################################

@registercomponent
class ClassifyAppend(WorkflowComponent):
    
    traininstances = Parameter()
    traininstances_append = Parameter()
    trainlabels = Parameter()
    testinstances = Parameter(default = 'xxx.xxx') # not obligatory, dummy extension to enable a pass
    testinstances_append = Parameter(default = 'xxx.xxx')
    traindocs = Parameter(default = 'xxx.xxx')

    # append parameters
    bow_as_feature = BoolParameter() # to combine bow as separate classification with other features, only relevant in case of train_append
    bow_classifier = Parameter(default='naive_bayes')
    bow_include_labels = Parameter(default='all') # will give prediction probs as feature for each label by default, can specify particular labels (separated by a space) here, only applies when 'bow_prediction_probs' is chosen
    bow_prediction_probs = BoolParameter() # choose to add prediction probabilities

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

    random_type = Parameter(default='equal')
    
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
                InputFormat(self, format_id='vectorized_train',extension='.vectors.npz',inputparameter='traininstances'),
                InputFormat(self, format_id='featurized_train',extension='.features.npz',inputparameter='traininstances'),
                InputFormat(self, format_id='featurized_train_csv',extension='.csv',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txt',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txtdir',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.json',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.jsondir',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.txt',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.txtdir',inputparameter='traininstances'),
                InputFormat(self, format_id='docs_train',extension='.txt',inputparameter='traininstances')
                ),
                (
                InputFormat(self, format_id='vectorized_train_append',extension='.vectors.npz',inputparameter='traininstances_append'),
                InputFormat(self, format_id='featurized_csv_train_append',extension='.csv',inputparameter='traininstances_append'),
                ),
                (
                InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels')
                ),
                (
                InputFormat(self, format_id='vectorized_test',extension='.vectors.npz',inputparameter='testinstances'),
                InputFormat(self, format_id='featurized_test',extension='.features.npz',inputparameter='testinstances'),
                InputFormat(self, format_id='featurized_test_csv',extension='.csv',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_featurized_test',extension='.tok.txt',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_featurized_test',extension='.tok.txtdir',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_featurized_test',extension='.frog.json',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_featurized_test',extension='.frog.jsondir',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_featurized_test',extension='.txt',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_featurized_test',extension='.txtdir',inputparameter='testinstances')
                ),
                (
                InputFormat(self, format_id='vectorized_test_append',extension='.vectors.npz',inputparameter='testinstances_append'),
                InputFormat(self, format_id='featurized_csv_test_append',extension='.csv',inputparameter='testinstances_append'),
                ),
                InputFormat(self, format_id='docs_train',extension='.txt',inputparameter='traindocs')
            ]
            )).T.reshape(-1,6)]

    def setup(self, workflow, input_feeds):

        ######################
        ### Training phase ###
        ######################
        
        trainlabels = input_feeds['labels_train']
        
        if 'vectorized_train' in input_feeds.keys():
            trainvectors = input_feeds['vectorized_train']

        else: # pre_vectorized

            if 'featurized_train_csv' in input_feeds.keys():
                trainvectorizer = workflow.new_task('vectorize_train_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter)
                trainvectorizer.in_csv = input_feeds['featurized_train_csv']
                
                trainvectors = trainvectorizer.out_vectors

            else:

                trainvectors = False
                if 'pre_featurized_train' in input_feeds.keys():
                    trainfeaturizer = workflow.new_task('featurize_train',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                    trainfeaturizer.in_pre_featurized = input_feeds['pre_featurized_train']

                    featurized_train = trainfeaturizer.out_featurized

                else:
                    featurized_train = input_feeds['featurized_train']

        if self.bow_as_feature:

            if trainvectors:
                print('Bag-of-words as features can only be ran on featurized train instances (ending with \'.features.npz\', exiting programme...')
                exit()

            # make bag-of-words predictions on training instances using 5-fold cv
            bin_maker = workflow.new_task('make_bins_bow', MakeBins, autopass=True, n=5)
            bin_maker.in_labels = trainlabels

            fold_runner = workflow.new_task('nfold_cv_bow', Folds, autopass=True, 
                n=5, 
                weight=self.weight, prune=self.prune, balance=self.balance, select=self.select, selector=self.selector, select_threshold=self.select_threshold,
                classifier=self.bow_classifier, ordinal=self.ordinal, jobs=self.jobs, iterations=self.iterations, scoring=self.scoring,
                random_type=self.random_type,
                ga=self.ga,instance_steps=self.instance_steps,num_iterations=self.num_iterations, population_size=self.population_size, elite=self.elite,crossover_probability=self.crossover_probability, sampling=self.sampling, samplesize=self.samplesize,mutation_rate=self.mutation_rate,tournament_size=self.tournament_size,n_crossovers=self.n_crossovers,stop_condition=self.stop_condition,weight_feature_size=self.weight_feature_size,
                nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
                svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
                lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter,
                xg_booster=self.xg_booster, xg_silent=self.xg_silent, xg_learning_rate=self.xg_learning_rate, xg_min_child_weight=self.xg_min_child_weight, 
                xg_max_depth=self.xg_max_depth, xg_gamma=self.xg_gamma, xg_max_delta_step=self.xg_max_delta_step, xg_subsample=self.xg_subsample, 
                xg_colsample_bytree=self.xg_colsample_bytree, xg_reg_lambda=self.xg_reg_lambda, xg_reg_alpha=self.xg_reg_alpha, xg_scale_pos_weight=self.xg_scale_pos_weight,
                xg_objective=self.xg_objective, xg_seed=self.xg_seed, xg_n_estimators=self.xg_n_estimators,
                knn_n_neighbors=self.knn_n_neighbors, knn_weights=self.knn_weights, knn_algorithm=self.knn_algorithm, knn_leaf_size=self.knn_leaf_size,
                knn_metric=self.knn_metric, knn_p=self.knn_p
            )
            fold_runner.in_bins = bin_maker.out_bins
            fold_runner.in_instances = featurized_train
            fold_runner.in_labels = trainlabels
            fold_runner.in_docs = input_feeds['docs_train']

            foldreporter = workflow.new_task('report_folds_bow', ReportFolds, autopass=True)
            foldreporter.in_exp = fold_runner.out_exp

            if self.bow_prediction_probs:
                fold_vectorizer = workflow.new_task('vectorize_foldreporter_probs', VectorizeFoldreporterProbs, autopass=True, include_labels=self.bow_include_labels)
                fold_vectorizer.in_full_predictions = foldreporter.out_full_predictions
                fold_vectorizer.in_bins = bin_maker.out_bins
            else:
                fold_vectorizer = workflow.new_task('vectorize_foldreporter', VectorizeFoldreporter, autopass=True)
                fold_vectorizer.in_predictions = foldreporter.out_predictions
                fold_vectorizer.in_bins = bin_maker.out_bins

            trainvectors = fold_vectorizer.out_vectors

            trainvectorizer_bow = workflow.new_task('vectorize_train_bow',VectorizeTrain,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance,select=self.select,selector=self.selector,select_threshold=self.select_threshold,traincsv=False,delimiter=self.delimiter,trainvec=False)
            trainvectorizer_bow.in_train = featurized_train
            trainvectorizer_bow.in_trainlabels = trainlabels

            trainvectors_bow = trainvectorizer_bow.out_train
            trainlabels = trainvectorizer_bow.out_trainlabels

        if 'vectorized_train_append' in input_feeds.keys():
            trainvectors_append = input_feeds['vectorized_train_append']
        elif 'featurized_csv_train_append' in input_feeds.keys():
            trainvectorizer_append = workflow.new_task('vectorize_train_csv_append',VectorizeCsv,autopass=True,delimiter=self.delimiter)
            trainvectorizer_append.in_csv = input_feeds['featurized_csv_train_append']
                
            trainvectors_append = trainvectorizer_append.out_vectors

        if not trainvectors:
            trainvectorizer = workflow.new_task('vectorize_train',VectorizeTrainTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
            trainvectorizer.in_trainfeatures = featurized_train
            trainvectorizer.in_trainlabels = trainlabels

            trainvectors = trainvectorizer.out_train
            trainlabels = trainvectorizer.out_trainlabels

        if self.scale and not self.bow_as_feature:
            if self.classifier == 'svm':
                trainscaler = workflow.new_task('scale_trainvectors',FitTransformScale,autopass=True,min_scale=-1,max_scale=1)
                trainscaler.in_vectors = trainvectors_append
                trainvectors_append = trainscaler.out_vectors
            elif self.classifier == 'xgboost':
                trainvectors_append = trainvectors_append
            else:
                trainscaler = workflow.new_task('scale_trainvectors',FitTransformScale,autopass=True,min_scale=0,max_scale=1)
                trainscaler.in_vectors = trainvectors_append
                trainvectors_append = trainscaler.out_vectors
                
        traincombiner = workflow.new_task('vectorize_trainvectors_combined',Combine,autopass=True)
        traincombiner.in_vectors = trainvectors
        traincombiner.in_vectors_append = trainvectors_append

        if self.scale and self.bow_as_feature:
            if self.classifier == 'svm':
                trainscaler = workflow.new_task('scale_trainvectors',FitTransformScale,autopass=True,min_scale=-1,max_scale=1)
                trainscaler.in_vectors = traincombiner.out_combined            
                trainvectors_combined = trainscaler.out_vectors
            elif self.classifier == 'xgboost':
                trainvectors_combined = traincombiner.out_combined
            else:
                trainscaler = workflow.new_task('scale_trainvectors',FitTransformScale,autopass=True,min_scale=0,max_scale=1)
                trainscaler.in_vectors = traincombiner.out_combined            
                trainvectors_combined = trainscaler.out_vectors
        else:
            trainvectors_combined = traincombiner.out_combined
                
        trainer = workflow.new_task('train',Train,autopass=True,classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations, scoring=self.scoring, linear_raw=self.linear_raw,
            nb_alpha=self.nb_alpha,nb_fit_prior=self.nb_fit_prior,
            svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
            lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter,
            xg_booster=self.xg_booster, xg_silent=self.xg_silent, xg_learning_rate=self.xg_learning_rate, xg_min_child_weight=self.xg_min_child_weight, 
            xg_max_depth=self.xg_max_depth, xg_gamma=self.xg_gamma, xg_max_delta_step=self.xg_max_delta_step, xg_subsample=self.xg_subsample, 
            xg_colsample_bytree=self.xg_colsample_bytree, xg_reg_lambda=self.xg_reg_lambda, xg_reg_alpha=self.xg_reg_alpha, xg_scale_pos_weight=self.xg_scale_pos_weight,
            xg_objective=self.xg_objective, xg_seed=self.xg_seed, xg_n_estimators=self.xg_n_estimators,
            knn_n_neighbors=self.knn_n_neighbors, knn_weights=self.knn_weights, knn_algorithm=self.knn_algorithm, knn_leaf_size=self.knn_leaf_size,
            knn_metric=self.knn_metric, knn_p=self.knn_p,
            linreg_normalize=self.linreg_normalize, linreg_fit_intercept=self.linreg_fit_intercept, linreg_copy_X=self.linreg_copy_X
        )
        trainer.in_train = trainvectors_combined
        trainer.in_trainlabels = trainlabels            

        ######################
        ### Testing phase ####
        ######################

        if len(list(set(['vectorized_test','featurized_test_csv','featurized_test','pre_featurized_test']) & set(list(input_feeds.keys())))) > 0:
 
            if 'vectorized_test' in input_feeds.keys():
                testvectors = input_feeds['vectorized_test']

            else: # pre_vectorized

                if 'featurized_test_csv' in input_feeds.keys():
                    testvectorizer = workflow.new_task('vectorize_test_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter)
                    testvectorizer.in_csv = input_feeds['featurized_test_csv']

                    testvectors = testvectorizer.out_vectors
                    
                else:

                    testvectors = False
                    if 'pre_featurized_test' in input_feeds.keys():
                        testfeaturizer = workflow.new_task('featurize_test',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                        testfeaturizer.in_pre_featurized = input_feeds['pre_featurized_test']

                        featurized_test = testfeaturizer.out_featurized

                    else:
                        featurized_test = input_feeds['featurized_test']
                
            if self.bow_as_feature:

                if testvectors:
                    print('Bag-of-words as features can only be ran on featurized test instances (ending with \'.features.npz\', exiting programme...')
                    exit()

                bow_trainer = workflow.new_task('train_bow',Train,autopass=True,classifier=self.bow_classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,linear_raw=self.linear_raw,
                    random_type=self.random_type,
                    ga=self.ga,instance_steps=self.instance_steps,num_iterations=self.num_iterations, population_size=self.population_size, elite=self.elite,crossover_probability=self.crossover_probability, sampling=self.sampling, samplesize=self.samplesize,
                    mutation_rate=self.mutation_rate,tournament_size=self.tournament_size,n_crossovers=self.n_crossovers,stop_condition=self.stop_condition,weight_feature_size=self.weight_feature_size,
                    nb_alpha=self.nb_alpha,nb_fit_prior=self.nb_fit_prior,
                    svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
                    lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter,
                    xg_booster=self.xg_booster, xg_silent=self.xg_silent, xg_learning_rate=self.xg_learning_rate, xg_min_child_weight=self.xg_min_child_weight, 
                    xg_max_depth=self.xg_max_depth, xg_gamma=self.xg_gamma, xg_max_delta_step=self.xg_max_delta_step, xg_subsample=self.xg_subsample, 
                    xg_colsample_bytree=self.xg_colsample_bytree, xg_reg_lambda=self.xg_reg_lambda, xg_reg_alpha=self.xg_reg_alpha, xg_scale_pos_weight=self.xg_scale_pos_weight,
                    xg_objective=self.xg_objective, xg_seed=self.xg_seed, xg_n_estimators=self.xg_n_estimators,
                    knn_n_neighbors=self.knn_n_neighbors, knn_weights=self.knn_weights, knn_algorithm=self.knn_algorithm, knn_leaf_size=self.knn_leaf_size,
                    knn_metric=self.knn_metric, knn_p=self.knn_p
                )
                bow_trainer.in_train = trainvectors_bow
                bow_trainer.in_trainlabels = trainlabels            

                testcsv=True if ('vectorized_test_csv' in input_feeds.keys()) else False
                testvec=True if ('vectorized_test' in input_feeds.keys()) else False
                testvectorizer_bow = workflow.new_task('vectorize_test_bow',VectorizeTrainTest,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance,select=self.select,selector=self.selector,select_threshold=self.select_threshold,delimiter=self.delimiter,traincsv=False,testcsv=testcsv,trainvec=False,testvec=testvec)
                testvectorizer_bow.in_train = trainvectors_bow
                testvectorizer_bow.in_trainlabels = trainlabels
                testvectorizer_bow.in_test = featurized_test

                bow_predictor = workflow.new_task('predictor_bow',Predict,autopass=True,classifier=self.bow_classifier,ordinal=self.ordinal)
                bow_predictor.in_test = testvectorizer_bow.out_test
                bow_predictor.in_trainlabels = trainlabels
                bow_predictor.in_model = bow_trainer.out_model

                if self.bow_prediction_probs:
                    prediction_vectorizer = workflow.new_task('vectorize_predictions_probs', VectorizePredictionsProbs, autopass=True, include_labels=self.bow_include_labels)
                    prediction_vectorizer.in_full_predictions = bow_predictor.out_full_predictions
                else:
                    prediction_vectorizer = workflow.new_task('vectorize_predictions', VectorizePredictions, autopass=True)
                    prediction_vectorizer.in_predictions = bow_predictor.out_predictions

                testvectors = prediction_vectorizer.out_vectors

            if 'vectorized_test_append' in input_feeds.keys():
                testvectors_append = input_feeds['vectorized_test_append']
            elif 'featurized_csv_test_append' in input_feeds.keys():
                testvectorizer_append = workflow.new_task('vectorize_test_csv_append',VectorizeCsv,autopass=True,delimiter=self.delimiter)
                testvectorizer_append.in_csv = input_feeds['featurized_csv_test_append']

                testvectors_append = testvectorizer_append.out_vectors

            if not testvectors:
                testvectorizer = workflow.new_task('vectorize_test_bow',VectorizeTestTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
                testvectorizer.in_trainvectors = trainvectors
                testvectorizer.in_trainlabels = trainlabels
                testvectorizer.in_testfeatures = featurized_test

                testvectors = testvectorizer.out_vectors

            if self.scale and not self.bow_as_feature and not self.classifier == 'xgboost':
                testscaler = workflow.new_task('scale_testvectors',TransformScale,autopass=True)
                testscaler.in_vectors = testvectors_append
                testscaler.in_scaler = trainscaler.out_scaler

                testvectors_append = testscaler.out_vectors

            testvector_combiner = workflow.new_task('vectorize_test_combined_vectors',Combine,autopass=True)
            testvector_combiner.in_vectors = testvectors
            testvector_combiner.in_vectors_append = testvectors_append
                            
            if self.scale and self.bow_as_feature and not self.classifier == 'xgboost':
                testscaler = workflow.new_task('scale_testvectors',TransformScale,autopass=True)
                testscaler.in_vectors = testvector_combiner.out_combined
                testscaler.in_scaler = trainscaler.out_scaler

                testvectors_combined = testscaler.out_vectors
            else:
                testvectors_combined = testvector_combiner.out_combined

            predictor = workflow.new_task('predictor',Predict,autopass=True,classifier=self.classifier,ordinal=self.ordinal)
            predictor.in_test = testvectors_combined
            predictor.in_trainlabels = trainlabels
            predictor.in_model = trainer.out_model

            return predictor

        else:

            return trainer
