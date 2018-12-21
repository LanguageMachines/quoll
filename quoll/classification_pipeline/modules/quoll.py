
import numpy
from scipy import sparse
import glob
from collections import defaultdict

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter, FloatParameter

from quoll.classification_pipeline.modules.validate import Validate
from quoll.classification_pipeline.modules.validate_append import ValidateAppend
from quoll.classification_pipeline.modules.report import Report, ReportPerformance, ReportDocpredictions, ReportFolds, ClassifyTask
from quoll.classification_pipeline.modules.classify import Train, Predict, VectorizeTrain, VectorizeTrainTest, VectorizeTrainCombinedTask, VectorizeTestCombinedTask, TransformScale, FitTransformScale, TranslatePredictions 
from quoll.classification_pipeline.modules.vectorize import Vectorize, VectorizeCsv, FeaturizeTask, Combine

from quoll.classification_pipeline.functions import reporter, nfold_cv_functions, linewriter, docreader

#################################################################
### Tasks #######################################################
#################################################################

class ReportTask(Task):

    in_train = InputSlot()
    in_test = InputSlot()
    in_trainlabels = InputSlot()
    in_testlabels = InputSlot()
    in_testdocs = InputSlot()

    testlabels_true = BoolParameter()
    
    # featureselection parameters
    ga = BoolParameter()
    num_iterations = IntParameter()
    population_size = IntParameter()
    elite = Parameter()
    crossover_probability = Parameter()
    mutation_rate = Parameter()
    tournament_size = IntParameter()
    n_crossovers = IntParameter()
    stop_condition = IntParameter()
    weight_feature_size = Parameter()
    instance_steps = IntParameter()
    sampling = BoolParameter()
    samplesize = Parameter()

    # classifier parameters
    classifier = Parameter()
    ordinal = BoolParameter()
    jobs = IntParameter()
    iterations = IntParameter()
    scoring = Parameter()
    linear_raw = BoolParameter()
    scale = BoolParameter()
    min_scale = Parameter()
    max_scale = Parameter()
    
    nb_alpha = Parameter()
    nb_fit_prior = BoolParameter()
    
    svm_c = Parameter()
    svm_kernel = Parameter()
    svm_gamma = Parameter()
    svm_degree = Parameter()
    svm_class_weight = Parameter()

    lr_c = Parameter()
    lr_solver = Parameter()
    lr_dual = BoolParameter()
    lr_penalty = Parameter()
    lr_multiclass = Parameter()
    lr_maxiter = Parameter()

    linreg_fit_intercept = Parameter()
    linreg_normalize = Parameter()
    linreg_copy_X = Parameter()

    xg_booster = Parameter() # choices: ['gbtree', 'gblinear']
    xg_silent = Parameter() # set to '1' to mute printed info on progress
    xg_learning_rate = Parameter() # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_min_child_weight = Parameter() # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_max_depth = Parameter() # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_gamma = Parameter() # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_max_delta_step = Parameter()
    xg_subsample = Parameter() # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_colsample_bytree = Parameter() # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_reg_lambda = Parameter()
    xg_reg_alpha = Parameter() # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_scale_pos_weight = Parameter()
    xg_objective = Parameter() # choices: ['binary:logistic', 'multi:softmax', 'multi:softprob']
    xg_seed = IntParameter()
    xg_n_estimators = Parameter() # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 

    knn_n_neighbors = Parameter()
    knn_weights = Parameter()
    knn_algorithm = Parameter()
    knn_leaf_size = Parameter()
    knn_metric = Parameter()
    knn_p = IntParameter()

    # vectorizer parameters
    weight = Parameter() # options: frequency, binary, tfidf
    prune = IntParameter() # after ranking the topfeatures in the training set, based on frequency or idf weighting
    balance = BoolParameter()
    delimiter = Parameter()
    select = BoolParameter()
    selector = Parameter()
    select_threshold = Parameter()

    def out_report(self):
        return self.outputfrominput(inputformat='test', stripextension='.' + '.'.join(self.in_test().path.split('.')[-2:]), addextension='.scaled.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.transformed.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.translated.report' if self.ga and self.linear_raw and self.scale and self.testlabels_true else '.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.transformed.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.translated.report' if self.ga and self.linear_raw and self.testlabels_true else '.scaled.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.transformed.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.report' if self.scale and self.ga and self.testlabels_true else '.scaled.transformed.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.report' if self.scale and self.linear_raw and self.testlabels_true else '.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.transformed.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.report' if self.ga and self.testlabels_true else '.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.translated.report' if self.linear_raw and self.testlabels_true else '.scaled.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.report' if self.scale and self.testlabels_true else '.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.report' if self.testlabels_true else '.scaled.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.transformed.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.translated.docpredictions.csv' if self.ga and self.linear_raw and self.scale else '.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.transformed.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.translated.docpredictions.csv' if self.ga and self.linear_raw else '.scaled.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.transformed.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.docpredictions.csv' if self.scale and self.ga else '.scaled.transformed.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.docpredictions.csv' if self.scale and self.linear_raw else '.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.transformed.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.docpredictions.csv' if self.ga else '.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.translated.predictions.txt' if self.linear_raw else '.scaled.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.docpredictions.csv' if self.scale else '.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.docpredictions.csv')

    def run(self):

        print(self.out_report().path)
        if self.complete(): # necessary as it will not complete otherwise
            return True

        # if self.in_testlabels().path == 'xxx.xxx':
        #     yield Report(
        #         train=self.in_train().path,test=self.in_test().path,trainlabels=self.in_trainlabels().path,testdocs=self.in_testdocs().path,
        #         weight=self.weight, prune=self.prune, balance=self.balance, select=self.select, select_threshold=self.select_threshold,
        #         ga=self.ga,instance_steps=self.steps,num_iterations=self.num_iterations, population_size=self.population_size, elite=self.elite,crossover_probability=self.crossover_probability, mutation_rate=self.mutation_rate,tournament_size=self.tournament_size,n_crossovers=self.n_crossovers,stop_condition=self.stop_condition,weight_feature_size=self.weight_feature_size,
        #         classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,linear_raw=self.linear_raw,scale=self.scale,min_scale=self.min_scale,max_scale=self.max_scale,
        #         nb_alpha=self.nb_alpha,nb_fit_prior=self.nb_fit_prior,
        #         svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
        #         lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter,
        #         xg_booster=self.xg_booster, xg_silent=self.xg_silent, xg_learning_rate=self.xg_learning_rate, xg_min_child_weight=self.xg_min_child_weight, 
        #         xg_max_depth=self.xg_max_depth, xg_gamma=self.xg_gamma, xg_max_delta_step=self.xg_max_delta_step, xg_subsample=self.xg_subsample, 
        #         xg_colsample_bytree=self.xg_colsample_bytree, xg_reg_lambda=self.xg_reg_lambda, xg_reg_alpha=self.xg_reg_alpha, xg_scale_pos_weight=self.xg_scale_pos_weight,
        #         xg_objective=self.xg_objective, xg_seed=self.xg_seed, xg_n_estimators=self.xg_n_estimators,
        #         knn_n_neighbors=self.knn_n_neighbors, knn_weights=self.knn_weights, knn_algorithm=self.knn_algorithm, knn_leaf_size=self.knn_leaf_size,
        #         knn_metric=self.knn_metric, knn_p=self.knn_p,
        #         linreg_normalize=self.linreg_normalize, linreg_fit_intercept=self.linreg_fit_intercept, linreg_copy_X=self.linreg_copy_X
        #     )
        # else:
        yield Report(
                train=self.in_train().path,test=self.in_test().path,trainlabels=self.in_trainlabels().path,testlabels=self.in_testlabels().path,testdocs=self.in_testdocs().path,
                weight=self.weight, prune=self.prune, balance=self.balance, select=self.select, selector=self.selector, select_threshold=self.select_threshold,
                ga=self.ga,instance_steps=self.instance_steps,num_iterations=self.num_iterations, population_size=self.population_size, elite=self.elite,crossover_probability=self.crossover_probability, mutation_rate=self.mutation_rate,tournament_size=self.tournament_size,n_crossovers=self.n_crossovers,stop_condition=self.stop_condition,weight_feature_size=self.weight_feature_size,sampling=self.sampling,samplesize=self.samplesize,
                classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,linear_raw=self.linear_raw,scale=self.scale,min_scale=self.min_scale,max_scale=self.max_scale,
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

class ValidateAppendTask(Task):

    in_instances = InputSlot()
    in_instances_append = InputSlot()
    in_labels = InputSlot()
    in_docs = InputSlot()

    # fold parameters
    n = IntParameter()
    steps = IntParameter() 
    teststart = IntParameter() 
    testend = IntParameter()

    # append parameters
    bow_as_feature = BoolParameter()
    bow_classifier = Parameter()
    bow_include_labels = Parameter() 
    bow_prediction_probs = BoolParameter() 
    
    # feature selection parameters
    ga = BoolParameter()
    num_iterations = IntParameter()
    population_size = IntParameter()
    elite = Parameter()
    crossover_probability = Parameter()
    mutation_rate = Parameter()
    tournament_size = IntParameter()
    n_crossovers = IntParameter()
    stop_condition = IntParameter()
    weight_feature_size = Parameter()
    instance_steps = IntParameter()
    sampling = BoolParameter()
    samplesize = Parameter()

    # classifier parameters
    classifier = Parameter()
    ordinal = BoolParameter()
    jobs = IntParameter()
    iterations = IntParameter()
    scoring = Parameter()
    linear_raw = BoolParameter()
    scale = BoolParameter()
    min_scale = Parameter()
    max_scale = Parameter()

    nb_alpha = Parameter()
    nb_fit_prior = BoolParameter()
    
    svm_c = Parameter()
    svm_kernel = Parameter()
    svm_gamma = Parameter()
    svm_degree = Parameter()
    svm_class_weight = Parameter()

    lr_c = Parameter()
    lr_solver = Parameter()
    lr_dual = BoolParameter()
    lr_penalty = Parameter()
    lr_multiclass = Parameter()
    lr_maxiter = Parameter()

    linreg_fit_intercept = Parameter()
    linreg_normalize = Parameter()
    linreg_copy_X = Parameter()

    xg_booster = Parameter() 
    xg_silent = Parameter()
    xg_learning_rate = Parameter() 
    xg_min_child_weight = Parameter() 
    xg_max_depth = Parameter() 
    xg_gamma = Parameter() 
    xg_max_delta_step = Parameter()
    xg_subsample = Parameter() 
    xg_colsample_bytree = Parameter() 
    xg_reg_lambda = Parameter()
    xg_reg_alpha = Parameter() 
    xg_scale_pos_weight = Parameter()
    xg_objective = Parameter() 
    xg_seed = IntParameter()
    xg_n_estimators = Parameter()

    knn_n_neighbors = Parameter()
    knn_weights = Parameter()
    knn_algorithm = Parameter()
    knn_leaf_size = Parameter()
    knn_metric = Parameter()
    knn_p = IntParameter()
    
    # vectorizer parameters
    weight = Parameter() # options: frequency, binary, tfidf
    prune = IntParameter() # after ranking the topfeatures in the training set, based on frequency or idf weighting
    balance = BoolParameter()
    select = BoolParameter()
    selector = Parameter()
    select_threshold = Parameter()
   
    def out_exp(self):
        return self.outputfrominput(inputformat='instances', stripextension='.' + '.'.join(self.in_instances().path.split('.')[-2:]), addextension='.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.labels_' + '_'.join(self.in_labels().path.split('/')[-1].split('.')[:-1]) + '.' + self.classifier + '.ga_' + self.ga.__str__() + '.featureweight_' + self.weight_feature_size + '.exp' if self.balance and '.'.join(self.in_instances().path.split('.')[-2:]) == 'features.npz' else '.balanced.labels_' + '_'.join(self.in_labels().path.split('/')[-1].split('.')[:-1]) + '.' + self.classifier + '.ga_' + self.ga.__str__() + '.featureweight_' + self.weight_feature_size + '.exp' if self.balance else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.labels_' + '_'.join(self.in_labels().path.split('/')[-1].split('.')[:-1]) + '.' + self.classifier + '.ga_' + self.ga.__str__() + '.featureweight_' + self.weight_feature_size + '.exp' if '.'.join(self.in_instances().path.split('.')[-2:]) == 'features.npz' else '.labels_' + '_'.join(self.in_labels().path.split('/')[-1].split('.')[:-1]) + '.' + self.classifier + '.ga_' + self.ga.__str__() + '.featureweight_' + self.weight_feature_size + '.exp')
                                    
    def run(self):

        if self.complete(): # necessary as it will not complete otherwise
            return True

        yield ValidateAppend(
            instances=self.in_instances().path,labels=self.in_labels().path,docs=self.in_docs().path,
            n=self.n, steps=self.steps, teststart=self.teststart, testend=self.testend,
            weight=self.weight, prune=self.prune, balance=self.balance, select=self.select, selector=self.selector, select_threshold=self.select_threshold,
            ga=self.ga,instance_steps=self.steps,num_iterations=self.num_iterations, population_size=self.population_size, elite=self.elite,crossover_probability=self.crossover_probability, mutation_rate=self.mutation_rate,tournament_size=self.tournament_size,n_crossovers=self.n_crossovers,stop_condition=self.stop_condition,weight_feature_size=self.weight_feature_size,sampling=self.sampling,samplesize=self.samplesize,
            classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,linear_raw=self.linear_raw,scale=self.scale,min_scale=self.min_scale,max_scale=self.max_scale,
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


class ValidateTask(Task):

    in_instances = InputSlot()
    in_labels = InputSlot()
    in_docs = InputSlot()

    # fold parameters
    n = IntParameter()
    steps = IntParameter() 
    teststart = IntParameter() 
    testend = IntParameter()
    
    # feature selection parameters
    ga = BoolParameter()
    num_iterations = IntParameter()
    population_size = IntParameter()
    elite = Parameter()
    crossover_probability = Parameter()
    mutation_rate = Parameter()
    tournament_size = IntParameter()
    n_crossovers = IntParameter()
    stop_condition = IntParameter()
    weight_feature_size = Parameter()
    instance_steps = IntParameter()
    sampling = BoolParameter()
    samplesize = Parameter()

    # classifier parameters
    classifier = Parameter()
    ordinal = BoolParameter()
    jobs = IntParameter()
    iterations = IntParameter()
    scoring = Parameter()
    linear_raw = BoolParameter()
    scale = BoolParameter()
    min_scale = Parameter()
    max_scale = Parameter()

    nb_alpha = Parameter()
    nb_fit_prior = BoolParameter()
    
    svm_c = Parameter()
    svm_kernel = Parameter()
    svm_gamma = Parameter()
    svm_degree = Parameter()
    svm_class_weight = Parameter()

    lr_c = Parameter()
    lr_solver = Parameter()
    lr_dual = BoolParameter()
    lr_penalty = Parameter()
    lr_multiclass = Parameter()
    lr_maxiter = Parameter()

    linreg_fit_intercept = Parameter()
    linreg_normalize = Parameter()
    linreg_copy_X = Parameter()

    xg_booster = Parameter() 
    xg_silent = Parameter()
    xg_learning_rate = Parameter() 
    xg_min_child_weight = Parameter() 
    xg_max_depth = Parameter() 
    xg_gamma = Parameter() 
    xg_max_delta_step = Parameter()
    xg_subsample = Parameter() 
    xg_colsample_bytree = Parameter() 
    xg_reg_lambda = Parameter()
    xg_reg_alpha = Parameter() 
    xg_scale_pos_weight = Parameter()
    xg_objective = Parameter() 
    xg_seed = IntParameter()
    xg_n_estimators = Parameter()

    knn_n_neighbors = Parameter()
    knn_weights = Parameter()
    knn_algorithm = Parameter()
    knn_leaf_size = Parameter()
    knn_metric = Parameter()
    knn_p = IntParameter()
    
    # vectorizer parameters
    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()
    select = BoolParameter()
    selector = Parameter()
    select_threshold = Parameter()
   
    def out_exp(self):
        return self.outputfrominput(inputformat='instances', stripextension='.' + '.'.join(self.in_instances().path.split('.')[-2:]), addextension='.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.labels_' + '_'.join(self.in_labels().path.split('/')[-1].split('.')[:-1]) + '.' + self.classifier + '.ga_' + self.ga.__str__() + '.featureweight_' + self.weight_feature_size + '.exp' if self.balance and '.'.join(self.in_instances().path.split('.')[-2:]) == 'features.npz' else '.balanced.labels_' + '_'.join(self.in_labels().path.split('/')[-1].split('.')[:-1]) + '.' + self.classifier + '.ga_' + self.ga.__str__() + '.featureweight_' + self.weight_feature_size + '.exp' if self.balance else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.labels_' + '_'.join(self.in_labels().path.split('/')[-1].split('.')[:-1]) + '.' + self.classifier + '.ga_' + self.ga.__str__() + '.featureweight_' + self.weight_feature_size + '.exp' if '.'.join(self.in_instances().path.split('.')[-2:]) == 'features.npz' else '.labels_' + '_'.join(self.in_labels().path.split('/')[-1].split('.')[:-1]) + '.' + self.classifier + '.ga_' + self.ga.__str__() + '.featureweight_' + self.weight_feature_size + '.exp')
                                    
    def run(self):

        if self.complete(): # necessary as it will not complete otherwise
            return True

        yield Validate(
            instances=self.in_instances().path,instances_append=self.in_instances_append().path,labels=self.in_labels().path,docs=self.in_docs().path,
            n=self.n, steps=self.steps, teststart=self.teststart, testend=self.testend,
            bow_as_feature=self.bow_as_feature, bow_classifier=self.bow_classifier, bow_include_labels=self.bow_include_labels, bow_prediction_probs=self.bow_prediction_probs,
            weight=self.weight, prune=self.prune, balance=self.balance, select=self.select, selector=self.selector, select_threshold=self.select_threshold,
            ga=self.ga,instance_steps=self.steps,num_iterations=self.num_iterations, population_size=self.population_size, elite=self.elite,crossover_probability=self.crossover_probability, mutation_rate=self.mutation_rate,tournament_size=self.tournament_size,n_crossovers=self.n_crossovers,stop_condition=self.stop_condition,weight_feature_size=self.weight_feature_size,sampling=self.sampling,samplesize=self.samplesize,
            classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,linear_raw=self.linear_raw,scale=self.scale,min_scale=self.min_scale,max_scale=self.max_scale,
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

################################################################################
### Component ##################################################################
################################################################################

@registercomponent
class Quoll(WorkflowComponent):

    train = Parameter() # only train for nfold cv
    train_append = Parameter(default = 'xxx.xxx') # additional features to combine with word features in .csv or .vector.npz
    test = Parameter(default = 'xxx.xxx')
    test_append = Parameter(default = 'xxx.xxx')
    trainlabels = Parameter()
    trainlabels_layer2 = Parameter(default = 'xxx.xxx')
    testlabels = Parameter(default = 'xxx.xxx')
    testlabels_layer2 = Parameter(default = 'xxx.xxx')
    docs = Parameter(default = 'xxx.xxx') # all docs for nfold cv, test docs for train and test

    # nfold-cv parameters
    n = IntParameter(default=10)
    steps = IntParameter(default=1) # useful to increase if close-by instances, for example sets of 2, are dependent
    teststart = IntParameter(default=0) # if part of the instances are only used for training and not for testing (for example because they are less reliable), specify the test indices via teststart and testend
    testend = IntParameter(default=-1)

    # featureselection parameters
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
    sampling = BoolParameter()
    samplesize = Parameter(default='0.8')

    # classifier parameters
    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter()
    jobs = IntParameter(default=1)
    iterations = IntParameter(default=10)
    scoring = Parameter(default='roc_auc')
    linear_raw = BoolParameter()
    scale = BoolParameter()
    min_scale = Parameter(default='0')
    max_scale = Parameter(default='1')
    
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
    bow_as_feature = BoolParameter() # to combine bow as separate classification with other features, only relevant in case of train_append
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
                InputFormat(self, format_id='vectorized_train',extension='.vectors.npz',inputparameter='train'),
                InputFormat(self, format_id='vectorized_csv_train',extension='.csv',inputparameter='train'),
                InputFormat(self, format_id='featurized_train',extension='.features.npz',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txt',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txtdir',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.json',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.jsondir',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.txtdir',inputparameter='train'),
                InputFormat(self, format_id='docs_train',extension='.txt',inputparameter='train')
                ),
                (
                InputFormat(self, format_id='vectorized_train_append',extension='.vectors.npz',inputparameter='train_append'),
                InputFormat(self, format_id='vectorized_csv_train_append',extension='.csv',inputparameter='train_append'),
                ),
                (
                InputFormat(self, format_id='classified_test',extension='.predictions.txt',inputparameter='test'),
                InputFormat(self, format_id='vectorized_test',extension='.vectors.npz',inputparameter='test'),
                InputFormat(self, format_id='vectorized_csv_test',extension='.csv',inputparameter='test'),
                InputFormat(self, format_id='featurized_test',extension='.features.npz',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.tok.txt',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.tok.txtdir',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.frog.json',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.frog.jsondir',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.txtdir',inputparameter='test'),
                InputFormat(self, format_id='docs_test',extension='.txt',inputparameter='test')
                ),
                (
                InputFormat(self, format_id='vectorized_test_append',extension='.vectors.npz',inputparameter='test_append'),
                InputFormat(self, format_id='vectorized_csv_test_append',extension='.csv',inputparameter='test_append')
                ),
                (
                InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels')
                ),
                (
                InputFormat(self, format_id='labels_train_layer2',extension='.labels',inputparameter='trainlabels_layer2')
                ),
                (
                InputFormat(self, format_id='labels_test',extension='.labels',inputparameter='testlabels')
                ),
                (
                InputFormat(self, format_id='labels_test_layer2',extension='.labels',inputparameter='testlabels_layer2')
                ),
                (
                InputFormat(self, format_id='docs',extension='.txt',inputparameter='docs')
                )
            ]
            )).T.reshape(-1,9)]

    def setup(self, workflow, input_feeds):

        ######################
        ### Training phase ###
        ######################

        trainlabels = input_feeds['labels_train']

        # make sure to have featurized train instances, needed for both the nfold-cv case and the train-test case

        featurized_train = False
        trainvectors = False

        if 'vectorized_train_append' in input_feeds.keys():
            trainvectors_append = input_feeds['vectorized_train_append']
        elif 'vectorized_csv_train_append' in input_feeds.keys():
            trainvectors_append = input_feeds['vectorized_csv_train_append']
        else:
            trainvectors_append = False

        if 'docs_train' in input_feeds.keys() or 'pre_featurized_train' in input_feeds.keys(): # both stored as docs and featurized

            if 'pre_featurized_train' in input_feeds.keys():
                pre_featurized = input_feeds['pre_featurized_train']

            else: # docs (.txt)
                docs = input_feeds['docs_train']
                pre_featurized = input_feeds['docs_train']

            trainfeaturizer = workflow.new_task('featurize_train',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
            trainfeaturizer.in_pre_featurized = pre_featurized

            traininstances = trainfeaturizer.out_featurized

        elif 'featurized_train' in input_feeds.keys(): 
            traininstances = input_feeds['featurized_train']
        elif 'vectorized_csv_train' in input_feeds.keys():
            traininstances = input_feeds['vectorized_csv_train']
        elif 'vectorized_train' in input_feeds.keys():
            traininstances = input_feeds['vectorized_train']

        if not 'test' in [x.split('_')[-1] for x in input_feeds.keys()]: # only train input --> nfold-cv

        ######################
        ### Nfold CV #########
        ######################

            if 'docs' in input_feeds.keys():
                docs = input_feeds['docs']

            if featurized_train:
                instances = featurized_train

            elif trainvectors:
                instances = trainvectors

            else:
                print('Invalid \'train\' input for Nfold CV; ' + 
                    'give either \'.txt\', \'.tok.txt\', \'frog.json\', \'.txtdir\', \'.tok.txtdir\', \'.frog.jsondir\', \'.features.npz\', \'.vectors.npz\' or \'.csv\'')
                exit()

            if trainvectors_append:

                validator = workflow.new_task('validate_append', ValidateAppendTask, autopass=True, 
                    n=self.n, steps=self.steps, teststart=self.teststart, testend=self.testend,
                    bow_as_feature=self.bow_as_feature, bow_classifier=self.bow_classifier, bow_include_labels=self.bow_include_labels, bow_prediction_probs=self.bow_prediction_probs,
                    weight=self.weight, prune=self.prune, balance=self.balance, select=self.select, selector=self.selector, select_threshold=self.select_threshold,
                    ga=self.ga,instance_steps=self.steps,num_iterations=self.num_iterations, population_size=self.population_size, elite=self.elite,crossover_probability=self.crossover_probability,mutation_rate=self.mutation_rate,tournament_size=self.tournament_size,n_crossovers=self.n_crossovers,stop_condition=self.stop_condition,weight_feature_size=self.weight_feature_size,sampling=self.sampling,samplesize=self.samplesize,
                    classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,linear_raw=self.linear_raw,scale=self.scale,min_scale=self.min_scale,max_scale=self.max_scale,
                    nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
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
                validator.in_instances = instances
                validator.in_instances_append = trainvectors_append
                validator.in_labels = trainlabels
                validator.in_docs = docs

            else:

                validator = workflow.new_task('validate', ValidateTask, autopass=True, 
                    n=self.n, steps=self.steps, teststart=self.teststart, testend=self.testend,
                    weight=self.weight, prune=self.prune, balance=self.balance, select=self.select, selector=self.selector, select_threshold=self.select_threshold,
                    ga=self.ga,instance_steps=self.steps,num_iterations=self.num_iterations, population_size=self.population_size, elite=self.elite,crossover_probability=self.crossover_probability,mutation_rate=self.mutation_rate,tournament_size=self.tournament_size,n_crossovers=self.n_crossovers,stop_condition=self.stop_condition,weight_feature_size=self.weight_feature_size,sampling=self.sampling,samplesize=self.samplesize,
                    classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,linear_raw=self.linear_raw,scale=self.scale,min_scale=self.min_scale,max_scale=self.max_scale,
                    nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
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
                validator.in_instances = instances
                validator.in_labels = trainlabels
                validator.in_docs = docs

            # also train a model on all data
            if trainvectors_append:
                if trainvectors:
                    trainvectorizer = workflow.new_task('vectorize_trainvectors_combined',Combine,autopass=True)
                    trainvectorizer.in_vectors = trainvectors
                    trainvectorizer.in_vectors_append = trainvectors_append

                    trainvectors_combined = trainvectorizer.out_combined

                else: # featurized train

                    trainvectorizer = workflow.new_task('vectorize_train_combined',VectorizeTrainCombinedTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
                    trainvectorizer.in_trainfeatures = featurized_train
                    trainvectorizer.in_trainvectors_append = trainvectors_append
                    trainvectorizer.in_trainlabels = trainlabels

                    trainvectors_combined = trainvectorizer.out_train_combined
                    trainvectors = trainvectorizer.out_train
                    trainlabels = trainvectorizer.out_trainlabels

            else:

                trainvectors_combined = False

                if not trainvectors:
                    trainvectorizer = workflow.new_task('vectorize_train',VectorizeTrainTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
                    trainvectorizer.in_trainfeatures = featurized_train
                    trainvectorizer.in_trainlabels = trainlabels

                    trainvectors = trainvectorizer.out_train
                    trainlabels = trainvectorizer.out_trainlabels

            foldtrainer = workflow.new_task('train',Train,autopass=True,classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,
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

            if trainvectors_combined:
                foldtrainer.in_train = trainvectors_combined
            else:
                foldtrainer.in_train = trainvectors
            foldtrainer.in_trainlabels = trainlabels    
            
            labels = foldreporter.out_labels
            predictions = foldreporter.out_predictions

        else:

            # if trainvectors_append:
            #     if trainvectors:
            #         trainvectorizer = workflow.new_task('vectorize_trainvectors_combined',Combine,autopass=True)
            #         trainvectorizer.in_vectors = trainvectors
            #         trainvectorizer.in_vectors_append = trainvectors_append

            #         trainvectors_combined = trainvectorizer.out_combined

            #     else: # featurized train

            #         trainvectorizer = workflow.new_task('vectorize_train_combined',VectorizeTrainCombinedTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
            #         trainvectorizer.in_trainfeatures = featurized_train
            #         trainvectorizer.in_trainvectors_append = trainvectors_append
            #         trainvectorizer.in_trainlabels = trainlabels

            #         trainvectors_combined = trainvectorizer.out_train_combined
            #         trainvectors = trainvectorizer.out_train
            #         trainlabels = trainvectorizer.out_trainlabels

            # else:

            #     trainvectors_combined = False

            #     if not trainvectors:
            #         trainvectorizer = workflow.new_task('vectorize_train',VectorizeTrainTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
            #         trainvectorizer.in_trainfeatures = featurized_train
            #         trainvectorizer.in_trainlabels = trainlabels

            #         trainvectors = trainvectorizer.out_train
            #         trainlabels = trainvectorizer.out_trainlabels

            #     trainer = workflow.new_task('train',Train,autopass=True,classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,
            #         nb_alpha=self.nb_alpha,nb_fit_prior=self.nb_fit_prior,
            #         svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
            #         lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter,
            #         xg_booster=self.xg_booster, xg_silent=self.xg_silent, xg_learning_rate=self.xg_learning_rate, xg_min_child_weight=self.xg_min_child_weight,
            #         xg_max_depth=self.xg_max_depth, xg_gamma=self.xg_gamma, xg_max_delta_step=self.xg_max_delta_step, xg_subsample=self.xg_subsample,
            #         xg_colsample_bytree=self.xg_colsample_bytree, xg_reg_lambda=self.xg_reg_lambda, xg_reg_alpha=self.xg_reg_alpha, xg_scale_pos_weight=self.xg_scale_pos_weight,
            #         xg_objective=self.xg_objective, xg_seed=self.xg_seed, xg_n_estimators=self.xg_n_estimators,
            #         knn_n_neighbors=self.knn_n_neighbors, knn_weights=self.knn_weights, knn_algorithm=self.knn_algorithm, knn_leaf_size=self.knn_leaf_size,
            #         knn_metric=self.knn_metric, knn_p=self.knn_p
            #     )
            #     if trainvectors_combined:
            #         trainer.in_train = trainvectors_combined
            #     else:
            #         trainer.in_train = trainvectors
            #     trainer.in_trainlabels = trainlabels    

            #     trainmodel = trainer.out_model 

            # if 'classified_test' in input_feeds.keys(): # reporter can be started
            #     predictions = input_feeds['classified_test']

            # else: 

        ###############################
        ### Train, test and report ####
        ###############################

                # if 'vectorized_test_append' in input_feeds.keys():
                #     testvectors_append = input_feeds['vectorized_test_append']
                # elif 'vectorized_csv_test_append' in input_feeds.keys():
                #     testvectors_append = input_feeds['vectorized_csv_test_append']
                # else:
                #     testvectors_append = False
                
            if 'vectorized_test' in input_feeds.keys():
                testinstances = input_feeds['vectorized_test']
            else:
                if 'vectorized_csv_test' in input_feeds.keys():
                    testinstances = input_feeds['vectorized_csv_test']
                else:
                    testinstances = False
                
                if 'docs_test' in input_feeds.keys() or 'pre_featurized_test' in input_feeds.keys():

                    if 'pre_featurized_test' in input_feeds.keys():
                        pre_featurized = input_feeds['pre_featurized_test']

                    else:
                        docs = input_feeds['docs_test']
                        pre_featurized = input_feeds['docs_test']

                    testfeaturizer = workflow.new_task('featurize_test',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                    testfeaturizer.in_pre_featurized = pre_featurized

                    testinstances = testfeaturizer.out_featurized

                else:
                    if not testinstances:
                        testinstances = input_feeds['featurized_test']

                traincsv=True if ('vectorized_csv_train' in input_feeds.keys()) else False
                trainvec=True if ('vectorized_train' in input_feeds.keys()) else False
                testcsv=True if ('vectorized_csv_test' in input_feeds.keys()) else False
                testvec=True if ('vectorized_test' in input_feeds.keys()) else False
                    
                vectorizer = workflow.new_task('vectorize_traintest',VectorizeTrainTest,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance,select=self.select,select_threshold=self.select_threshold,delimiter=self.delimiter,traincsv=traincsv,trainvec=trainvec,testcsv=testcsv,testvec=testvec)
                vectorizer.in_train = traininstances
                vectorizer.in_trainlabels = trainlabels
                vectorizer.in_test = testinstances

                traininstances = vectorizer.out_train
                trainlabels = vectorizer.out_trainlabels
                testinstances = vectorizer.out_test
                    
            if 'labels_test' in input_feeds.keys():
                testlabels = input_feeds['labels_test']
                testlabels_true = True
            else:
                testlabels = testinstances
                testlabels_true = False
                
            if 'docs' in input_feeds.keys():
                docs = input_feeds['docs']
            else:
                if not 'docs_test' in input_feeds.keys():
                    print('No docs in input parameters, exiting programme...')
                    quit()

            reporter = workflow.new_task('report', ReportTask, autopass=True, 
                testlabels_true=testlabels_true,
                weight=self.weight, prune=self.prune, balance=self.balance, select=self.select, selector=self.selector, select_threshold=self.select_threshold,
                ga=self.ga,instance_steps=self.steps,num_iterations=self.num_iterations, population_size=self.population_size, elite=self.elite,crossover_probability=self.crossover_probability,mutation_rate=self.mutation_rate,tournament_size=self.tournament_size,n_crossovers=self.n_crossovers,stop_condition=self.stop_condition,weight_feature_size=self.weight_feature_size,sampling=self.sampling,samplesize=self.samplesize,
                classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,linear_raw=self.linear_raw,scale=self.scale,min_scale=self.min_scale,max_scale=self.max_scale,
                nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
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
            reporter.in_train = traininstances
            reporter.in_test = testinstances
            reporter.in_trainlabels = trainlabels
            reporter.in_testlabels = testlabels
            reporter.in_testdocs = docs

            #     if testvectors_append:
            #         if testvectors:
            #             testvectorizer = workflow.new_task('vectorize_test_combined_vectors',Combine,autopass=True)
            #             testvectorizer.in_vectors = testvectors
            #             testvectorizer.in_vectors_append = testvectors_append

            #             testvectors = testvectorizer.out_combined
                        
            #         else:
            #             testvectorizer = workflow.new_task('vectorize_test_combined',VectorizeTestCombinedTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
            #             testvectorizer.in_trainvectors = trainvectors
            #             testvectorizer.in_trainlabels = trainlabels
            #             testvectorizer.in_testfeatures = featurized_test
            #             testvectorizer.in_testvectors_append = testvectors_append

            #             testvectors = testvectorizer.out_vectors
                        
            #     else:
            #         if not testvectors:
            #             testvectorizer = workflow.new_task('vectorize_test',VectorizeTestTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
            #             testvectorizer.in_trainvectors = trainvectors
            #             testvectorizer.in_trainlabels = trainlabels
            #             testvectorizer.in_testfeatures = featurized_test

            #             testvectors = testvectorizer.out_vectors

            #     predictor = workflow.new_task('predictor',Predict,autopass=True,classifier=self.classifier,ordinal=self.ordinal)
            #     predictor.in_test = testvectors
            #     predictor.in_trainlabels = trainlabels
            #     predictor.in_train = trainvectors

            #     predictions = predictor.out_predictions

            # if 'docs' in input_feeds.keys():
            #     docs = input_feeds['docs']
                
            # if 'labels_test' in input_feeds.keys(): # full performance reporter
            #     labels = input_feeds['labels_test']

            # else:
            #     labels = False

        ######################
        ### Reporting phase ##
        ######################

        # if labels:

        #     reporter = workflow.new_task('report_performance',ReportPerformance,autopass=True,ordinal=self.ordinal,teststart=self.teststart)
        #     reporter.in_predictions = predictions
        #     reporter.in_testlabels = labels
        #     reporter.in_testdocuments = docs

        # else: # report docpredictions

        #     reporter = workflow.new_task('report_docpredictions',ReportDocpredictions,autopass=True)
        #     reporter.in_predictions = predictions
        #     reporter.in_testdocuments = docs

        if not 'test' in [x.split('_')[-1] for x in input_feeds.keys()]:
            return reporter, foldtrainer
        else:
            return reporter
