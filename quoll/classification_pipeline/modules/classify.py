
import os
import numpy
from scipy import sparse
import pickle
import math
import random
from collections import defaultdict

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.modules.vectorize import Vectorize, VectorizeCsv, FeaturizeTask, Combine

from quoll.classification_pipeline.functions.classifier import *
from quoll.classification_pipeline.functions import ga, vectorizer

#################################################################
### Tasks #######################################################
#################################################################

class Train(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()

    classifier = Parameter()
    ordinal = BoolParameter()
    jobs = IntParameter()
    iterations = IntParameter()
    scoring = Parameter()
    linear_raw = BoolParameter()

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
    xg_seed = Parameter()
    xg_n_estimators = Parameter() 

    knn_n_neighbors = Parameter()
    knn_weights = Parameter()
    knn_algorithm = Parameter()
    knn_leaf_size = Parameter()
    knn_metric = Parameter()
    knn_p = IntParameter()
   
    def in_featureselection(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.featureselection.txt')   

    def out_model(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.model.pkl')
    
    def out_model_insights(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.model_insights')

    def run(self):

        # initiate directory with model insights
        self.setup_output_dir(self.out_model_insights().path)
        
        # initiate classifier
        classifierdict = {
                        'naive_bayes':[NaiveBayesClassifier(),[self.nb_alpha,self.nb_fit_prior,self.jobs]],
                        'logistic_regression':[LogisticRegressionClassifier(),[self.lr_c,self.lr_solver,self.lr_dual,self.lr_penalty,self.lr_multiclass,self.lr_maxiter]],
                        'svm':[SVMClassifier(),[self.svm_c,self.svm_kernel,self.svm_gamma,self.svm_degree,self.svm_class_weight,self.iterations,self.jobs]], 
                        'svorim':[SvorimClassifier(),[self.svm_c,self.svm_kernel,self.svm_gamma,self.svm_degree]],
                        'xgboost':[XGBoostClassifier(),[self.xg_booster, self.xg_silent, self.jobs, self.xg_learning_rate, self.xg_min_child_weight, self.xg_max_depth, self.xg_gamma, 
                            self.xg_max_delta_step, self.xg_subsample, self.xg_colsample_bytree, self.xg_reg_lambda, self.xg_reg_alpha, self.xg_scale_pos_weight, 
                            self.xg_objective, self.xg_seed, self.xg_n_estimators, self.scoring, self.iterations,self.jobs]],
                        'knn':[KNNClassifier(),[self.knn_n_neighbors, self.knn_weights, self.knn_algorithm, self.knn_leaf_size, self.knn_metric, self.knn_p, self.scoring, self.jobs]], 
                        'tree':[TreeClassifier(),[]], 
                        'perceptron':[PerceptronLClassifier(),[]], 
                        'linreg':[LinearRegressionClassifier(),[self.linreg_fit_intercept, self.linreg_normalize, self.linreg_copy_X]]
                        }
        clf = classifierdict[self.classifier][0]

        # load vectorized instances
        loader = numpy.load(self.in_train().path)
        vectorized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load trainlabels
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')

        # load featureselection
        with open(self.in_featureselection().path,'r',encoding='utf-8') as infile:
            vocab = infile.read().strip().split('\n')
            featureselection = [line.split('\t') for line in vocab]
            featureselection_names = [x[0] for x in featureselection]

        # train classifier
        if self.ordinal or self.linear_raw:
            clflabels = [float(x) for x in trainlabels]
        else:
            clf.set_label_encoder(sorted(list(set(trainlabels))))
            clflabels = clf.label_encoder.transform(trainlabels)
        clf.train_classifier(vectorized_instances, clflabels, *classifierdict[self.classifier][1])

        # save classifier
        model = clf.return_classifier()
        with open(self.out_model().path, 'wb') as fid:
            pickle.dump(model, fid)

        # save model insights
        model_insights = clf.return_model_insights(featureselection_names)
        for mi in model_insights:
            with open(self.out_model_insights().path + '/' + mi[0],'w',encoding='utf-8') as outfile:
                outfile.write(mi[1])

class Predict(Task):

    in_train = InputSlot()
    in_test = InputSlot()
    in_trainlabels = InputSlot()

    classifier = Parameter()
    ordinal = BoolParameter()
    linear_raw = BoolParameter()

    def in_model(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.model.pkl')
        
    def out_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.predictions.txt')

    def out_full_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.full_predictions.txt')

    def run(self):

        # load vectorized instances
        loader = numpy.load(self.in_test().path)
        vectorized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load classifier
        with open(self.in_model().path, 'rb') as fid:
            model = pickle.load(fid)

        # needed to prevent Knn classifier bug
        if self.classifier == 'knn':
            model.n_neighbors = int(model.n_neighbors)

        # inititate classifier
        clf = AbstractSKLearnClassifier()

        # load labels (for the label encoder)
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')
        
        # apply classifier
        if self.ordinal:
            predictions, full_predictions = clf.apply_model(model,vectorized_instances,sorted(list(set([int(x) for x in trainlabels]))))
            predictions = [str(int(x)) for x in predictions]
        elif self.linear_raw:
            predictions, full_predictions = clf.apply_model(model,vectorized_instances,['raw'])
            predictions = [str(x) for x in predictions]
        else:
            clf.set_label_encoder(trainlabels)
            predictions, full_predictions = clf.apply_model(model,vectorized_instances)
            predictions = clf.label_encoder.inverse_transform(predictions)

        # write predictions to file
        with open(self.out_predictions().path,'w',encoding='utf-8') as pr_out:
            pr_out.write('\n'.join(predictions))

        # write full predictions to file
        with open(self.out_full_predictions().path,'w',encoding='utf-8') as fpr_out:
            fpr_out.write('\n'.join(['\t'.join([str(prob) for prob in full_prediction]) for full_prediction in full_predictions]))

class TrainGA(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()

    weight_feature_size = Parameter()
    num_iterations = IntParameter()
    elite = Parameter()
    population_size = IntParameter()
    crossover_probability = Parameter()
    mutation_rate = Parameter()
    tournament_size = IntParameter()
    n_crossovers = IntParameter()
    stop_condition = IntParameter()
    instance_steps = IntParameter()
    sampling = BoolParameter()
    samplesize = Parameter()

    classifier = Parameter()
    ordinal = BoolParameter()
    linear_raw = BoolParameter()
    jobs = IntParameter()
    iterations = IntParameter()
    scoring = Parameter()
    
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
    xg_seed = Parameter()
    xg_n_estimators = Parameter() 

    knn_n_neighbors = Parameter()
    knn_weights = Parameter()
    knn_algorithm = Parameter()
    knn_leaf_size = Parameter()
    knn_metric = Parameter()
    knn_p = IntParameter()
   
    def in_featureselection(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.featureselection.txt')   

    def out_vectors(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.vectors.npz')   

    def out_selected_features(self):
        return self.outputfrominput(inputformat='featureselection', stripextension='.featureselection.txt', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.featureselection.txt')

    def out_model(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.model.pkl')
    
    def out_model_insights(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.model_insights')

    def out_report(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.report.txt')   

    def out_clf_fitness(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.clf_fitness.txt')   

    def out_features_fitness(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.features_fitness.txt')   

    def out_weighted_fitness(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.weighted_fitness.txt')   

    def out_best_features(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.best_features.txt')   

    def out_best_parameters(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.best_parameters.txt')   

    def out_evolution(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.evolution.txt')   

    def out_parameter_evolution(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.parameter_evolution.txt')   

    def run(self):

        # initiate directory with model insights
        self.setup_output_dir(self.out_model_insights().path)
        
        # load trainvectors instances
        loader = numpy.load(self.in_train().path)
        vectorized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load trainlabels
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')
        if self.ordinal or self.linear_raw:
            trainlabels = [float(x) for x in trainlabels]
        trainlabels = numpy.array(trainlabels)

        # load featureselection
        with open(self.in_featureselection().path,'r',encoding='utf-8') as infile:
            vocab = infile.read().strip().split('\n')
            featureselection = [line.split('\t') for line in vocab]
            featureselection_names = [x[0] for x in featureselection]

        # load GA class
        ga_instance = ga.GA(vectorized_instances,trainlabels,featureselection_names)
        best_features, best_parameters, best_features_output, best_parameters_output, clf_output, evolution_output, parameter_evolution_output, features_output, weighted_output, ga_report = ga_instance.run(
            num_iterations=self.num_iterations,population_size=self.population_size,elite=float(self.elite),crossover_probability=self.crossover_probability,mutation_rate=self.mutation_rate,tournament_size=self.tournament_size,n_crossovers=self.n_crossovers,stop_condition=self.stop_condition,
            classifier=self.classifier,jobs=self.jobs,ordinal=self.ordinal,fitness_metric=self.scoring,weight_feature_size=float(self.weight_feature_size),steps=self.instance_steps,linear_raw=self.linear_raw,sampling=self.sampling,samplesize=self.samplesize,
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

        # collect output
        selection = random.choice(range(len(best_features)))
        feature_selection_indices = best_features[selection]
        new_parameters = best_parameters[selection]
        new_trainvectors = vectorized_instances[:,feature_selection_indices]
        new_featureselection = [vocab[i] for i in feature_selection_indices]
        
        # train classifier on all traindata
        classifierdict = {
                        'naive_bayes':[NaiveBayesClassifier(),[self.nb_alpha,self.nb_fit_prior,self.jobs]],
                        'logistic_regression':[LogisticRegressionClassifier(),[self.lr_c,self.lr_solver,self.lr_dual,self.lr_penalty,self.lr_multiclass,self.lr_maxiter]],
                        'svm':[SVMClassifier(),[self.svm_c,self.svm_kernel,self.svm_gamma,self.svm_degree,self.svm_class_weight,self.iterations,self.jobs]], 
                        'xgboost':[XGBoostClassifier(),[self.xg_booster, self.xg_silent, self.jobs, self.xg_learning_rate, self.xg_min_child_weight, self.xg_max_depth, self.xg_gamma, 
                            self.xg_max_delta_step, self.xg_subsample, self.xg_colsample_bytree, self.xg_reg_lambda, self.xg_reg_alpha, self.xg_scale_pos_weight, 
                                                        self.xg_objective, self.xg_seed, self.xg_n_estimators, self.scoring, self.jobs]],
                        'knn':[KNNClassifier(),[self.knn_n_neighbors, self.knn_weights, self.knn_algorithm, self.knn_leaf_size, self.knn_metric, self.knn_p, self.scoring, self.jobs]], 
                        'tree':[TreeClassifier(),[]], 
                        'perceptron':[PerceptronLClassifier(),[]], 
                        'linreg':[LinearRegressionClassifier(),[self.linreg_fit_intercept, self.linreg_normalize, self.linreg_copy_X]]
                        }
        clf = classifierdict[self.classifier][0]
        if self.ordinal or self.linear_raw:
            clflabels = [float(x) for x in trainlabels]
        else:
            clf.set_label_encoder(sorted(list(set(trainlabels))))
            clflabels = clf.label_encoder.transform(trainlabels)
        clf.train_classifier(new_trainvectors, clflabels, *new_parameters)

        # save classifier
        model = clf.return_classifier()
        with open(self.out_model().path, 'wb') as fid:
            pickle.dump(model, fid)

        # save model insights
        model_insights = clf.return_model_insights(featureselection_names)
        for mi in model_insights:
            with open(self.out_model_insights().path + '/' + mi[0],'w',encoding='utf-8') as outfile:
                outfile.write(mi[1])
        
        # save featureselection
        with open(self.out_selected_features().path,'w',encoding='utf-8') as f_out:
            f_out.write('\n'.join(new_featureselection))

        # write vectors
        gavectors = vectorized_instances[:,feature_selection_indices]
        numpy.savez(self.out_vectors().path, data=gavectors.data, indices=gavectors.indices, indptr=gavectors.indptr, shape=gavectors.shape)

        # save reports
        with open(self.out_report().path,'w',encoding='utf-8') as r_out:
            r_out.write(ga_report)

        with open(self.out_clf_fitness().path,'w',encoding='utf-8') as r_out:
            r_out.write(clf_output)

        with open(self.out_features_fitness().path,'w',encoding='utf-8') as r_out:
            r_out.write(features_output)

        with open(self.out_weighted_fitness().path,'w',encoding='utf-8') as r_out:
            r_out.write(weighted_output)

        with open(self.out_best_features().path,'w',encoding='utf-8') as r_out:
            r_out.write(best_features_output)

        with open(self.out_best_parameters().path,'w',encoding='utf-8') as r_out:
            r_out.write(best_parameters_output)

        with open(self.out_evolution().path,'w',encoding='utf-8') as r_out:
            r_out.write(evolution_output)

        with open(self.out_parameter_evolution().path,'w',encoding='utf-8') as r_out:
            r_out.write(parameter_evolution_output)

class TransformVectors(Task):

    in_train = InputSlot()
    in_test = InputSlot()

    vectors = BoolParameter()

    def in_train_featureselection(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.featureselection.txt')

    def in_test_featureselection(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.featureselection.txt' if self.vectors else '.vocabulary.txt')

    def out_vectors(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.'.join(self.in_train().path.split('/')[-1].split('.')[-6:-2]) + '.transformed.vectors.npz')

    def out_test_featureselection(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.'.join(self.in_train().path.split('/')[-1].split('.')[-6:-2]) + '.transformed.featureselection.txt')        

    def run(self):

        # assert that train and featureselection files exists (not checked in component)
        assert os.path.exists(self.in_train_featureselection().path), 'Train featureselection file not found, make sure the file exists and/or change vocabulary path name to ' + self.in_train_featureselection().path 
        assert os.path.exists(self.in_test_featureselection().path), 'Test featureselection file not found, make sure the file exists and/or change vocabulary path name to ' + self.in_test_featureselection().path 

        # load train
        loader = numpy.load(self.in_train().path)
        traininstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load test
        loader = numpy.load(self.in_test().path)
        testinstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load train featureselection
        with open(self.in_train_featureselection().path,'r',encoding='utf-8') as infile:
            trainlines = infile.read().split('\n')
            train_featureselection = [line.split('\t') for line in trainlines]
        train_featureselection_vocabulary = [x[0] for x in train_featureselection]
        print('Loaded',len(train_featureselection_vocabulary),'selected trainfeatures, now loading test featureselection...')

        # load test featureselection
        with open(self.in_test_featureselection().path,'r',encoding='utf-8') as infile:
            testlines = infile.read().split('\n')
            test_featureselection = [line.split('\t') for line in testlines]
        test_featureselection_vocabulary = [x[0] for x in test_featureselection]
        print('Loaded',len(test_featureselection_vocabulary),'selected testfeatures, now aligning testvectors...')

        # transform vectors
        testvectors = vectorizer.align_vectors(testinstances, train_featureselection_vocabulary, test_featureselection_vocabulary)
        
        # write instances to file
        numpy.savez(self.out_vectors().path, data=testvectors.data, indices=testvectors.indices, indptr=testvectors.indptr, shape=testvectors.shape)

        # write featureselection to file
        print('WRITING',len(trainlines),'TRAINLINES')
        with open(self.out_test_featureselection().path, 'w', encoding = 'utf-8') as t_out:
            t_out.write('\n'.join(trainlines))
 
class TranslatePredictions(Task):

    in_linear_labels = InputSlot()
    in_predictions = InputSlot()

    def in_nominal_labels(self):
        return self.outputfrominput(inputformat='linear_labels', stripextension='.raw.labels', addextension='.labels')

    def out_predictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.translated.predictions.txt')

    def out_full_predictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.translated.full_predictions.txt')

    def out_translator(self):
        return self.outputfrominput(inputformat='linear_labels', stripextension='.labels', addextension='.labeltranslator.txt')

    def run(self):

        # open linear labels
        with open(self.in_linear_labels().path,'r',encoding='utf-8') as linear_labels_in:
            linear_labels = [float(x) for x in linear_labels_in.read().strip().split('\n')]
            
        # open nominal labels
        with open(self.in_nominal_labels().path,'r',encoding='utf-8') as nominal_labels_in:
            nominal_labels = nominal_labels_in.read().strip().split('\n')

        # open predictions
        with open(self.in_predictions().path,'r',encoding='utf-8') as predictions_in:
            predictions = [float(x) for x in predictions_in.read().strip().split('\n')]

        # check if both lists have the same length
        if len(linear_labels) != len(nominal_labels):
            print('Linear labels (',len(linear_labels),') and nominal labels (',len(nominal_labels),') do not have the same length; exiting program...')
            quit()

        # generate dictionary (1 0 14.4\n2 14.4 15.6)
        translated_labels = []
        for label in sorted(list(set(nominal_labels))):
            fitting_linear_labels = [x for i,x in enumerate(linear_labels) if nominal_labels[i] == label]
            translated_labels.append([label,min(fitting_linear_labels),max(fitting_linear_labels)])
        translated_labels_sorted = sorted(translated_labels,key = lambda k : k[1])
        translator = []
        for i,tl in enumerate(translated_labels_sorted):
            if i == 0:
                new_min = translated_labels_sorted[i][1] - 500000
            else:
                new_min = translated_labels_sorted[i-1][2]
            if i == len(translated_labels_sorted)-1:
                new_max = translated_labels_sorted[i][2] * 50
            else:
                new_max = translated_labels_sorted[i][2]
            new_tl = [str(translated_labels_sorted[i][0]),new_min,new_max]
            translator.append(new_tl)

        # translate predictions
        translated_predictions = []
        for prediction in predictions:
            for candidate in translator:
                if prediction > candidate[1] and prediction < candidate[2]:
                    translated_predictions.append(candidate[0])
                    break

        # write translated predictions to files
        with open(self.out_predictions().path,'w',encoding='utf-8') as out:
            out.write('\n'.join(translated_predictions))

        with open(self.out_full_predictions().path,'w',encoding='utf-8') as out:
            classes = sorted([x[0] for x in translator])
            full_predictions = [classes]
            for i,x in enumerate(translated_predictions):
                full_predictions.append(['-'] * len(classes)) # dummy output
            out.write('\n'.join(['\t'.join([prob for prob in full_prediction]) for full_prediction in full_predictions]))

        # write translator to file
        with open(self.out_translator().path,'w',encoding='utf-8') as out:
            out.write('\n'.join([' '.join([str(x) for x in line]) for line in translator]))

class FitTransformScale(Task):

    in_vectors = InputSlot()

    min_scale = Parameter()
    max_scale = Parameter()
    
    def in_featureselection(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.featureselection.txt')       
    
    def out_vectors(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.scaled_' + self.min_scale + '_' + self.max_scale + '.vectors.npz')

    def out_scaler(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.scaled_' + self.min_scale + '_' + self.max_scale + '.scaler.pkl')

    def out_featureselection(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.scaled_' + self.min_scale + '_' + self.max_scale + '.featureselection.txt')       
    
    def run(self):

        min_scale = float(self.min_scale)
        max_scale = float(self.max_scale)
        
        # read vectors
        loader = numpy.load(self.in_vectors().path)
        vectors = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # read vocabulary
        with open(self.in_featureselection().path,'r',encoding='utf-8') as file_in:
            featureselection = file_in.read().strip().split('\n')
        
        # scale vectors
        scaler = vectorizer.fit_scale(vectors,min_scale,max_scale)
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
    in_train = InputSlot()

    min_scale = Parameter()
    max_scale = Parameter()

    def in_scaler(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.scaler.pkl')

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

class VectorizeTrain(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()

    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()
    select = BoolParameter()
    selector = Parameter()
    select_threshold = Parameter()
    traincsv = BoolParameter()
    delimiter=Parameter()
    trainvec = BoolParameter()
    
    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.csv' if self.traincsv else '.'.join(self.in_train().path.split('.')[-2:]), addextension='.' + self.selector + '.' + self.select_threshold + '.balanced.vectors.npz' if self.select and self.balance and self.traincsv else '.' + self.selector + '.' + self.select_threshold + '.vectors.npz' if self.select and self.traincsv else '.balanced.vectors.npz' if self.balance and self.traincsv else '.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.select and self.balance else '.' + self.selector + '.' + self.select_threshold + '.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.select else '.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.balance else '.vectors.npz' if self.trainvec else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz')

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.balanced.labels' if self.balance else '.labels')       

    def run(self):
        
        if self.complete(): # necessary as it will not complete otherwise
            return True
        else:
            yield Vectorize(traininstances=self.in_train().path,trainlabels=self.in_trainlabels().path,weight=self.weight,prune=self.prune,balance=self.balance,select=self.select,selector=self.selector,select_threshold=self.select_threshold)

class VectorizeTrainTest(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_test = InputSlot()

    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()
    select = BoolParameter()
    selector = Parameter()
    select_threshold = Parameter()
    traincsv = BoolParameter()
    testcsv = BoolParameter()
    delimiter=Parameter()
    trainvec = BoolParameter()
    testvec = BoolParameter()

    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.csv' if self.traincsv else '.'.join(self.in_train().path.split('.')[-2:]), addextension='.' + self.selector + '.' + self.select_threshold + '.balanced.vectors.npz' if self.select and self.balance and (self.traincsv or self.trainvec)  else '.' + self.selector + '.' + self.select_threshold + '.vectors.npz' if self.select and (self.traincsv or self.trainvec) else '.balanced.vectors.npz' if self.balance and (self.traincsv or self.trainvec) else '.' + self.selector + '.' + self.select_threshold + '.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.select and self.balance else '.' + self.selector + '.' + self.select_threshold + '.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.select else '.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.balance else '.vectors.npz' if self.traincsv or self.trainvec else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz')
    
    def out_trainlabels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.balanced.labels' if self.balance else '.labels')       

    def out_test(self):
        return self.outputfrominput(inputformat='test', stripextension='.csv' if self.testcsv else '.'.join(self.in_test().path.split('.')[-2:]), addextension='.' + self.selector + '.' + self.select_threshold + '.balanced.vectors.npz' if self.select and self.balance and (self.testcsv or self.testvec) else '.' + self.selector + '.' + self.select_threshold + '.vectors.npz' if self.select and (self.testcsv or self.testvec) else '.balanced.vectors.npz' if self.balance and (self.testcsv or self.testvec) else '.' + self.selector + '.' + self.select_threshold + '.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.select and self.balance else '.' + self.selector + '.' + self.select_threshold + '.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.select else '.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.balance else '.vectors.npz' if self.testcsv or self.testvec else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz')

    def run(self):
        
        if self.complete(): # necessary as it will not complete otherwise
            return True
        else:
            yield Vectorize(traininstances=self.in_train().path,trainlabels=self.in_trainlabels().path,testinstances=self.in_test().path,weight=self.weight,prune=self.prune,balance=self.balance,select=self.select,selector=self.selector,select_threshold=self.select_threshold)


class VectorizeTrainCombinedTask(Task):

    in_trainfeatures = InputSlot()
    in_trainvectors_append = InputSlot()
    in_trainlabels = InputSlot()

    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()

    def out_train_combined(self):
        return self.outputfrominput(inputformat='trainfeatures', stripextension='.features.npz', addextension='.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.' + self.in_trainvectors_append().path.split('.')[-3] + '.vectors.npz' if self.balance and self.in_trainvectors_append().path[-3:] == 'npz' else '.balance.weight_' + self.weight + '.prune_' + str(self.prune) + '.' + self.in_trainvectors_append().path.split('.')[-2] + '.vectors.npz' if self.balance and self.in_trainvectors_append().path[-3:] == 'csv' else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.' + self.in_trainvectors_append().path.split('.')[-3] + '.vectors.npz' if self.in_trainvectors_append().path[-3:] == 'npz' else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.' + self.in_trainvectors_append().path.split('.')[-2] + '.vectors.npz')

    def out_train(self):
        return self.outputfrominput(inputformat='trainfeatures', stripextension='.features.npz', addextension='.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.balance else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz')

    def out_trainlabels(self):
           return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.balanced.labels' if self.balance else '.labels')       
    
    def run(self):
        
        if self.complete(): # necessary check as it will not complete otherwise
            return True
        else:
            yield Vectorize(traininstances=self.in_trainfeatures().path,traininstances_append=self.in_trainvectors_append().path,trainlabels=self.in_trainlabels().path,weight=self.weight,prune=self.prune,balance=self.balance)
    
class VectorizeTestCombinedTask(Task):

    in_trainvectors = InputSlot()
    in_testfeatures = InputSlot()
    in_testvectors_append = InputSlot()
    in_trainlabels = InputSlot()

    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()
    
    def out_vectors(self):
        return self.outputfrominput(inputformat='testfeatures', stripextension='.features.npz', addextension='.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.' + self.in_testvectors_append().path.split('.')[-3] + '.vectors.npz' if self.balance and self.in_testvectors_append().path[-3:] == 'npz' else '.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.' + self.in_testvectors_append().path.split('.')[-2] + '.vectors.npz' if self.balance and self.in_testvectors_append().path[-3:] == 'csv' else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.' + self.in_testvectors_append().path.split('.')[-3] + '.vectors.npz' if self.in_testvectors_append().path[-3:] == 'npz' else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.' + self.in_testvectors_append().path.split('.')[-2] + '.vectors.npz')

    def run(self):

        if self.complete(): # necessary as it will not complete otherwise
            return True
        else:
            yield Vectorize(traininstances=self.in_trainvectors().path,trainlabels=self.in_trainlabels().path,testinstances=self.in_testfeatures().path,testinstances_append=self.in_testvectors_append().path,weight=self.weight,prune=self.prune,balance=self.balance)


#################################################################
### Component ###################################################
#################################################################

@registercomponent
class Classify(WorkflowComponent):
    
    traininstances = Parameter()
    trainlabels = Parameter()
    testinstances = Parameter(default = 'xxx.xxx') # not obligatory, dummy extension to enable a pass

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
    instance_steps = IntParameter(default=1)
    sampling = BoolParameter() # repeated resampling to prevent overfitting
    samplesize = Parameter(default='0.8') # size of trainsample
    
    # classifier parameters
    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter()
    jobs = IntParameter(default=1)
    iterations = IntParameter(default=10)
    scoring = Parameter(default='roc_auc') # optimization metric for grid search
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
    xg_seed = Parameter(default='7')
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
                InputFormat(self, format_id='vectorized_train_csv',extension='.csv',inputparameter='traininstances'),
                InputFormat(self, format_id='featurized_train',extension='.features.npz',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txt',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txtdir',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.json',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.jsondir',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.txt',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_featurized_train',extension='.txtdir',inputparameter='traininstances'),
                InputFormat(self, format_id='docs_train',extension='.txt',inputparameter='traininstances')
                ),
                (
                InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels')
                ),
                (
                InputFormat(self, format_id='vectorized_test',extension='.vectors.npz',inputparameter='testinstances'),
                InputFormat(self, format_id='vectorized_test_csv',extension='.csv',inputparameter='testinstances'),
                InputFormat(self, format_id='featurized_test',extension='.features.npz',inputparameter='testinstances'),
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
        ### vectorize ########
        ######################
        
        trainlabels = input_feeds['labels_train']

        if 'vectorized_train' in input_feeds.keys():
            traininstances = input_feeds['vectorized_train']

        elif 'vectorized_train_csv' in input_feeds.keys():
            traininstances = input_feeds['vectorized_train_csv']

        elif 'featurized_train_csv' in input_feeds.keys():
            traininstances = input_feeds['featurized_train']

        elif 'pre_featurized_train' in input_feeds.keys():
            trainfeaturizer = workflow.new_task('featurize_train',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
            trainfeaturizer.in_pre_featurized = input_feeds['pre_featurized_train']

            traininstances = trainfeaturizer.out_featurized

        traincsv=True if ('vectorized_train_csv' in input_feeds.keys()) else False
        trainvec=True if ('vectorized_train' in input_feeds.keys()) else False

        if len(list(set(['vectorized_test','vectorized_test_csv','featurized_test','pre_featurized_test']) & set(list(input_feeds.keys())))) > 0:
 
            if 'vectorized_test' in input_feeds.keys():
                testinstances = input_feeds['vectorized_test']

            elif 'vectorized_test_csv' in input_feeds.keys():
                testinstances = input_feeds['vectorized_test_csv']

            elif 'featurized_test' in input_feeds.keys():
                testinstances = input_feeds['featurized_test']

            else:
                testfeaturizer = workflow.new_task('featurize_test',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                testfeaturizer.in_pre_featurized = input_feeds['pre_featurized_test']

                testinstances = trainfeaturizer.out_featurized

            testcsv=True if ('vectorized_test_csv' in input_feeds.keys()) else False
            testvec=True if ('vectorized_test' in input_feeds.keys()) else False
            if (self.select in testinstances().path.split('.')) or (self.balance and 'balanced' in testinstances().path.split('.')) or (testcsv and self.select == 'False' and self.balance == False):
                trainvectors = traininstances
                testvectors = testinstances
            elif 'classifier_model' in input_feeds.keys(): # not trainfile to base vectorization on
                if not 'vectorized_test' in input_feeds.keys():
                    print('Testinstances can not be vectorized when the classifier model is given as input (traininstances required), exiting program...')
                    quit()
                else:
                    testvectors = testinstances
            else:
                vectorizer = workflow.new_task('vectorize_traintest',VectorizeTrainTest,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance,select=self.select,selector=self.selector,select_threshold=self.select_threshold,delimiter=self.delimiter,traincsv=traincsv,testcsv=testcsv,trainvec=trainvec,testvec=testvec)
                vectorizer.in_train = traininstances
                vectorizer.in_trainlabels = trainlabels
                vectorizer.in_test = testinstances

                trainvectors = vectorizer.out_train
                trainlabels = vectorizer.out_trainlabels
                testvectors = vectorizer.out_test

        else: # only train
            if (self.select in traininstances().path.split('.')) or (self.balance and 'balanced' in traininstances().path.split('.')) or (not self.select and not self.balance):
                trainvectors = traininstances
            else:
                vectorizer = workflow.new_task('vectorize_train',VectorizeTrain,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance,select=self.select,selector=self.selector,select_threshold=self.select_threshold,delimiter=self.delimiter,traincsv=traincsv,trainvec=trainvec)
                vectorizer.in_train = traininstances
                vectorizer.in_trainlabels = trainlabels

                trainvectors = vectorizer.out_train
                trainlabels = vectorizer.out_trainlabels

        ######################
        ### Training phase ###
        ######################

        if self.scale:
            scaler = workflow.new_task('scale_trainvectors',FitTransformScale,autopass=True,min_scale=self.min_scale,max_scale=self.max_scale)
            scaler.in_vectors = trainvectors

            trainvectors = scaler.out_vectors

        if self.ga:
            trainer = workflow.new_task('train_ga',TrainGA,autopass=True,
                instance_steps=self.instance_steps,num_iterations=self.num_iterations, population_size=self.population_size, elite=self.elite, crossover_probability=self.crossover_probability,
                mutation_rate=self.mutation_rate,tournament_size=self.tournament_size,n_crossovers=self.n_crossovers,stop_condition=self.stop_condition,weight_feature_size=self.weight_feature_size,
                classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,linear_raw=self.linear_raw,sampling=self.sampling,samplesize=self.samplesize,
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
            trainer.in_train = trainvectors
            trainer.in_trainlabels = trainlabels
            
        else:
            trainer = workflow.new_task('train',Train,autopass=True,
                classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,linear_raw=self.linear_raw,
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
            trainer.in_train = trainvectors
            trainer.in_trainlabels = trainlabels            

        ######################
        ### Testing phase ####
        ######################

        if len(list(set(['vectorized_test','vectorized_test_csv','featurized_test','pre_featurized_test']) & set(list(input_feeds.keys())))) > 0:

            if self.scale:
                testscaler = workflow.new_task('scale_testvectors',TransformScale,autopass=True,min_scale=self.min_scale,max_scale=self.max_scale)
                testscaler.in_vectors = testvectors
                testscaler.in_train = trainvectors

                testvectors = testscaler.out_vectors

            if self.ga:
                vectors = True if 'vectorized_test' in input_feeds.keys() or 'vectorized_test_csv' in input_feeds.keys() else False      
                transformer = workflow.new_task('tranformer',TransformVectors,autopass=True,vectors=vectors)
                transformer.in_train = trainer.out_vectors
                transformer.in_test = testvectors
                testvectors = transformer.out_vectors
                
            if 'classifier_model' in input_feeds.keys():
                model = input_feeds['classifier_model']
            else:
                model = trainer.out_model

            predictor = workflow.new_task('predictor',Predict,autopass=True,classifier=self.classifier,ordinal=self.ordinal)
            predictor.in_test = testvectors
            predictor.in_trainlabels = trainlabels
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
