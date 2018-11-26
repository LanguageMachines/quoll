
import os
import numpy
from scipy import sparse
import pickle
import math
import random
from collections import defaultdict

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.modules.vectorize import Vectorize, VectorizeCsv, FeaturizeTask, FitTransformScale, TransformScale, Combine, Balance

from quoll.classification_pipeline.functions.classifier import *
from quoll.classification_pipeline.functions import wrapper, vectorizer

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

    def out_label_encoding(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.le')
    
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
                        'xgboost':[XGBoostClassifier(),[self.xg_booster, self.xg_silent, self.jobs, self.xg_learning_rate, self.xg_min_child_weight, self.xg_max_depth, self.xg_gamma, 
                            self.xg_max_delta_step, self.xg_subsample, self.xg_colsample_bytree, self.xg_reg_lambda, self.xg_reg_alpha, self.xg_scale_pos_weight, 
                                                        self.xg_objective, self.xg_seed, self.xg_n_estimators, self.scoring, self.jobs]],
                        'knn':[KNNClassifier(),[self.knn_n_neighbors, self.knn_weights, self.knn_algorithm, self.knn_leaf_size, self.knn_metric, self.knn_p, self.scoring, self.jobs]], 
                        'tree':[TreeClassifier(),[]], 
                        'perceptron':[PerceptronLClassifier(),[]], 
                        'linear_regression':[LinearRegressionClassifier(),[]]
                        }
        clf = classifierdict[self.classifier][0]

        # load vectorized instances
        loader = numpy.load(self.in_train().path)
        vectorized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load trainlabels
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')
        if self.ordinal:
            trainlabels = [float(x) for x in trainlabels]

        # load featureselection
        with open(self.in_featureselection().path,'r',encoding='utf-8') as infile:
            vocab = infile.read().strip().split('\n')
            featureselection = [line.split('\t') for line in vocab]
            featureselection_names = [x[0] for x in featureselection]

        # transform trainlabels
        clf.set_label_encoder(sorted(list(set(trainlabels))))

        # train classifier
        clf.train_classifier(vectorized_instances, trainlabels, *classifierdict[self.classifier][1])

        # save classifier
        model = clf.return_classifier()
        with open(self.out_model().path, 'wb') as fid:
            pickle.dump(model, fid)

        # save model insights
        model_insights = clf.return_model_insights(featureselection_names)
        for mi in model_insights:
            with open(self.out_model_insights().path + '/' + mi[0],'w',encoding='utf-8') as outfile:
                outfile.write(mi[1])

        # save label encoding
        label_encoding = clf.return_label_encoding(sorted(list(set(trainlabels))))
        with open(self.out_label_encoding().path,'w',encoding='utf-8') as le_out:
            le_out.write('\n'.join([' '.join(le) for le in label_encoding]))

class Predict(Task):

    in_train = InputSlot()
    in_test = InputSlot()
    in_trainlabels = InputSlot()

    classifier = Parameter()
    ordinal = BoolParameter()

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

        # load labels (for the label encoder)
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')
        if self.ordinal:
            trainlabels = [float(x) for x in labels]

        # inititate classifier
        clf = AbstractSKLearnClassifier()

        # transform labels
        clf.set_label_encoder(trainlabels)

        # apply classifier
        predictions, full_predictions = clf.apply_model(model,vectorized_instances)

        # write predictions to file
        with open(self.out_predictions().path,'w',encoding='utf-8') as pr_out:
            pr_out.write('\n'.join(predictions))

        # write full predictions to file
        with open(self.out_full_predictions().path,'w',encoding='utf-8') as fpr_out:
            fpr_out.write('\n'.join(['\t'.join([str(prob) for prob in full_prediction]) for full_prediction in full_predictions]))

class RunGA(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()

    num_folds = IntParameter()
    fold_steps = IntParameter()
    num_iterations = IntParameter()
    population_size = IntParameter()
    crossover_probability = Parameter()
    mutation_rate = Parameter()
    tournament_size = IntParameter()
    n_crossovers = IntParameter()
    stop_condition = IntParameter()

    classifier = Parameter()
    ordinal = BoolParameter()
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
        return self.outputfrominput(inputformat='train', stripextension='.'.join(self.in_train().path.split('.')[-2:]), addextension='.featureselection.txt')   

    def out_vectors(self):
        return self.outputfrominput(inputformat='train', stripextension='.'.join(self.in_train().path.split('.')[-2:]), addextension='.ga.vectors.npz')   

    def out_selected_features(self):
        return self.outputfrominput(inputformat='featureselection', stripextension='.featureselection.txt', addextension='.ga.featureselection.txt')

    def out_report(self):
        return self.outputfrominput(inputformat='train', stripextension='.'.join(self.in_train().path.split('.')[-2:]), addextension='.ga.report.txt')   

    def run(self):

        # load trainvectors instances
        loader = numpy.load(self.in_train().path)
        vectorized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load trainlabels
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')
        if self.ordinal:
            trainlabels = [float(x) for x in trainlabels]
        trainlabels = numpy.array(trainlabels)

        # load featureselection
        with open(self.in_featureselection().path,'r',encoding='utf-8') as infile:
            vocab = infile.read().strip().split('\n')
            featureselection = [line.split('\t') for line in vocab]
            featureselection_names = [x[0] for x in featureselection]

        # load GA class
        ga = wrapper.GA(vectorized_instances,trainlabels,featureselection_names)
        ga.make_folds(self.num_folds,self.fold_steps)
        ga.run(
            num_iterations=self.num_iterations,population_size=self.population_size,crossover_probability=self.crossover_probability,mutation_rate=self.mutation_rate,tournament_size=self.tournament_size,n_crossovers=self.n_crossovers,stop_condition=self.stop_condition,
            classifier=self.classifier,jobs=self.jobs,ordinal=self.ordinal,fitness_metric=self.scoring,
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

        # collect output
        feature_selections, ga_report = ga.return_overall_report()
        feature_selection_indices = random.choice(feature_selections)

        # save featureselection
        new_featureselection = [vocab[i] for i in feature_selection_indices]
        with open(self.out_selected_features().path,'w',encoding='utf-8') as f_out:
            f_out.write('\n'.join(new_featureselection))

        # write vectors
        gavectors = vectorized_instances[:,feature_selection_indices]
        numpy.savez(self.out_vectors().path, data=gavectors.data, indices=gavectors.indices, indptr=gavectors.indptr, shape=gavectors.shape)

        # save report
        with open(self.out_report().path,'w',encoding='utf-8') as r_out:
            r_out.write(ga_report)


class TransformVectors(Task):

    in_train = InputSlot()
    in_test = InputSlot()

    def in_train_featureselection(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.featureselection.txt')

    def in_test_featureselection(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.featureselection.txt')

    def out_vectors(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.transformed.vectors.npz')

    def out_test_featureselection(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.transformed.featureselection.txt')        

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
            featureselection = [line.split('\t') for line in trainlines]
        train_featureselection_vocabulary = [x[0] for x in featureselection]
        print('Loaded',len(train_featureselection_vocabulary),'selected trainfeatures, now loading test featureselection...')

        # load test featureselection
        with open(self.in_test_featureselection().path,'r',encoding='utf-8') as infile:
            testlines = infile.read().split('\n')
            featureselection = [line.split('\t') for line in testlines]
        test_featureselection_vocabulary = [x[0] for x in featureselection]
        print('Loaded',len(test_featureselection_vocabulary),'selected testfeatures, now aligning testvectors...')

        # transform vectors
        testvectors = vectorizer.align_vectors(testinstances, train_featureselection_vocabulary, test_featureselection_vocabulary)
        
        # write instances to file
        numpy.savez(self.out_vectors().path, data=testvectors.data, indices=testvectors.indices, indptr=testvectors.indptr, shape=testvectors.shape)

        # write featureselection to file
        with open(self.out_test_featureselection().path, 'w', encoding = 'utf-8') as t_out:
            t_out.write('\n'.join(trainlines))
 
class VectorizeTrainTask(Task):

    in_trainfeatures = InputSlot()
    in_trainlabels = InputSlot()

    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()

    def out_train(self):
        return self.outputfrominput(inputformat='trainfeatures', stripextension='.features.npz', addextension='.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.balance else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz')

    def out_trainlabels(self):
           return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.balanced.labels' if self.balance else '.labels')       
    
    def run(self):
        
        if self.complete(): # necessary as it will not complete otherwise
            return True
        else:
            yield Vectorize(traininstances=self.in_trainfeatures().path,trainlabels=self.in_trainlabels().path,weight=self.weight,prune=self.prune,balance=self.balance)

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
    
class VectorizeTestTask(Task):

    in_trainvectors = InputSlot()
    in_testfeatures = InputSlot()
    in_trainlabels = InputSlot()

    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()
     
    def out_vectors(self):
        return self.outputfrominput(inputformat='testfeatures', stripextension='.features.npz', addextension='.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.balance else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz')

    def run(self):

        if self.complete(): # necessary as it will not complete otherwise
            return True
        else:
            yield Vectorize(traininstances=self.in_trainvectors().path,trainlabels=self.in_trainlabels().path,testinstances=self.in_testfeatures().path,weight=self.weight,prune=self.prune,balance=self.balance)

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
    num_folds = IntParameter(default=5)
    fold_steps = IntParameter(default=1)
    num_iterations = IntParameter(default=300)
    population_size = IntParameter(default=100)
    crossover_probability = Parameter(default='0.9')
    mutation_rate = Parameter(default='0.3')
    tournament_size = IntParameter(default=2)
    n_crossovers = IntParameter(default=1)
    stop_condition = IntParameter(default=5)

    # classifier parameters
    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter()
    jobs = IntParameter(default=1)
    iterations = IntParameter(default=10)
    scoring = Parameter(default='roc_auc') # optimization metric for grid search
    
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
            ]
            )).T.reshape(-1,3)]

    def setup(self, workflow, input_feeds):
        
        ######################
        ### Training phase ###
        ######################
        
        trainlabels = input_feeds['labels_train']
        
        if 'vectorized_train' in input_feeds.keys():
            if self.balance and 'balanced' not in input_feeds['vectorized_train'].split('/')[-1].split('.'):
                balancetask = workflow.new_task('BalanceTaskVector',Balance,autopass=True)
                balancetask.in_train = input_feeds['vectorized_train']
                balancetask.in_trainlabels = trainlabels

                trainvectors = balancetask.out_train
                trainlabels = balancetask.out_labels
            else:
                trainvectors = input_feeds['vectorized_train']

            
            if self.ga and 'ga' not in input_feeds['vectorized_train']().path.split('/')[-1].split('.'):
                wrapper = workflow.new_task('run_ga',RunGA,autopass=True,
                    num_folds=self.num_folds,fold_steps=self.fold_steps,num_iterations=self.num_iterations, population_size=self.population_size, crossover_probability=self.crossover_probability,
                    mutation_rate=self.mutation_rate,tournament_size=self.tournament_size,n_crossovers=self.n_crossovers,stop_condition=self.stop_condition,
                    classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,
                    nb_alpha=self.nb_alpha,nb_fit_prior=self.nb_fit_prior,
                    svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
                    lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter,
                    xg_booster=self.xg_booster, xg_silent=self.xg_silent, xg_learning_rate=self.xg_learning_rate, xg_min_child_weight=self.xg_min_child_weight, 
                    xg_max_depth=self.xg_max_depth, xg_gamma=self.xg_gamma, xg_max_delta_step=self.xg_max_delta_step, xg_subsample=self.xg_subsample, 
                    xg_colsample_bytree=self.xg_colsample_bytree, xg_reg_lambda=self.xg_reg_lambda, xg_reg_alpha=self.xg_reg_alpha, xg_scale_pos_weight=self.xg_scale_pos_weight,
                    xg_objective=self.xg_objective, xg_seed=self.xg_seed, xg_n_estimators=self.xg_n_estimators,
                    knn_n_neighbors=self.knn_n_neighbors, knn_weights=self.knn_weights, knn_algorithm=self.knn_algorithm, knn_leaf_size=self.knn_leaf_size,
                    knn_metric=self.knn_metric, knn_p=self.knn_p)
                wrapper.in_train = trainvectors
                wrapper.in_trainlabels = trainlabels
                trainvectors = wrapper.out_vectors
                
        else: # pre_vectorized

            if 'featurized_train_csv' in input_feeds.keys():
                trainvectorizer_csv = workflow.new_task('train_vectorizer_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter)
                trainvectorizer_csv.in_csv = input_feeds['featurized_train_csv']

                if self.scale:
                    trainvectorizer = workflow.new_task('scale_trainvectors',FitTransformScale,autopass=True)
                    trainvectorizer.in_vectors = trainvectorizer_csv.out_vectors

                else:
                    trainvectorizer = trainvectorizer_csv

                if self.balance:
                    balancetask = workflow.new_task('BalanceTaskCSV',Balance,autopass=True)
                    balancetask.in_train = trainvectorizer.out_vectors
                    balancetask.in_trainlabels = labels

                    trainvectors = balancetask.out_train
                    trainlabels = balancetask.out_labels
                else:
                    trainvectors = trainvectorizer.out_vectors

                if self.ga:
                    wrapper = workflow.new_task('run_ga',RunGA,autopass=True,
                        num_folds=self.num_folds,fold_steps=self.fold_steps,num_iterations=self.num_iterations, population_size=self.population_size, crossover_probability=self.crossover_probability,
                        mutation_rate=self.mutation_rate,tournament_size=self.tournament_size,n_crossovers=self.n_crossovers,stop_condition=self.stop_condition,
                        classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,
                        nb_alpha=self.nb_alpha,nb_fit_prior=self.nb_fit_prior,
                        svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
                        lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter,
                        xg_booster=self.xg_booster, xg_silent=self.xg_silent, xg_learning_rate=self.xg_learning_rate, xg_min_child_weight=self.xg_min_child_weight, 
                        xg_max_depth=self.xg_max_depth, xg_gamma=self.xg_gamma, xg_max_delta_step=self.xg_max_delta_step, xg_subsample=self.xg_subsample, 
                        xg_colsample_bytree=self.xg_colsample_bytree, xg_reg_lambda=self.xg_reg_lambda, xg_reg_alpha=self.xg_reg_alpha, xg_scale_pos_weight=self.xg_scale_pos_weight,
                        xg_objective=self.xg_objective, xg_seed=self.xg_seed, xg_n_estimators=self.xg_n_estimators,
                        knn_n_neighbors=self.knn_n_neighbors, knn_weights=self.knn_weights, knn_algorithm=self.knn_algorithm, knn_leaf_size=self.knn_leaf_size,
                        knn_metric=self.knn_metric, knn_p=self.knn_p)
                    wrapper.in_train = trainvectors
                    wrapper.in_trainlabels = trainlabels
                    trainvectors = wrapper.out_vectors
                    
            else:

                if 'pre_featurized_train' in input_feeds.keys():
                    trainfeaturizer = workflow.new_task('featurize_train',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                    trainfeaturizer.in_pre_featurized = input_feeds['pre_featurized_train']

                    featurized_train = trainfeaturizer.out_featurized

                else:
                    featurized_train = input_feeds['featurized_train']

                trainvectorizer = workflow.new_task('vectorize_train',VectorizeTrainTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
                trainvectorizer.in_trainfeatures = featurized_train
                trainvectorizer.in_trainlabels = trainlabels

                trainvectors = trainvectorizer.out_train
                trainlabels = trainvectorizer.out_trainlabels

                if self.ga:
                    wrapper = workflow.new_task('run_ga',RunGA,autopass=True,
                        num_folds=self.num_folds,fold_steps=self.fold_steps,num_iterations=self.num_iterations, population_size=self.population_size, crossover_probability=self.crossover_probability,
                        mutation_rate=self.mutation_rate,tournament_size=self.tournament_size,n_crossovers=self.n_crossovers,stop_condition=self.stop_condition,
                        classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,
                        nb_alpha=self.nb_alpha,nb_fit_prior=self.nb_fit_prior,
                        svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
                        lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter,
                        xg_booster=self.xg_booster, xg_silent=self.xg_silent, xg_learning_rate=self.xg_learning_rate, xg_min_child_weight=self.xg_min_child_weight, 
                        xg_max_depth=self.xg_max_depth, xg_gamma=self.xg_gamma, xg_max_delta_step=self.xg_max_delta_step, xg_subsample=self.xg_subsample, 
                        xg_colsample_bytree=self.xg_colsample_bytree, xg_reg_lambda=self.xg_reg_lambda, xg_reg_alpha=self.xg_reg_alpha, xg_scale_pos_weight=self.xg_scale_pos_weight,
                        xg_objective=self.xg_objective, xg_seed=self.xg_seed, xg_n_estimators=self.xg_n_estimators,
                        knn_n_neighbors=self.knn_n_neighbors, knn_weights=self.knn_weights, knn_algorithm=self.knn_algorithm, knn_leaf_size=self.knn_leaf_size,
                        knn_metric=self.knn_metric, knn_p=self.knn_p)
                    wrapper.in_train = trainvectors
                    wrapper.in_trainlabels = trainlabels
                    trainvectors = wrapper.out_vectors
                
        trainer = workflow.new_task('train',Train,autopass=True,classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,
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
        trainer.in_train = trainvectors
        trainer.in_trainlabels = trainlabels            

        ######################
        ### Testing phase ####
        ######################

        if len(list(set(['vectorized_test','featurized_test_csv','featurized_test','pre_featurized_test']) & set(list(input_feeds.keys())))) > 0:
 
            if 'vectorized_test' in input_feeds.keys():
                testvectors = input_feeds['vectorized_test']

                if self.ga and 'ga' not in input_feeds['vectorized_test'].split('/')[-1].split('.'):
                    ga_vectorizer = workflow.new_task('ga_vectorizer',TransformVectors,autopass=True)
                    ga_vectorizer.in_test = testvectors
                    ga_vectorizer.in_train = trainvectors
                    testvectors = ga_vectorizer.out_vectors

            else: # pre_vectorized

                if 'featurized_test_csv' in input_feeds.keys():
                    testvectorizer_csv = workflow.new_task('vectorizer_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter)
                    testvectorizer_csv.in_csv = input_feeds['featurized_test_csv']

                    if self.scale:
                        testvectorizer = workflow.new_task('scale_testvectors',TransformScale,autopass=True)
                        testvectorizer.in_vectors = testvectorizer_csv.out_vectors
                        testvectorizer.in_scaler = trainvectorizer.out_scaler

                    else:
                        testvectorizer = testvectorizer_csv

                    testvectors = testvectorizer.out_vectors

                    if self.ga:
                        ga_vectorizer = workflow.new_task('ga_vectorizer',TransformVectors,autopass=True)
                        ga_vectorizer.in_test = testvectors
                        ga_vectorizer.in_train = trainvectors
                        testvectors = ga_vectorizer.out_vectors
        
                else:

                    if 'pre_featurized_test' in input_feeds.keys():
                        testfeaturizer = workflow.new_task('featurize_test',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                        testfeaturizer.in_pre_featurized = input_feeds['pre_featurized_test']

                        featurized_test = testfeaturizer.out_featurized

                    else:
                        featurized_test = input_feeds['featurized_test']
                    
                    testvectorizer = workflow.new_task('vectorize_test',VectorizeTestTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
                    testvectorizer.in_trainvectors = trainvectors
                    testvectorizer.in_trainlabels = trainlabels
                    testvectorizer.in_testfeatures = featurized_test

                    testvectors = testvectorizer.out_vectors

                    if self.ga:
                        ga_vectorizer = workflow.new_task('ga_vectorizer',TransformVectors,autopass=True)
                        ga_vectorizer.in_test = testvectors
                        ga_vectorizer.in_train = trainvectors
                        testvectors = ga_vectorizer.out_vectors
                    
            if 'classifier_model' in input_feeds.keys():
                model = input_feeds['classifier_model']
            else:
                model = trainer.out_model

            predictor = workflow.new_task('predictor',Predict,autopass=True,classifier=self.classifier,ordinal=self.ordinal)
            predictor.in_test = testvectors
            predictor.in_trainlabels = trainlabels
            predictor.in_model = model

            return predictor

        else:

            return trainer
