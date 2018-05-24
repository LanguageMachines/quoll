
import os
import numpy
from scipy import sparse
import pickle
import math
from collections import defaultdict

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.functions.classifier import *
from quoll.classification_pipeline.modules.report import MakeBins, Folds, ReportFolds
from quoll.classification_pipeline.modules.vectorize import Vectorize, VectorizeCsv, FeaturizeTask, Combine

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
                        'knn':[KNNClassifier(),[]], 
                        'svm':[SVMClassifier(),[self.svm_c,self.svm_kernel,self.svm_gamma,self.svm_degree,self.svm_class_weight,self.iterations,self.jobs]], 
                        'tree':[TreeClassifier(),[]], 
                        'perceptron':[PerceptronLClassifier(),[]], 
                        'logistic_regression':[LogisticRegressionClassifier(),[self.lr_c,self.lr_solver,self.lr_dual,self.lr_penalty,self.lr_multiclass,self.lr_maxiter]], 
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
    traininstances_append = Parameter(default = 'xxx.xxx')
    trainlabels = Parameter()
    testinstances = Parameter(default = 'xxx.xxx') # not obligatory, dummy extension to enable a pass
    testinstances_append = Parameter(default = 'xxx.xxx')
    traindocs = Parameter(default = 'xxx.xxx')

    # classifier parameters
    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter()
    jobs = IntParameter(default=1)
    iterations = IntParameter(default=10)
    
    bow_as_feature = BoolParameter() # to combine bow as separate classification with other features, only relevant in case of train_append
    bow_classifier = Parameter(default=self.classifier)

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
    
    # vectorizer parameters
    weight = Parameter(default = 'frequency') # options: frequency, binary, tfidf
    prune = IntParameter(default = 5000) # after ranking the topfeatures in the training set, based on frequency or idf weighting
    balance = BoolParameter()
    delimiter = Parameter(default=',')

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

        if 'vectorized_train_append' in input_feeds.keys():
            trainvectors_append = input_feeds['vectorized_train_append']
        elif 'featurized_csv_train_append' in input_feeds.keys():
            trainvectorizer_append = workflow.new_task('vectorize_train_csv_append',VectorizeCsv,autopass=True,delimiter=self.delimiter)
            trainvectorizer_append.in_csv = input_feeds['featurized_csv_train_append']
                
            trainvectors_append = trainvectorizer_append.out_vectors
        else:
            trainvectors_append = False
        
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
                
        if trainvectors_append:

            if self.bow_as_feature:
                if trainvectors:
                    print('Bag-of-words as features can only be ran on featurized train instances (ending with \'.features.npz\', exiting programme...')
                    exit()

                # make bag-of-words predictions on training instances using 5-fold cv
                bin_maker = workflow.new_task('make_bins_bow', MakeBins, autopass=True, n=5)
                bin_maker.in_labels = trainlabels

                fold_runner = workflow.new_task('nfold_cv_bow', Folds, autopass=True, 
                    n=5, 
                    weight=self.weight, prune=self.prune, balance=self.balance, 
                    classifier=self.bow_classifier, ordinal=self.ordinal, jobs=self.jobs, iterations=self.iterations,
                    nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
                    svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
                    lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter
                )
                fold_runner.in_bins = bin_maker.out_bins
                fold_runner.in_instances = featurized_train
                fold_runner.in_labels = trainlabels
                fold_runner.in_docs = input_feeds['docs_train']

                foldreporter = workflow.new_task('report_folds_bow', ReportFolds, autopass=True)
                foldreporter.in_exp = fold_runner.out_exp

                fold_vectorizer = workflow.new_task('vectorize_foldreporter', VectorizeFoldreporter, autopass=True)
                fold_vectorizer.in_predictions = foldreporter.out_predictions
                fold_vectorizer.in_bins = bin_maker.out_bins

                trainvectors = fold_vectorizer.out_vectors

                trainvectorizer = workflow.new_task('vectorize_train',VectorizeTrainTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
                trainvectorizer.in_trainfeatures = featurized_train
                trainvectorizer.in_trainlabels = trainlabels

                trainvectors_bow = trainvectorizer.out_vectors
                trainlabels = trainvectorizer.out_trainlabels

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
                
        trainer = workflow.new_task('train',Train,autopass=True,classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,
            nb_alpha=self.nb_alpha,nb_fit_prior=self.nb_fit_prior,
            svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
            lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter
        )
        if trainvectors_combined:
            trainer.in_train = trainvectors_combined
        else:
            trainer.in_train = trainvectors
        trainer.in_trainlabels = trainlabels            

        ######################
        ### Testing phase ####
        ######################

        if len(list(set(['vectorized_test','featurized_test_csv','featurized_test','pre_featurized_test']) & set(list(input_feeds.keys())))) > 0:

            if 'vectorized_test_append' in input_feeds.keys():
                testvectors_append = input_feeds['vectorized_test_append']
            elif 'featurized_csv_test_append' in input_feeds.keys():
                testvectorizer_append = workflow.new_task('vectorize_test_csv_append',VectorizeCsv,autopass=True,delimiter=self.delimiter)
                testvectorizer_append.in_csv = input_feeds['featurized_csv_test_append']

                testvectors_append = testvectorizer_append.out_vectors
            else:
                testvectors_append = False
 
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

            if testvectors_append:
                
                if self.bow_as_feature:
                    if testvectors:
                        print('Bag-of-words as features can only be ran on featurized test instances (ending with \'.features.npz\', exiting programme...')
                        exit()

                    bow_trainer = workflow.new_task('train_bow',Train,autopass=True,classifier=self.bow_classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,
                        nb_alpha=self.nb_alpha,nb_fit_prior=self.nb_fit_prior,
                        svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
                        lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter
                    )
                    bow_trainer.in_train = trainvectors_bow
                    bow_trainer.in_trainlabels = trainlabels            

                    testvectorizer = workflow.new_task('vectorize_test',VectorizeTestTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
                    testvectorizer.in_trainvectors = trainvectors_bow
                    testvectorizer.in_trainlabels = trainlabels
                    testvectorizer.in_testfeatures = featurized_test

                    bow_predictor = workflow.new_task('predictor_bow',Predict,autopass=True,classifier=self.bow_classifier,ordinal=self.ordinal)
                    bow_predictor.in_test = testvectorizer.out_vectors
                    bow_predictor.in_trainlabels = trainlabels
                    bow_predictor.in_model = bow_trainer.out_model

                    prediction_vectorizer = workflow.new_task('vectorize_predictions', VectorizePredictions, autopass=True)
                    prediction_vectorizer.in_predictions = bow_predictor.out_predictions

                    testvectors = prediction_vectorizer.out_vectors

                if testvectors:
                    testvectorizer = workflow.new_task('vectorize_test_combined_vectors',Combine,autopass=True)
                    testvectorizer.in_vectors = testvectors
                    testvectorizer.in_vectors_append = testvectors_append

                    testvectors = testvectorizer.out_combined
                    
                else:
                    testvectorizer = workflow.new_task('vectorize_test_combined',VectorizeTestCombinedTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
                    testvectorizer.in_trainvectors = trainvectors
                    testvectorizer.in_trainlabels = trainlabels
                    testvectorizer.in_testfeatures = featurized_test
                    testvectorizer.in_testvectors_append = testvectors_append

                    testvectors=testvectorizer.out_vectors
                    
            else:
                if not testvectors:
                    testvectorizer = workflow.new_task('vectorize_test',VectorizeTestTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
                    testvectorizer.in_trainvectors = trainvectors
                    testvectorizer.in_trainlabels = trainlabels
                    testvectorizer.in_testfeatures = featurized_test

                    testvectors = testvectorizer.out_vectors
                    
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
