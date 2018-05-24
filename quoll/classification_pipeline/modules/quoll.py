
import numpy
from scipy import sparse
import glob
from collections import defaultdict

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter, FloatParameter

from quoll.classification_pipeline.modules.validate import  MakeBins, Folds, FoldsAppend, ReportFolds
from quoll.classification_pipeline.modules.report import ReportPerformance, ReportDocpredictions
from quoll.classification_pipeline.modules.classify import Train, Predict, VectorizeTrainTask, VectorizeTrainCombinedTask, VectorizeTestTask, VectorizeTestCombinedTask
from quoll.classification_pipeline.modules.vectorize import Vectorize, VectorizeCsv, FeaturizeTask, Combine

from quoll.classification_pipeline.functions import reporter, nfold_cv_functions, linewriter, docreader

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

    # classifier parameters
    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter()
    jobs = IntParameter(default=1)
    iterations = IntParameter(default=10)
    
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
    bow_as_feature = BoolParameter() # to combine bow as separate classification with other features, only relevant in case of train_append
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
                InputFormat(self, format_id='modeled_train',extension ='.model.pkl',inputparameter='train'),
                InputFormat(self, format_id='vectorized_train',extension='.vectors.npz',inputparameter='train'),
                InputFormat(self, format_id='featurized_train',extension='.features.npz',inputparameter='train'),
                InputFormat(self, format_id='featurized_csv_train',extension='.csv',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txt',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txtdir',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.json',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.jsondir',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.txtdir',inputparameter='train'),
                InputFormat(self, format_id='docs_train',extension='.txt',inputparameter='train')
                ),
                (
                InputFormat(self, format_id='vectorized_train_append',extension='.vectors.npz',inputparameter='train_append'),
                InputFormat(self, format_id='featurized_csv_train_append',extension='.csv',inputparameter='train_append'),
                ),
                (
                InputFormat(self, format_id='classified_test',extension='.predictions.txt',inputparameter='test'),
                InputFormat(self, format_id='vectorized_test',extension='.vectors.npz',inputparameter='test'),
                InputFormat(self, format_id='featurized_test',extension='.features.npz',inputparameter='test'),
                InputFormat(self, format_id='featurized_csv_test',extension='.csv',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.tok.txt',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.tok.txtdir',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.frog.json',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.frog.jsondir',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.txtdir',inputparameter='test'),
                InputFormat(self, format_id='docs_test',extension='.txt',inputparameter='test')
                ),
                (
                InputFormat(self, format_id='vectorized_test_append',extension='.vectors.npz',inputparameter='test_append'),
                InputFormat(self, format_id='featurized_csv_test_append',extension='.csv',inputparameter='test_append'),
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
        elif 'featurized_csv_train_append' in input_feeds.keys():
            trainvectorizer_append = workflow.new_task('vectorize_train_csv_append',VectorizeCsv,autopass=True,delimiter=self.delimiter)
            trainvectorizer_append.in_csv = input_feeds['featurized_csv_train_append']
                
            trainvectors_append = trainvectorizer_append.out_vectors
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

            featurized_train = trainfeaturizer.out_featurized

        elif 'featurized_train' in input_feeds.keys(): 
            featurized_train = input_feeds['featurized_train']

        elif 'featurized_csv_train' in input_feeds.keys():
            trainvectorizer = workflow.new_task('vectorize_train_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter)
            trainvectorizer.in_csv = input_feeds['featurized_csv_train']

            trainvectors = trainvectorizer.out_vectors

        elif 'vectorized_train' in input_feeds.keys():
            trainvectors = input_feeds['vectorized_train']

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

            bin_maker = workflow.new_task('make_bins', MakeBins, autopass=True, n=self.n, teststart=self.teststart, testend=self.testend)
            bin_maker.in_labels = trainlabels

            if trainvectors_append:

                fold_runner = workflow.new_task('nfold_cv_append', FoldsAppend, autopass=True, 
                    n=self.n, 
                    weight=self.weight, prune=self.prune, balance=self.balance, 
                    classifier=self.classifier, ordinal=self.ordinal, jobs=self.jobs, iterations=self.iterations,
                    nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
                    svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
                    lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter
                )
                fold_runner.in_bins = bin_maker.out_bins
                fold_runner.in_instances = instances
                fold_runner.in_instances_append = trainvectors_append
                fold_runner.in_labels = trainlabels
                fold_runner.in_docs = docs

            else:

                fold_runner = workflow.new_task('nfold_cv', Folds, autopass=True, 
                    n=self.n, 
                    weight=self.weight, prune=self.prune, balance=self.balance, 
                    classifier=self.classifier, ordinal=self.ordinal, jobs=self.jobs, iterations=self.iterations,
                    nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
                    svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
                    lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter
                )
                fold_runner.in_bins = bin_maker.out_bins
                fold_runner.in_instances = instances
                fold_runner.in_labels = trainlabels
                fold_runner.in_docs = docs      

            foldreporter = workflow.new_task('report_folds', ReportFolds, autopass=True)
            foldreporter.in_exp = fold_runner.out_exp

            labels = foldreporter.out_labels
            predictions = foldreporter.out_predictions

        else:

            if 'modeled_train' in input_feeds.keys():
                trainmodel = input_feeds['modeled_train']

            else:

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

                trainmodel = trainer.out_model 

            if 'classified_test' in input_feeds.keys(): # reporter can be started
                predictions = input_feeds['classified_test']

            else: 

            ######################
            ### Testing phase ####
            ######################

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

                elif 'featurized_csv_test' in input_feeds.keys():
                    testvectorizer = workflow.new_task('vectorize_test_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter)
                    testvectorizer.in_csv = input_feeds['featurized_csv_test']

                    testvectors = testvectorizer.out_vectors

                else:
                    
                    testvectors = False
                    if 'docs_test' in input_feeds.keys() or 'pre_featurized_test' in input_feeds.keys():

                        if 'pre_featurized_test' in input_feeds.keys():
                            pre_featurized = input_feeds['pre_featurized_test']

                        else:
                            docs = input_feeds['docs_test']
                            pre_featurized = input_feeds['docs_test']

                        testfeaturizer = workflow.new_task('featurize_test',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                        testfeaturizer.in_pre_featurized = pre_featurized

                        featurized_test = testfeaturizer.out_featurized

                    else:
                        featurized_test = input_feeds['featurized_test']

                if testvectors_append:
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

                        testvectors = testvectorizer.out_vectors
                        
                else:
                    if not testvectors:
                        testvectorizer = workflow.new_task('vectorize_test',VectorizeTestTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
                        testvectorizer.in_trainvectors = trainvectors
                        testvectorizer.in_trainlabels = trainlabels
                        testvectorizer.in_testfeatures = featurized_test

                        testvectors = testvectorizer.out_vectors

                predictor = workflow.new_task('predictor',Predict,autopass=True,classifier=self.classifier,ordinal=self.ordinal)
                predictor.in_test = testvectors
                predictor.in_trainlabels = trainlabels
                predictor.in_model = trainmodel

                predictions = predictor.out_predictions

            if 'docs' in input_feeds.keys():
                docs = input_feeds['docs']
                
            if 'labels_test' in input_feeds.keys(): # full performance reporter
                labels = input_feeds['labels_test']

            else:
                labels = False

        ######################
        ### Reporting phase ##
        ######################

        if labels:

            reporter = workflow.new_task('report_performance',ReportPerformance,autopass=True,ordinal=self.ordinal)
            reporter.in_predictions = predictions
            reporter.in_testlabels = labels
            reporter.in_testdocuments = docs

        else: # report docpredictions

            reporter = workflow.new_task('report_docpredictions',ReportDocpredictions,autopass=True)
            reporter.in_predictions = predictions
            reporter.in_testdocuments = docs

        return reporter
