
import numpy
from scipy import sparse
import glob
from collections import defaultdict

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter, FloatParameter

from quoll.classification_pipeline.modules.validate import Validate
from quoll.classification_pipeline.modules.validate_append import ValidateAppend
from quoll.classification_pipeline.modules.report import Report, ReportPerformance, ReportDocpredictions, ReportFolds, ClassifyTask, TrainTask
from quoll.classification_pipeline.modules.classify import Train, Predict, VectorizeTrain, VectorizeTrainTest, VectorizeTrainCombinedTask, VectorizeTestCombinedTask, TranslatePredictions 
from quoll.classification_pipeline.modules.vectorize import Vectorize, TransformCsv, FeaturizeTask, Combine

from quoll.classification_pipeline.functions import reporter, quoll_helpers, linewriter, docreader

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
    
    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()
    featurize_parameters = Parameter()
    preprocess_parameters = Parameter()

    def out_report(self):
        return self.outputfrominput(inputformat='test', stripextension='.'.join(self.in_test().path.split('.')[-2:]) if (self.in_test().path[-3:] == 'npz' or self.in_test().path[-7:-4] == 'tok') else '.' + self.in_test().path.split('.')[-1], addextension='.report' if self.testlabels_true else '.docpredictions.csv')

    def run(self):

        if self.complete(): # necessary as it will not complete otherwise
            return True

        kwargs = quoll_helpers.decode_task_input(['ga','classify','vectorize','featurize','preprocess'],[self.ga_parameters,self.classify_parameters,self.vectorize_parameters,self.featurize_parameters,self.preprocess_parameters])
        print('REPORT KWARGS',kwargs)
        yield Report(train=self.in_train().path,test=self.in_test().path,trainlabels=self.in_trainlabels().path,testlabels=self.in_testlabels().path,testdocs=self.in_testdocs().path,**kwargs)

class ValidateAppendTask(Task):

    in_instances = InputSlot()
    in_instances_append = InputSlot()
    in_labels = InputSlot()
    in_docs = InputSlot()
    
    validate_parameters = Parameter()
    append_parameters = Parameter()
    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()
    featurize_parameters = Parameter()
    preprocess_parameters = Parameter()
   
    def out_exp(self):
        return self.outputfrominput(inputformat='instances', stripextension='.'.join(self.in_instances().path.split('.')[-2:]) if (self.in_instances().path[-3:] == 'npz' or self.in_instances().path[-7:-4] == 'tok') else '.' + self.in_instances().path.split('.')[-1], addextension='.validated.report')
                                    
    def run(self):

        if self.complete(): # necessary as it will not complete otherwise
            return True

        kwargs = quoll_helpers.decode_task_input(['validate','append','ga','classify','vectorize','featurize','preprocess'],[self.validate_parameters,self.append_parameters,self.ga_parameters,self.classify_parameters,self.vectorize_parameters,self.featurize_parameters,self.preprocess_parameters])
        yield ValidateAppend(instances=self.in_instances().path,labels=self.in_labels().path,docs=self.in_docs().path,**kwargs)


class ValidateTask(Task):

    in_instances = InputSlot()
    in_labels = InputSlot()
    in_docs = InputSlot()

    validate_parameters = Parameter()
    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()
    featurize_parameters = Parameter()
    preprocess_parameters = Parameter()
   
    def out_exp(self):
        return self.outputfrominput(inputformat='instances', stripextension='.'.join(self.in_instances().path.split('.')[-2:]) if (self.in_instances().path[-3:] == 'npz' or self.in_instances().path[-7:-4] == 'tok') else '.' + self.in_instances().path.split('.')[-1], addextension='.validated.report')
                                    
    def run(self):

        if self.complete(): # necessary as it will not complete otherwise
            return True

        kwargs = quoll_helpers.decode_task_input(['validate','ga','classify','vectorize','featurize','preprocess'],[self.validate_parameters,self.ga_parameters,self.classify_parameters,self.vectorize_parameters,self.featurize_parameters,self.preprocess_parameters])
        yield Validate(instances=self.in_instances().path,labels=self.in_labels().path,docs=self.in_docs().path,**kwargs)


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

    # append parameters
    bow_as_feature = BoolParameter() # to combine bow as separate classification with other features, only relevant in case of train_append
    bow_classifier = Parameter(default='naive_bayes')
    bow_include_labels = Parameter(default='all') # will give prediction probs as feature for each label by default, can specify particular labels (separated by a space) here, only applies when 'bow_prediction_probs' is chosen
    bow_prediction_probs = BoolParameter() # choose to add prediction probabilities                
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
        
        task_args = quoll_helpers.prepare_task_input(['preprocess','featurize','vectorize','classify','ga','append','validate'],workflow.param_kwargs)

        ######################
        ### Training phase ###
        ######################

        trainlabels = input_feeds['labels_train']

        if 'vectorized_train_append' in input_feeds.keys():
            train_append = input_feeds['vectorized_train_append']
        elif 'vectorized_csv_train_append' in input_feeds.keys():
            train_append = input_feeds['vectorized_csv_train_append']
        else:
            train_append = False

        if 'docs_train' in input_feeds.keys() or 'pre_featurized_train' in input_feeds.keys(): # both stored as docs and featurized

            if 'pre_featurized_train' in input_feeds.keys():
                train = input_feeds['pre_featurized_train']
            else: # docs (.txt)
                docs = input_feeds['docs_train']
                train = input_feeds['docs_train']

        elif 'featurized_train' in input_feeds.keys(): 
            train = input_feeds['featurized_train']
        elif 'vectorized_csv_train' in input_feeds.keys():
            train = input_feeds['vectorized_csv_train']
        elif 'vectorized_train' in input_feeds.keys():
            train = input_feeds['vectorized_train']

        if not 'test' in [x.split('_')[-1] for x in input_feeds.keys()]: # only train input --> nfold-cv

        ######################
        ### Nfold CV #########
        ######################

            if 'docs' in input_feeds.keys():
                docs = input_feeds['docs']

            if train_append:

                validator = workflow.new_task('validate_append', ValidateAppendTask, autopass=True,
                    preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],
                    classify_parameters=task_args['classify'],ga_parameters=task_args['ga'],validate_parameters=task_args['validate']                   
                )
                validator.in_instances = train
                validator.in_instances_append = train_append
                validator.in_labels = trainlabels
                validator.in_docs = docs

            else:

                validator = workflow.new_task('validate', ValidateTask, autopass=True,
                    preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],
                    classify_parameters=task_args['classify'],ga_parameters=task_args['ga'],validate_parameters=task_args['validate']  
                )
                validator.in_instances = train
                validator.in_labels = trainlabels
                validator.in_docs = docs

            # also train a model on all data
            if train_append:
                # if train:
                    trainvectorizer = workflow.new_task('vectorize_trainvectors_combined',Combine,autopass=True)
                    trainvectorizer.in_vectors = train
                    trainvectorizer.in_vectors_append = train_append

                    trainvectors_combined = trainvectorizer.out_combined

                # else: # featurized train

                #     trainvectorizer = workflow.new_task('vectorize_train_combined',VectorizeTrainCombinedTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
                #     trainvectorizer.in_trainfeatures = featurized_train
                #     trainvectorizer.in_trainvectors_append = trainvectors_append
                #     trainvectorizer.in_trainlabels = trainlabels

                #     trainvectors_combined = trainvectorizer.out_train_combined
                #     trainvectors = trainvectorizer.out_train
                #     trainlabels = trainvectorizer.out_trainlabels

            else:

                trainvectors_combined = False

                # if not trainvectors:
                #     trainvectorizer = workflow.new_task('vectorize_train',VectorizeTrainTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
                #     trainvectorizer.in_trainfeatures = featurized_train
                #     trainvectorizer.in_trainlabels = trainlabels

                #     trainvectors = trainvectorizer.out_train
                #     trainlabels = trainvectorizer.out_trainlabels
            foldtrainer = workflow.new_task('train',TrainTask,autopass=True,
                preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],classify_parameters=task_args['classify'],ga_parameters=task_args['ga']
            )
            if trainvectors_combined:
                foldtrainer.in_train = trainvectors_combined
                fold_trainer.in_test = trainvectors_combined
            else:
                foldtrainer.in_train = train
                foldtrainer.in_test = train
            foldtrainer.in_trainlabels = trainlabels    

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
            #     test_append = input_feeds['vectorized_test_append']
            # elif 'vectorized_csv_test_append' in input_feeds.keys():
            #     test_append = input_feeds['vectorized_csv_test_append']
            # else:
            #     test_append = False
                
            if 'vectorized_test' in input_feeds.keys():
                test = input_feeds['vectorized_test']
            elif 'vectorized_csv_test' in input_feeds.keys():
                test = input_feeds['vectorized_csv_test']
            elif 'featurized_test' in input_feeds.keys():
                test = input_feeds['featurized_test']
            elif 'pre_featurized_test' in input_feeds.keys():
                test = input_feeds['pre_featurized_test']
            else:
                docs = input_feeds['docs_test']
                test = input_feeds['docs_test']

                    # testfeaturizer = workflow.new_task('featurize_test',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                    # testfeaturizer.in_pre_featurized = pre_featurized

                    # testinstances = testfeaturizer.out_featurized


            #     traincsv=True if ('vectorized_csv_train' in input_feeds.keys()) else False
            #     trainvec=True if ('vectorized_train' in input_feeds.keys()) else False
            #     testcsv=True if ('vectorized_csv_test' in input_feeds.keys()) else False
            #     testvec=True if ('vectorized_test' in input_feeds.keys()) else False
                    
            #     vectorizer = workflow.new_task('vectorize_traintest',VectorizeTrainTest,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance,select=self.select,select_threshold=self.select_threshold,delimiter=self.delimiter,traincsv=traincsv,trainvec=trainvec,testcsv=testcsv,testvec=testvec)
            #     vectorizer.in_train = traininstances
            #     vectorizer.in_trainlabels = trainlabels
            #     vectorizer.in_test = testinstances

            #     traininstances = vectorizer.out_train
            #     trainlabels = vectorizer.out_trainlabels
            #     testinstances = vectorizer.out_test
                    
            if 'labels_test' in input_feeds.keys():
                testlabels = input_feeds['labels_test']
                testlabels_true = True
            else:
                testlabels = test
                testlabels_true = False
                
            if 'docs' in input_feeds.keys():
                docs = input_feeds['docs']
            else:
                if not 'docs_test' in input_feeds.keys():
                    print('No docs in input parameters, exiting programme...')
                    quit()

            reporter = workflow.new_task('report', ReportTask, autopass=True, 
                testlabels_true=testlabels_true,preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],classify_parameters=task_args['classify'],ga_parameters=task_args['ga']                   
            )
            reporter.in_train = train
            reporter.in_test = test
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
            return foldtrainer, validator
        else:
            return reporter
