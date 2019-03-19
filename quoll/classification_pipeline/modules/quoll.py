
import numpy

from luiginlp.engine import Task, WorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.modules.report import Report, TrainTask
from quoll.classification_pipeline.modules.validate import ValidateTask
from quoll.classification_pipeline.modules.validate_append import ValidateAppendTask
from quoll.classification_pipeline.modules.validate_ensemble import ValidateEnsembleTask
from quoll.classification_pipeline.modules.classify_ensemble import EnsembleTrain, EnsembleTrainTest
from quoll.classification_pipeline.modules.classify_append import ClassifyAppend 

from quoll.classification_pipeline.functions import quoll_helpers

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
        return self.outputfrominput(inputformat='test', stripextension='.'.join(self.in_test().path.split('.')[-2:]) if (self.in_test().path[-3:] == 'npz' or self.in_test().path[-7:-4] == 'tok' or self.in_test().path[-15:] == 'predictions.txt') else '.' + self.in_test().path.split('.')[-1], addextension='.report' if self.testlabels_true else '.docpredictions.csv')

    def run(self):

        if self.complete(): # necessary as it will not complete otherwise
            return True

        kwargs = quoll_helpers.decode_task_input(['ga','classify','vectorize','featurize','preprocess'],[self.ga_parameters,self.classify_parameters,self.vectorize_parameters,self.featurize_parameters,self.preprocess_parameters])
        yield Report(train=self.in_train().path,test=self.in_test().path,trainlabels=self.in_trainlabels().path,testlabels=self.in_testlabels().path,testdocs=self.in_testdocs().path,**kwargs)


class ClassifyAppendTask(Task):

    in_train = InputSlot()
    in_train_append = InputSlot()
    in_test = InputSlot()
    in_test_append = InputSlot()
    in_trainlabels = InputSlot()

    bow_as_feature = BoolParameter()
 
    append_parameters = Parameter()
    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()
    featurize_parameters = Parameter()
    preprocess_parameters = Parameter()

    def out_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.'.join(self.in_test().path.split('.')[-2:]) if (self.in_test().path[-3:] == 'npz' or self.in_test().path[-7:-4] == 'tok') else '.' + self.in_test().path.split('.')[-1], addextension='.bow.combined.predictions.txt' if self.bow_as_feature else '.combined.predictions.txt')
    
    def run(self):

        if self.complete(): # necessary as it will not complete otherwise
            return True

        kwargs = quoll_helpers.decode_task_input(['append','ga','classify','vectorize','featurize','preprocess'],[self.append_parameters,self.ga_parameters,self.classify_parameters,self.vectorize_parameters,self.featurize_parameters,self.preprocess_parameters])
        yield ClassifyAppend(train=self.in_train().path,train_append=self.in_train_append().path,trainlabels=self.in_trainlabels().path,test=self.in_test().path,test_append=self.in_test_append().path,**kwargs)


class TrainAppendTask(Task):

    in_train = InputSlot()
    in_train_append = InputSlot()
    in_trainlabels = InputSlot()
    in_traindocs = InputSlot()
    
    bow_as_feature = BoolParameter()
 
    append_parameters = Parameter()
    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()
    featurize_parameters = Parameter()
    preprocess_parameters = Parameter()
    
    def out_model(self):
        return self.outputfrominput(inputformat='train', stripextension='.'.join(self.in_train().path.split('.')[-2:]) if (self.in_train().path[-3:] == 'npz' or self.in_train().path[-7:-4] == 'tok') else '.' + self.in_train().path.split('.')[-1], addextension='.bow.combined.model.pkl' if self.bow_as_feature else '.combined.model.pkl')
    
    def run(self):
        
        if self.complete(): # necessary as it will not complete otherwise
            return True

        kwargs = quoll_helpers.decode_task_input(['append','ga','classify','vectorize','featurize','preprocess'],[self.append_parameters,self.ga_parameters,self.classify_parameters,self.vectorize_parameters,self.featurize_parameters,self.preprocess_parameters])
        yield ClassifyAppend(train=self.in_train().path,train_append=self.in_train_append().path,trainlabels=self.in_trainlabels().path,traindocs=self.in_traindocs().path,**kwargs)


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
    testlabels = Parameter(default = 'xxx.xxx')
    docs = Parameter(default = 'xxx.xxx') # all docs for nfold cv, test docs for train and test

    # append parameters
    bow_as_feature = BoolParameter() # to combine bow as separate classification with other features, only relevant in case of train_append
    bow_classifier = Parameter(default='naive_bayes')
    bow_nfolds = Parameter(default=5)
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
    ensemble = Parameter(default=False)
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

    perceptron_alpha = Parameter(default='1.0')

    tree_class_weight = Parameter(default=False)

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
                InputFormat(self, format_id='train',extension='.vectors.npz',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.csv',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.features.npz',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.tok.txt',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.tok.txtdir',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.frog.json',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.frog.jsondir',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.txtdir',inputparameter='train'),
                InputFormat(self, format_id='docs_train',extension='.txt',inputparameter='train')
                ),
                (
                InputFormat(self, format_id='train_append',extension='.vectors.npz',inputparameter='train_append'),
                InputFormat(self, format_id='train_append',extension='.csv',inputparameter='train_append'),
                ),
                (
                InputFormat(self, format_id='test',extension='.predictions.txt',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.vectors.npz',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.csv',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.features.npz',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.tok.txt',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.tok.txtdir',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.frog.json',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.frog.jsondir',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.txtdir',inputparameter='test'),
                InputFormat(self, format_id='docs_test',extension='.txt',inputparameter='test')
                ),
                (
                InputFormat(self, format_id='test_append',extension='.vectors.npz',inputparameter='test_append'),
                InputFormat(self, format_id='test_append',extension='.csv',inputparameter='test_append')
                ),
                (
                InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels')
                ),
                (
                InputFormat(self, format_id='labels_test',extension='.labels',inputparameter='testlabels')
                ),
                (
                InputFormat(self, format_id='docs',extension='.txt',inputparameter='docs')
                )
            ]
            )).T.reshape(-1,7)]

    def setup(self, workflow, input_feeds):
        
        task_args = quoll_helpers.prepare_task_input(['preprocess','featurize','vectorize','classify','ga','append','validate'],workflow.param_kwargs)

        ######################
        ### Training phase ###
        ######################
        
        trainlabels = input_feeds['labels_train']

        if 'train_append' in input_feeds.keys():
            trainvectors_append = input_feeds['train_append']
            train_append = True
        else:
            train_append = False

        if 'train' in input_feeds.keys():
            train = input_feeds['train']
            docs = False
        elif 'docs_train' in input_feeds.keys():
            train = input_feeds['docs_train']
            traindocs = input_feeds['docs_train']
            docs = True
        else:
            print('No train data inserted, exiting program...')
            exit()

        if not 'test' in input_feeds.keys() and not 'docs_test' in input_feeds.keys(): # only train input --> nfold-cv

        ######################
        ### Nfold CV #########
        ######################

            if not docs:
                if 'docs' in input_feeds.keys():
                    traindocs = input_feeds['docs']
                else:
                    print('No train documents inserted, exiting program...')
                    exit()

            if train_append:
                validator = workflow.new_task('validate_append', ValidateAppendTask, autopass=False,
                    preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],
                    classify_parameters=task_args['classify'],ga_parameters=task_args['ga'],validate_parameters=task_args['validate'],append_parameters=task_args['append']                   
                )
                validator.in_instances_append = trainvectors_append
            elif self.ensemble:
                validator = workflow.new_task('validate_ensemble', ValidateEnsembleTask, autopass=False,
                    preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],
                    classify_parameters=task_args['classify'],ga_parameters=task_args['ga'],validate_parameters=task_args['validate']
                )             
            else:
                validator = workflow.new_task('validate', ValidateTask, autopass=True,
                        classifier=self.classifier,n=self.n,preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],classify_parameters=task_args['classify'],ga_parameters=task_args['ga'],validate_parameters=task_args['validate']  
                )
            
            validator.in_instances = train
            validator.in_labels = trainlabels
            validator.in_docs = traindocs

            # also train a model on all data
            if train_append:
                foldtrainer = workflow.new_task('train_append',TrainAppendTask,autopass=True,
                    preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],classify_parameters=task_args['classify'],ga_parameters=task_args['ga'],append_parameters=task_args['append']
                ) 
                foldtrainer.in_train_append = trainvectors_append
            elif self.ensemble:
                foldtrainer = workflow.new_task('train_ensemble',EnsembleTrain,autopass=True,
                    preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],classify_parameters=task_args['classify'],ga_parameters=task_args['ga'],linear_raw=self.linear_raw
                ) 
            else:

                foldtrainer = workflow.new_task('train',TrainTask,autopass=True,
                    preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],classify_parameters=task_args['classify'],ga_parameters=task_args['ga']
                )

            foldtrainer.in_trainlabels = trainlabels
            foldtrainer.in_train = train
            foldtrainer.in_traindocs = traindocs

            return foldtrainer, validator

        else: 

        ###############################
        ### Train, test and report ####
        ###############################

            if 'test_append' in input_feeds.keys():
                testvectors_append = input_feeds['test_append']
                test_append = True
            else:
                test_append = False

            if 'test' in input_feeds.keys():
                test = input_feeds['test']
                if 'docs' in input_feeds.keys():
                    testdocs = input_feeds['docs']
                else:
                    print('No test docs inserted, exiting program...')
                    exit()
            elif 'docs_test' in input_feeds.keys():
                test = input_feeds['docs_test']
                testdocs = input_feeds['docs_test']

            if 'labels_test' in input_feeds.keys():
                testlabels = input_feeds['labels_test']
                testlabels_true = True
            else:
                testlabels = test
                testlabels_true = False

            if test_append:

                if not train_append:
                    print('test_append inserted without train_append, exiting program')
                    exit()

                append_classifier = workflow.new_task('classify_append',ClassifyAppendTask,autopass=True,
                    bow_as_feature=self.bow_as_feature,preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],classify_parameters=task_args['classify'],ga_parameters=task_args['ga'],append_parameters=task_args['append']
                )
                append_classifier.in_train = train
                append_classifier.in_train_append = trainvectors_append
                append_classifier.in_trainlabels = trainlabels
                append_classifier.in_test = test
                append_classifier.in_test_append = testvectors_append

                reporter = workflow.new_task('report_append', ReportTask, autopass=True, 
                    testlabels_true=testlabels_true,preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],classify_parameters=task_args['classify'],ga_parameters=task_args['ga']                   
                )
                reporter.in_train = train
                reporter.in_test = append_classifier.out_predictions
                reporter.in_trainlabels = trainlabels
                reporter.in_testlabels = testlabels
                reporter.in_testdocs = testdocs

            elif self.ensemble:
                ensemble_classifier = workflow.new_task('ensemble',EnsembleTrainTest, autopass=True,
                    preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],classify_parameters=task_args['classify'],ga_parameters=task_args['ga'],linear_raw=self.linear_raw
                )
                ensemble_classifier.in_train = train
                ensemble_classifier.in_trainlabels = trainlabels
                ensemble_classifier.in_test = test

                reporter = workflow.new_task('report_append', ReportTask, autopass=True, 
                    testlabels_true=testlabels_true,preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],classify_parameters=task_args['classify'],ga_parameters=task_args['ga']                   
                )
                reporter.in_train = ensemble_classifier.out_train
                reporter.in_test = ensemble_classifier.out_predictions
                reporter.in_trainlabels = ensemble_classifier.out_trainlabels
                reporter.in_testlabels = testlabels
                reporter.in_testdocs = testdocs                

            else:

                reporter = workflow.new_task('report', ReportTask, autopass=True, 
                    testlabels_true=testlabels_true,preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize'],classify_parameters=task_args['classify'],ga_parameters=task_args['ga']
                )
                reporter.in_train = train
                reporter.in_test = test
                reporter.in_trainlabels = trainlabels
                reporter.in_testlabels = testlabels
                reporter.in_testdocs = testdocs

            return reporter
