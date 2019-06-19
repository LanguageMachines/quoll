
import numpy
from scipy import sparse

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter, FloatParameter

from quoll.classification_pipeline.modules.validate import MakeBins
from quoll.classification_pipeline.modules.report import Report, ReportFolds, ReportPerformance
from quoll.classification_pipeline.modules.classify_append import ClassifyAppend
from quoll.classification_pipeline.modules.vectorize import TransformCsv, FeaturizeTask

from quoll.classification_pipeline.functions import docreader, quoll_helpers

#################################################################
### Tasks #######################################################
#################################################################

class FoldAppend(Task):

    in_directory = InputSlot()
    in_instances = InputSlot()
    in_instances_append = InputSlot()
    in_labels = InputSlot()
    in_docs = InputSlot()
    in_bins = InputSlot()
    
    i = IntParameter()
    linear_raw = BoolParameter()
    bow_as_feature = BoolParameter()
    
    append_parameters = Parameter()
    validate_parameters = Parameter()
    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()

    def in_vocabulary(self):
        return self.outputfrominput(inputformat='instances', stripextension='.' + '.'.join(self.in_instances().path.split('.')[-2:]), addextension='.vocabulary.txt' if '.'.join(self.in_instances().path.split('.')[-2:]) == 'features.npz' else '.featureselection.txt')   
    
    def in_vocabulary_append(self):
        return self.outputfrominput(inputformat='instances_append', stripextension='.' + '.'.join(self.in_instances_append().path.split('.')[-2:]), addextension='.vocabulary.txt' if '.'.join(self.in_instances_append().path.split('.')[-2:]) == 'features.npz' else '.featureselection.txt')   

    def in_nominal_labels(self):
        return self.outputfrominput(inputformat='labels', stripextension='.raw.labels' if self.linear_raw else '.labels', addextension='.labels')   

    def out_fold(self):
        return self.outputfrominput(inputformat='directory', stripextension='.nfoldcv', addextension='.nfoldcv/fold' + str(self.i+1))    

    def out_train(self):
        return self.outputfrominput(inputformat='directory', stripextension='.nfoldcv', addextension='.nfoldcv/fold' + str(self.i+1) + '/train.' + '.'.join(self.in_instances().path.split('.')[-2:]))        

    def out_train_append(self):
        return self.outputfrominput(inputformat='directory', stripextension='.nfoldcv', addextension='.nfoldcv/fold' + str(self.i+1) + '/train_append.' + '.'.join(self.in_instances_append().path.split('.')[-2:]))   

    def out_test(self):
        return self.outputfrominput(inputformat='directory', stripextension='.nfoldcv', addextension='.nfoldcv/fold' + str(self.i+1) + '/test.' + '.'.join(self.in_instances().path.split('.')[-2:]))         

    def out_test_append(self):
        return self.outputfrominput(inputformat='directory', stripextension='.nfoldcv', addextension='.nfoldcv/fold' + str(self.i+1) + '/test_append.' + '.'.join(self.in_instances_append().path.split('.')[-2:]))

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.nfoldcv', addextension='.nfoldcv/fold' + str(self.i+1) + '/train.raw.labels' if self.linear_raw else '.nfoldcv/fold' + str(self.i+1) + '/train.labels')

    def out_testlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.nfoldcv', addextension='.nfoldcv/fold' + str(self.i+1) + '/test.labels')

    def out_nominal_trainlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.nfoldcv', addextension='.nfoldcv/fold' + str(self.i+1) + '/train.labels')

    def out_trainvocabulary(self):
        return self.outputfrominput(inputformat='directory', stripextension='.nfoldcv', addextension='.nfoldcv/fold' + str(self.i+1) + '/train.vocabulary.txt' if '.'.join(self.in_instances().path.split('.')[-2:]) == 'features.npz' else '.nfoldcv/fold' + str(self.i+1) + '/train.featureselection.txt')

    def out_trainvocabulary_append(self):
        return self.outputfrominput(inputformat='directory', stripextension='.nfoldcv', addextension='.nfoldcv/fold' + str(self.i+1) + '/train_append.vocabulary.txt' if '.'.join(self.in_instances().path.split('.')[-2:]) == 'features.npz' else '.nfoldcv/fold' + str(self.i+1) + '/train_append.featureselection.txt')
    
    def out_testvocabulary(self):
        return self.outputfrominput(inputformat='directory', stripextension='.nfoldcv', addextension='.nfoldcv/fold' + str(self.i+1) + '/test.vocabulary.txt' if '.'.join(self.in_instances().path.split('.')[-2:]) == 'features.npz' else '.nfoldcv/fold' + str(self.i+1) + '/test.featureselection.txt')

    def out_testvocabulary_append(self):
        return self.outputfrominput(inputformat='directory', stripextension='.nfoldcv', addextension='.nfoldcv/fold' + str(self.i+1) + '/test_append.vocabulary.txt' if '.'.join(self.in_instances().path.split('.')[-2:]) == 'features.npz' else '.nfoldcv/fold' + str(self.i+1) + '/test_append.featureselection.txt')

    def out_traindocs(self):
        return self.outputfrominput(inputformat='directory', stripextension='.nfoldcv', addextension='.nfoldcv/fold' + str(self.i+1) + '/train.docs.txt')

    def out_testdocs(self):
        return self.outputfrominput(inputformat='directory', stripextension='.nfoldcv', addextension='.nfoldcv/fold' + str(self.i+1) + '/test.docs.txt')

    def out_predictions(self):
        return self.outputfrominput(inputformat='directory', stripextension='.nfoldcv', addextension='.nfoldcv/fold' + str(self.i+1) + '/test.predictions.combined.predictions.txt' if self.bow_as_feature else '.nfoldcv/fold' + str(self.i+1) + '/test.combined.predictions.txt')

    def run(self):
        
        if self.complete(): # needed as it will not complete otherwise
            return True
        
        # make fold directory
        self.setup_output_dir(self.out_fold().path)

        # open bin indices
        dr = docreader.Docreader()
        bins_str = dr.parse_csv(self.in_bins().path)
        bins = [[int(x) for x in bin] for bin in bins_str]
        bin_range = list(set(sum(bins,[])))
        bin_min = min(bin_range)
        bin_max = max(bin_range)

        # open instances
        loader = numpy.load(self.in_instances().path)
        instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # open instances append
        loader = numpy.load(self.in_instances_append().path)
        instances_append = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # if applicable, set fixed train indices
        fixed_train_indices = []
        if bin_min > 0:
            fixed_train_indices.extend(list(range(bin_min)))
        if (bin_max+1) < instances.shape[0]:
            fixed_train_indices.extend(list(range(bin_max+1,instances.shape[0])))

        # load vocabulary
        with open(self.in_vocabulary().path,'r',encoding='utf-8') as infile:
            vocabulary = infile.read().strip().split('\n')

        # load vocabulary append
        with open(self.in_vocabulary_append().path,'r',encoding='utf-8') as infile:
            vocabulary_append = infile.read().strip().split('\n')

        # open labels
        with open(self.in_labels().path,'r',encoding='utf-8') as infile:
            labels = numpy.array(infile.read().strip().split('\n'))

        # open documents
        with open(self.in_docs().path,'r',encoding='utf-8') as infile:
            documents = numpy.array(infile.read().strip().split('\n'))

        # set training and test data
        train_instances = sparse.vstack([instances[indices,:] for j,indices in enumerate(bins) if j != self.i] + [instances[fixed_train_indices,:]])
        train_instances_append = sparse.vstack([instances_append[indices,:] for j,indices in enumerate(bins) if j != self.i] + [instances_append[fixed_train_indices,:]])
        train_labels = numpy.concatenate([labels[indices] for j,indices in enumerate(bins) if j != self.i] + [labels[fixed_train_indices]])
        train_documents = numpy.concatenate([documents[indices] for j,indices in enumerate(bins) if j != self.i] + [documents[fixed_train_indices]])
        test_instances = instances[bins[self.i]]
        test_instances_append = instances_append[bins[self.i]]
        test_labels = labels[bins[self.i]]
        test_documents = documents[bins[self.i]]

        # set nominal labels and write to files
        if self.linear_raw:
            # open labels
            with open(self.in_nominal_labels().path,'r',encoding='utf-8') as infile:
                nominal_labels = numpy.array(infile.read().strip().split('\n'))        
            train_labels_nominal = numpy.concatenate([nominal_labels[indices] for j,indices in enumerate(bins) if j != self.i] + [nominal_labels[fixed_train_indices]])
            test_labels = nominal_labels[bins[self.i]]
            with open(self.out_nominal_trainlabels().path,'w',encoding='utf-8') as outfile:
                outfile.write('\n'.join(train_labels_nominal))

        # write experiment data to files in fold directory
        numpy.savez(self.out_train().path, data=train_instances.data, indices=train_instances.indices, indptr=train_instances.indptr, shape=train_instances.shape)
        numpy.savez(self.out_train_append().path, data=train_instances_append.data, indices=train_instances_append.indices, indptr=train_instances_append.indptr, shape=train_instances_append.shape)
        numpy.savez(self.out_test().path, data=test_instances.data, indices=test_instances.indices, indptr=test_instances.indptr, shape=test_instances.shape)
        numpy.savez(self.out_test_append().path, data=test_instances_append.data, indices=test_instances_append.indices, indptr=test_instances_append.indptr, shape=test_instances_append.shape)
        with open(self.out_trainvocabulary().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(vocabulary))
        with open(self.out_trainvocabulary_append().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(vocabulary_append))
        with open(self.out_testvocabulary().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(vocabulary))
        with open(self.out_testvocabulary_append().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(vocabulary_append))
        with open(self.out_trainlabels().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(train_labels))
        with open(self.out_testlabels().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_labels))
        with open(self.out_traindocs().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(train_documents))
        with open(self.out_testdocs().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_documents))

        print('Running experiment for fold',self.i+1)

        kwargs = quoll_helpers.decode_task_input(['ga','classify','vectorize','append'],[self.ga_parameters,self.classify_parameters,self.vectorize_parameters,self.append_parameters])
        yield ClassifyAppend(train=self.out_train().path,train_append=self.out_train_append().path,trainlabels=self.out_trainlabels().path,test=self.out_test().path,test_append=self.out_test_append().path,traindocs=self.out_traindocs().path,**kwargs) 


class FoldsAppend(Task):

    in_bins = InputSlot()
    in_instances = InputSlot()
    in_instances_append = InputSlot()
    in_labels = InputSlot()
    in_docs = InputSlot()

    n = IntParameter()
    
    append_parameters = Parameter()
    validate_parameters = Parameter()
    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()
    
    def out_exp(self):
        return self.outputfrominput(inputformat='instances', stripextension='.' + '.'.join(self.in_instances().path.split('.')[-2:]), addextension='.append.nfoldcv')
                                    
    def run(self):

        # make experiment directory
        self.setup_output_dir(self.out_exp().path)

        # for each fold
        for fold in range(self.n):
            yield RunFoldAppend(
                directory=self.out_exp().path, instances=self.in_instances().path, instances_append=self.in_instances_append().path, labels=self.in_labels().path, bins=self.in_bins().path, docs=self.in_docs().path, 
                i=fold,append_parameters=self.append_parameters,validate_parameters=self.validate_parameters,ga_parameters=self.ga_parameters,classify_parameters=self.classify_parameters,vectorize_parameters=self.vectorize_parameters
            )                

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
        return self.outputfrominput(inputformat='instances', stripextension='.'.join(self.in_instances().path.split('.')[-2:]) if (self.in_instances().path[-3:] == 'npz' or self.in_instances().path[-7:-4] == 'tok') else '.' + self.in_instances().path.split('.')[-1], addextension='.append.validated.report')
                                    
    def run(self):

        if self.complete(): # necessary as it will not complete otherwise
            return True

        kwargs = quoll_helpers.decode_task_input(['validate','append','ga','classify','vectorize','featurize','preprocess'],[self.validate_parameters,self.append_parameters,self.ga_parameters,self.classify_parameters,self.vectorize_parameters,self.featurize_parameters,self.preprocess_parameters])
        yield ValidateAppend(instances=self.in_instances().path,instances_append=self.in_instances_append().path,labels=self.in_labels().path,docs=self.in_docs().path,**kwargs)



################################################################################
### Components #################################################################
################################################################################

@registercomponent
class RunFoldAppend(WorkflowComponent):

    directory = Parameter()
    instances = Parameter()
    instances_append = Parameter()
    labels = Parameter()
    docs = Parameter()
    bins = Parameter()

    # fold-parameters
    i = IntParameter()

    append_parameters = Parameter()
    validate_parameters = Parameter()
    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()
    
    def accepts(self):
        return [ ( 
            InputFormat(self,format_id='directory',extension='.nfoldcv',inputparameter='directory'), 
            InputFormat(self,format_id='instances',extension='.features.npz',inputparameter='instances'),
            InputFormat(self,format_id='instances_append',extension='.features.npz',inputparameter='instances_append'), 
            InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), 
            InputFormat(self,format_id='docs',extension='.txt',inputparameter='docs'),
            InputFormat(self,format_id='bins',extension='.bins.csv',inputparameter='bins') 
        ) ]
 
    def setup(self, workflow, input_feeds):

        kwargs = quoll_helpers.decode_task_input(['classify','validate','append'],[self.classify_parameters,self.validate_parameters,self.append_parameters])

        fold_append_runner = workflow.new_task(
            'run_fold_append', FoldAppend, autopass=False, 
            i=self.i,linear_raw=kwargs['linear_raw'],bow_as_feature=kwargs['bow_as_feature'],validate_parameters=self.validate_parameters,ga_parameters=self.ga_parameters,classify_parameters=self.classify_parameters,vectorize_parameters=self.vectorize_parameters,append_parameters=self.append_parameters
        )
        fold_append_runner.in_directory = input_feeds['directory']
        fold_append_runner.in_instances = input_feeds['instances']
        fold_append_runner.in_instances_append = input_feeds['instances_append']
        fold_append_runner.in_labels = input_feeds['labels']
        fold_append_runner.in_docs = input_feeds['docs']
        fold_append_runner.in_bins = input_feeds['bins']

        fold_append_reporter = workflow.new_task('report_fold_append', ReportPerformance, autopass=True, ordinal=kwargs['ordinal'],teststart=kwargs['teststart'])
        fold_append_reporter.in_predictions = fold_append_runner.out_predictions
        fold_append_reporter.in_testlabels = fold_append_runner.out_testlabels
        fold_append_reporter.in_testdocuments = fold_append_runner.out_testdocs

        return fold_append_reporter

@registercomponent
class ValidateAppend(WorkflowComponent):

    instances = Parameter()
    instances_append = Parameter()
    labels = Parameter()
    docs = Parameter(default = 'xxx.xxx')

    # append parameters
    bow_as_feature = BoolParameter() # to combine bow as separate classification with other features, only relevant in case of train_append
    bow_classifier = Parameter(default='naive_bayes')
    bow_nfolds = Parameter(default=5)
    bow_include_labels = Parameter(default='all') # will give prediction probs as feature for each label by default, can specify particular labels (separated by a space) here, only applies when 'bow_prediction_probs' is chosen
    bow_prediction_probs = BoolParameter() # choose to add prediction probabilities

    # fold-parameters
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
    strip_punctuation = BoolParameter(default=False)

    def accepts(self):
        return [tuple(x) for x in numpy.array(numpy.meshgrid(*
            [
                (
                InputFormat(self, format_id='featurized',extension='.features.npz',inputparameter='instances'),
                InputFormat(self, format_id='featurized_csv',extension='.csv',inputparameter='instances'),
                InputFormat(self, format_id='pre_featurized',extension='.tok.txt',inputparameter='instances'),
                InputFormat(self, format_id='pre_featurized',extension='.tok.txtdir',inputparameter='instances'),
                InputFormat(self, format_id='pre_featurized',extension='.frog.json',inputparameter='instances'),
                InputFormat(self, format_id='pre_featurized',extension='.frog.jsondir',inputparameter='instances'),
                InputFormat(self, format_id='pre_featurized',extension='.txtdir',inputparameter='instances'),
                InputFormat(self, format_id='docs_instances',extension='.txt',inputparameter='instances'),
                ),
                (
                InputFormat(self, format_id='vectorized_append',extension='.vectors.npz',inputparameter='instances_append'),
                InputFormat(self, format_id='featurized_csv_append',extension='.csv',inputparameter='instances_append'),
                ),
                (
                InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels')
                ),
                (
                InputFormat(self, format_id='docs',extension='.txt',inputparameter='docs')
                )
            ]
            )).T.reshape(-1,4)]

 
    def setup(self, workflow, input_feeds):

        task_args = quoll_helpers.prepare_task_input(['preprocess','featurize','vectorize','classify','ga','validate','append'],workflow.param_kwargs)

        if 'docs_instances' in input_feeds.keys():
            docs = input_feeds['docs_instances']
        else:
            docs = input_feeds['docs']

        if 'featurized' in input_feeds.keys():
            instances = input_feeds['featurized']
        
        elif 'featurized_csv' in input_feeds.keys():
            csvtransformer = workflow.new_task('transformer_csv',TransformCsv,autopass=True,delimiter=self.delimiter)
            csvtransformer.in_csv = input_feeds['featurized_train_csv']
            instances = csvtransformer.out_features

        else:
            if 'pre_featurized' in input_feeds.keys():
                pre_featurized = input_feeds['pre_featurized']
            else:
                pre_featurized = input_feeds['docs_instances']

            featurizer = workflow.new_task('featurize',FeaturizeTask,autopass=True,preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'])
            featurizer.in_pre_featurized = pre_featurized

            instances = featurizer.out_featurized

        if 'vectorized_append' in input_feeds.keys():
            instances_append = input_feeds['vectorized_append']
        elif 'featurized_csv_append' in input_feeds.keys():
            vectorizer_append = workflow.new_task('vectorize_csv_append',TransformCsv,autopass=True,delimiter=self.delimiter)
            vectorizer_append.in_csv = input_feeds['featurized_csv_append']
                
            instances_append = vectorizer_append.out_features

        bin_maker = workflow.new_task('make_bins', MakeBins, autopass=True, n=self.n, steps=self.steps, teststart=self.teststart, testend=self.testend)
        bin_maker.in_labels = input_feeds['labels']

        foldrunner_append = workflow.new_task('foldrunner_append', FoldsAppend, autopass=True, 
            n=self.n,vectorize_parameters=task_args['vectorize'],classify_parameters=task_args['classify'],ga_parameters=task_args['ga'],validate_parameters=task_args['validate'],append_parameters=task_args['append']
        ) 
        foldrunner_append.in_bins = bin_maker.out_bins
        foldrunner_append.in_instances = instances
        foldrunner_append.in_instances_append = instances_append
        foldrunner_append.in_labels = input_feeds['labels']
        foldrunner_append.in_docs = docs

        foldreporter = workflow.new_task('report_folds', ReportFolds, autopass=True)
        foldreporter.in_exp = foldrunner_append.out_exp

        validator_append = workflow.new_task('report_performance',ReportPerformance,autopass=True,ordinal=self.ordinal)
        validator_append.in_predictions = foldreporter.out_predictions
        validator_append.in_testlabels = foldreporter.out_labels
        validator_append.in_testdocuments = foldreporter.out_docs

        return validator_append
