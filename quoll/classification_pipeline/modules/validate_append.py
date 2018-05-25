
import numpy
from scipy import sparse

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter, FloatParameter

from quoll.classification_pipeline.modules.validate import MakeBins
from quoll.classification_pipeline.modules.report import Report, ReportFolds, ReportPerformance
from quoll.classification_pipeline.modules.classify_append import ClassifyAppend
from quoll.classification_pipeline.modules.vectorize import VectorizeCsv, FeaturizeTask

from quoll.classification_pipeline.functions import docreader

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
    
    # fold parameters
    i = IntParameter()

    # append parameters
    bow_as_feature = BoolParameter() # to combine bow as separate classification with other features, only relevant in case of train_append
    bow_classifier = Parameter()

    # classifier parameters
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
    
    # vectorizer parameters
    weight = Parameter() # options: frequency, binary, tfidf
    prune = IntParameter() # after ranking the topfeatures in the training set, based on frequency or idf weighting
    balance = BoolParameter()
    scale = BoolParameter()

    def in_vocabulary(self):
        return self.outputfrominput(inputformat='instances', stripextension='.' + '.'.join(self.in_instances().path.split('.')[-2:]), addextension='.vocabulary.txt' if '.'.join(self.in_instances().path.split('.')[-2:]) == 'features.npz' else '.featureselection.txt')   

    def in_vocabulary_append(self):
        return self.outputfrominput(inputformat='instances_append', stripextension='.' + '.'.join(self.in_instances_append().path.split('.')[-2:]), addextension='.featureselection.txt')   

    def out_fold(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i))    

    def out_train(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.' + '.'.join(self.in_instances().path.split('.')[-2:]))        

    def out_train_append(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.append.vectors.npz')            

    def out_test(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.' + '.'.join(self.in_instances().path.split('.')[-2:]))

    def out_test_append(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.append.vectors.npz')

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.labels')

    def out_testlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.labels')

    def out_trainvocabulary(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.vocabulary.txt' if '.'.join(self.in_instances().path.split('.')[-2:]) == 'features.npz' else '.exp/fold' + str(self.i) + '/train.featureselection.txt')

    def out_trainvocabulary_append(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.append.featureselection.txt')

    def out_testvocabulary(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.vocabulary.txt')

    def out_testvocabulary_append(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.append.featureselection.txt')

    def out_testlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.labels')

    def out_traindocs(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.docs.txt')

    def out_testdocs(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.docs.txt')

    def out_predictions(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.labels_train.' + self.bow_classifier + '.bow.append.scaled.labels_train.' + self.classifier + '.predictions.txt' if self.bow_as_feature and self.balance and self.scale else '.exp/fold' + str(self.i) + '/test.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.labels_train.' + self.bow_classifier + '.bow.append.labels_train.' + self.classifier + '.predictions.txt' if self.bow_as_feature and self.balance else '.exp/fold' + str(self.i) + '/test.weight_' + self.weight + '.prune_' + str(self.prune) + '.labels_train.' + self.bow_classifier + '.bow.append.scaled.labels_train.' + self.classifier + '.predictions.txt' if self.bow_as_feature and self.scale else '.exp/fold' + str(self.i) + '/test.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.labels_train.' + self.bow_classifier + '.append.scaled.labels_train.' + self.classifier + '.predictions.txt' if self.balance and self.scale else '.exp/fold' + str(self.i) + '/test.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.append.labels_train.' + self.classifier + '.predictions.txt' if self.balance else '.exp/fold' + str(self.i) + '/test.weight_' + self.weight + '.prune_' + str(self.prune) + '.labels_train.' + self.bow_classifier + '.bow.append.labels_train.' + self.classifier + '.predictions.txt' if self.bow_as_feature else '.exp/fold' + str(self.i) + '/test.weight_' + self.weight + '.prune_' + str(self.prune) + '.labels_train.' + self.bow_classifier + '.append.scaled.labels_train.' + self.classifier + '.predictions.txt' if self.scale else '.exp/fold' + str(self.i) + '/test.weight_' + self.weight + '.prune_' + str(self.prune) + '.append.labels_train.' + self.classifier + '.predictions.txt')

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

        print('Running experiment for fold',self.i)

        yield ClassifyAppend(
            traininstances=self.out_train().path, traininstances_append=self.out_train_append().path, trainlabels=self.out_trainlabels().path, 
            testinstances=self.out_test().path, testinstances_append=self.out_test_append().path, 
            traindocs=self.out_traindocs().path, 
            weight=self.weight, prune=self.prune, balance=self.balance, scale=self.scale,
            bow_as_feature=self.bow_as_feature, bow_classifier=self.bow_classifier,
            classifier=self.classifier, ordinal=self.ordinal, jobs=self.jobs, iterations=self.iterations,
            nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
            svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
            lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter
        )

class FoldsAppend(Task):

    in_bins = InputSlot()
    in_instances = InputSlot()
    in_instances_append = InputSlot()
    in_labels = InputSlot()
    in_docs = InputSlot()

    # nfold-cv parameters
    n = IntParameter(default=10)
    steps = IntParameter(default=1) # useful to increase if close-by instances, for example sets of 2, are dependent
    teststart = IntParameter(default=0) # if part of the instances are only used for training and not for testing (for example because they are less reliable), specify the test indices via teststart and testend
    testend = IntParameter(default=-1)

    # append parameters
    bow_as_feature = BoolParameter() # to combine bow as separate classification with other features, only relevant in case of train_append
    bow_classifier = Parameter()

    # classifier parameters
    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter()
    jobs = IntParameter(default=1)
    iterations = IntParameter(default=10)
    
    nb_alpha = Parameter(default=1.0)
    nb_fit_prior = BoolParameter()
    
    svm_c = Parameter(default=1.0)
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
    scale = BoolParameter()
    
    def out_exp(self):
        return self.outputfrominput(inputformat='instances', stripextension='.' + '.'.join(self.in_instances().path.split('.')[-2:]), addextension='.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.bow.' + self.in_instances_append().path.split('.')[-3] + '.labels_' + self.in_labels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.exp' if self.balance and self.bow_as_feature and '.'.join(self.in_instances().path.split('.')[-2:]) == 'features.npz' else '.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.' + self.in_instances_append().path.split('.')[-3] + '.labels_' + self.in_labels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.exp' if self.balance and '.'.join(self.in_instances().path.split('.')[-2:]) == 'features.npz' else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.bow.' + self.in_instances_append().path.split('.')[-3] + '.labels_' + self.in_labels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.exp' if self.bow_as_feature and '.'.join(self.in_instances().path.split('.')[-2:]) == 'features.npz' else '.bow.' + self.in_instances_append().path.split('.')[-3] + '.labels_' + self.in_labels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.exp' if self.bow_as_feature else '.labels_' + self.in_labels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.exp')
                                    
    def run(self):

        # make experiment directory
        self.setup_output_dir(self.out_exp().path)

        # for each fold
        for fold in range(self.n):
            yield RunFoldAppend(
                directory=self.out_exp().path, instances=self.in_instances().path, instances_append=self.in_instances_append().path, labels=self.in_labels().path, bins=self.in_bins().path, docs=self.in_docs().path, 
                i=fold,
                weight=self.weight, prune=self.prune, balance=self.balance, scale=self.scale,
                bow_as_feature=self.bow_as_feature, bow_classifier=self.bow_classifier,
                classifier=self.classifier, ordinal=self.ordinal, jobs=self.jobs, iterations=self.iterations,
                nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
                svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
                lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter
            )                


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

    # append parameters
    bow_as_feature = BoolParameter() # to combine bow as separate classification with other features, only relevant in case of train_append
    bow_classifier = Parameter(default='naive_bayes')

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
    scale = BoolParameter()
    
    def accepts(self):
        return [ ( 
            InputFormat(self,format_id='directory',extension='.exp',inputparameter='directory'), 
            InputFormat(self,format_id='instances',extension='.features.npz',inputparameter='instances'),
            InputFormat(self,format_id='instances_append',extension='.vectors.npz',inputparameter='instances_append'), 
            InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), 
            InputFormat(self,format_id='docs',extension='.txt',inputparameter='docs'),
            InputFormat(self,format_id='bins',extension='.bins.csv',inputparameter='bins') 
        ),
        (
            InputFormat(self,format_id='directory',extension='.exp',inputparameter='directory'), 
            InputFormat(self,format_id='instances',extension='.vectors.npz',inputparameter='instances'),
            InputFormat(self,format_id='instances_append',extension='.vectors.npz',inputparameter='instances_append'), 
            InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), 
            InputFormat(self,format_id='docs',extension='.txt',inputparameter='docs'),
            InputFormat(self,format_id='bins',extension='.bins.csv',inputparameter='bins') 
        ) ]
 
    def setup(self, workflow, input_feeds):

        fold_append_runner = workflow.new_task(
            'run_fold_append', FoldAppend, autopass=True, 
            i=self.i, 
            weight=self.weight, prune=self.prune, balance=self.balance, scale=self.scale,
            bow_as_feature=self.bow_as_feature, bow_classifier=self.bow_classifier,
            classifier=self.classifier, ordinal=self.ordinal, jobs=self.jobs, iterations=self.iterations,
            nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
            svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
            lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter
        )
        fold_append_runner.in_directory = input_feeds['directory']
        fold_append_runner.in_instances = input_feeds['instances']
        fold_append_runner.in_instances_append = input_feeds['instances_append']
        fold_append_runner.in_labels = input_feeds['labels']
        fold_append_runner.in_docs = input_feeds['docs']
        fold_append_runner.in_bins = input_feeds['bins']

        fold_append_reporter = workflow.new_task('report_fold_append', ReportPerformance, autopass=True, ordinal=self.ordinal)
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

    # fold-parameters
    n = IntParameter(default=10)

    # append parameters
    bow_as_feature = BoolParameter() # to combine bow as separate classification with other features, only relevant in case of train_append
    bow_classifier = Parameter(default='naive_bayes')

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
                InputFormat(self, format_id='vectorized',extension='.vectors.npz',inputparameter='instances'),
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

        if 'docs_instances' in input_feeds.keys():
            docs = input_feeds['docs_instances']
        else:
            docs = input_feeds['docs']

        if 'vectorized' in input_feeds.keys():
            instances = input_feeds['vectorized']
        elif 'featurized' in input_feeds.keys():
            instances = input_feeds['featurized']
        
        elif 'featurized_csv' in input_feeds.keys():
            vectorizer = workflow.new_task('vectorize_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter)
            vectorizer.in_csv = input_feeds['featurized_csv']
                
            instances = vectorizer.out_vectors

        else:
            if 'pre_featurized' in input_feeds.keys():
                pre_featurized = input_feeds['pre_featurized']
            else:
                pre_featurized = input_feeds['docs_instances']

            featurizer = workflow.new_task('featurize',FeaturizeTask,autopass=False,
                ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,
                minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,
                tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation
            )
            featurizer.in_pre_featurized = pre_featurized

            instances = featurizer.out_featurized

        if 'vectorized_append' in input_feeds.keys():
            instances_append = input_feeds['vectorized_append']
        elif 'featurized_csv_append' in input_feeds.keys():
            vectorizer_append = workflow.new_task('vectorize_csv_append',VectorizeCsv,autopass=True,delimiter=self.delimiter)
            vectorizer_append.in_csv = input_feeds['featurized_csv_append']
                
            instances_append = vectorizer_append.out_vectors

        bin_maker = workflow.new_task('make_bins', MakeBins, autopass=True, n=self.n)
        bin_maker.in_labels = input_feeds['labels']

        foldrunner_append = workflow.new_task(
            'foldrunner_append', FoldsAppend, autopass=False, 
            n=self.n, 
            weight=self.weight, prune=self.prune, balance=self.balance, scale=self.scale,
            bow_as_feature=self.bow_as_feature, bow_classifier=self.bow_classifier,
            classifier=self.classifier, ordinal=self.ordinal, jobs=self.jobs, iterations=self.iterations,
            nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
            svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
            lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter
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
