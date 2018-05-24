
import numpy
from scipy import sparse
import glob
from collections import defaultdict

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter, FloatParameter

from quoll.classification_pipeline.modules.report import Report

from quoll.classification_pipeline.functions import reporter, nfold_cv_functions, linewriter, docreader

#################################################################
### Tasks #######################################################
#################################################################

class MakeBins(Task):

    in_labels = InputSlot()

    n = IntParameter()
    steps = IntParameter(default = 1)
    teststart = IntParameter(default=0)
    testend = IntParameter(default=-1)

    def out_bins(self):
        return self.outputfrominput(inputformat='labels', stripextension='.labels', addextension='.' + str(self.n) + 'folds.bins.csv')
        
    def run(self):

        # open labels
        with open(self.in_labels().path,'r',encoding='utf-8') as infile:
            labels = numpy.array(infile.read().strip().split('\n'))

        # select rows per fold based on shape of the features
        if self.testend == -1:
            num_instances = len(labels)
        else:
            num_instances = self.testend
        fold_indices = nfold_cv_functions.return_fold_indices(num_instances,self.n,self.steps,self.teststart)        
        
        # write indices of bins to file
        lw = linewriter.Linewriter(fold_indices)
        lw.write_csv(self.out_bins().path)


class Fold(Task):

    in_directory = InputSlot()
    in_instances = InputSlot()
    in_labels = InputSlot()
    in_docs = InputSlot()
    in_bins = InputSlot()
    
    # fold parameters
    i = IntParameter()

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

    def in_vocabulary(self):
        return self.outputfrominput(inputformat='instances', stripextension='.' + '.'.join(self.in_instances().path.split('.')[-2:]), addextension='.vocabulary.txt')   

    def out_fold(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i))    

    def out_train(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.' + '.'.join(self.in_instances().path.split('.')[-2:]))        

    def out_test(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.' + '.'.join(self.in_instances().path.split('.')[-2:]))

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.labels')

    def out_testlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.labels')

    def out_trainvocabulary(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.vocabulary.txt' if '.'.join(self.in_instances().path.split('.')[-2:]) == 'features.npz' else '.exp/fold' + str(self.i) + '/train.featureselection.txt')

    def out_testvocabulary(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.vocabulary.txt')

    def out_testlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.labels')

    def out_testdocs(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.docs.txt')

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

        # if applicable, set fixed train indices
        fixed_train_indices = []
        if bin_min > 0:
            fixed_train_indices.extend(list(range(bin_min)))
        if (bin_max+1) < instances.shape[0]:
            fixed_train_indices.extend(list(range(bin_max+1,instances.shape[0])))

        # load vocabulary
        with open(self.in_vocabulary().path,'r',encoding='utf-8') as infile:
            vocabulary = infile.read().strip().split('\n')

        # open labels
        with open(self.in_labels().path,'r',encoding='utf-8') as infile:
            labels = numpy.array(infile.read().strip().split('\n'))

        # open documents
        with open(self.in_docs().path,'r',encoding='utf-8') as infile:
            documents = numpy.array(infile.read().strip().split('\n'))

        # set training and test data
        train_instances = sparse.vstack([instances[indices,:] for j,indices in enumerate(bins) if j != self.i] + [instances[fixed_train_indices,:]])
        train_labels = numpy.concatenate([labels[indices] for j,indices in enumerate(bins) if j != self.i] + [labels[fixed_train_indices]])
        train_documents = numpy.concatenate([documents[indices] for j,indices in enumerate(bins) if j != self.i] + [documents[fixed_train_indices]])
        test_instances = instances[bins[self.i]]
        test_labels = labels[bins[self.i]]
        test_documents = documents[bins[self.i]]

        # write experiment data to files in fold directory
        numpy.savez(self.out_train().path, data=train_instances.data, indices=train_instances.indices, indptr=train_instances.indptr, shape=train_instances.shape)
        numpy.savez(self.out_test().path, data=test_instances.data, indices=test_instances.indices, indptr=test_instances.indptr, shape=test_instances.shape)
        with open(self.out_trainvocabulary().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(vocabulary))
        with open(self.out_testvocabulary().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(vocabulary))
        with open(self.out_trainlabels().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(train_labels))
        with open(self.out_testlabels().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_labels))
        with open(self.out_testdocs().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_documents))

        print('Running experiment for fold',self.i)

        yield Report(
            train=self.out_train().path, trainlabels=self.out_trainlabels().path, test=self.out_test().path, testlabels=self.out_testlabels().path, testdocs=self.out_testdocs().path, 
            weight=self.weight, prune=self.prune, balance=self.balance, 
            classifier=self.classifier, ordinal=self.ordinal, jobs=self.jobs, iterations=self.iterations,
            nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
            svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
            lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter
        )

class Folds(Task):

    in_bins = InputSlot()
    in_instances = InputSlot()
    in_labels = InputSlot()
    in_docs = InputSlot()

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
    
    def out_exp(self):
        return self.outputfrominput(inputformat='instances', stripextension='.' + '.'.join(self.in_instances().path.split('.')[-2:]), addextension='.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.labels_' + self.in_labels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.exp' if self.balance and '.'.join(self.in_instances().path.split('.')[-2:]) == 'features.npz' else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.labels_' + self.in_labels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.exp' if '.'.join(self.in_instances().path.split('.')[-2:]) == 'features.npz' else '.labels_' + self.in_labels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.exp')
                                    
    def run(self):

        # make experiment directory
        self.setup_output_dir(self.out_exp().path)

        # for each fold
        for fold in range(self.n):
            yield RunFold(
                directory=self.out_exp().path, instances=self.in_instances().path, labels=self.in_labels().path, bins=self.in_bins().path, docs=self.in_docs().path, 
                i=fold, 
                weight=self.weight, prune=self.prune, balance=self.balance, 
                classifier=self.classifier, ordinal=self.ordinal, jobs=self.jobs, iterations=self.iterations,
                nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
                svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
                lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter
            )


################################################################################
### Components #################################################################
################################################################################

@registercomponent
class RunFold(WorkflowComponent):

    directory = Parameter()
    instances = Parameter()
    labels = Parameter()
    docs = Parameter()
    bins = Parameter()

    # fold-parameters
    i = IntParameter()

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
    
    def accepts(self):
        return [ ( 
            InputFormat(self,format_id='directory',extension='.exp',inputparameter='directory'), 
            InputFormat(self,format_id='instances',extension='.features.npz',inputparameter='instances'), 
            InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), 
            InputFormat(self,format_id='docs',extension='.txt',inputparameter='docs'),
            InputFormat(self,format_id='bins',extension='.bins.csv',inputparameter='bins') 
        ),
        (
            InputFormat(self,format_id='directory',extension='.exp',inputparameter='directory'), 
            InputFormat(self,format_id='instances',extension='.vectors.npz',inputparameter='instances'), 
            InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), 
            InputFormat(self,format_id='docs',extension='.txt',inputparameter='docs'),
            InputFormat(self,format_id='bins',extension='.bins.csv',inputparameter='bins') 
        ) ]
 
    def setup(self, workflow, input_feeds):

        run_fold = workflow.new_task(
            'run_fold', Fold, autopass=False, 
            i=self.i, 
            weight=self.weight, prune=self.prune, balance=self.balance, 
            classifier=self.classifier, ordinal=self.ordinal, jobs=self.jobs, iterations=self.iterations,
            nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
            svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
            lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter
        )
        run_fold.in_directory = input_feeds['directory']
        run_fold.in_instances = input_feeds['instances']
        run_fold.in_labels = input_feeds['labels']
        run_fold.in_docs = input_feeds['docs']
        run_fold.in_bins = input_feeds['bins']   

        return run_fold


@registercomponent
class Validate(WorkflowComponent):

    instances = Parameter()
    labels = Parameter()
    docs = Parameter()

    # fold-parameters
    n = IntParameter()

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
    
    def accepts(self):
        return [ ( 
            InputFormat(self,format_id='instances',extension='.features.npz',inputparameter='instances'), 
            InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), 
            InputFormat(self,format_id='docs',extension='.txt',inputparameter='docs'),
        ),
        (
            InputFormat(self,format_id='instances',extension='.vectors.npz',inputparameter='instances'), 
            InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), 
            InputFormat(self,format_id='docs',extension='.txt',inputparameter='docs'),
        ) ]
 
    def setup(self, workflow, input_feeds):

        bin_maker = workflow.new_task('make_bins', MakeBins, autopass=True, n=5)
        bin_maker.in_labels = input_feeds['labels']

        foldrunner = workflow.new_task(
            'foldrunner', Folds, autopass=False, 
            n=self.n, 
            weight=self.weight, prune=self.prune, balance=self.balance, 
            classifier=self.classifier, ordinal=self.ordinal, jobs=self.jobs, iterations=self.iterations,
            nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
            svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
            lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter
        )
        foldrunner.in_bins = bin_maker.out_bins
        foldrunner.in_instances = input_feeds['instances']
        foldrunner.in_labels = input_feeds['labels']
        foldrunner.in_docs = input_feeds['docs']

        return foldrunner
