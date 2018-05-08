
from luiginlp.engine import Task, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, IntParameter, BoolParameter, FloatParameter
import numpy
from scipy import sparse
from collections import defaultdict

from quoll.classification_pipeline.functions import nfold_cv_functions, linewriter, docreader

from quoll.classification_pipeline.modules.vectorize import VectorizeCsv


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
    
    nb_alpha = FloatParameter(default=1.0)
    nb_fit_prior = BoolParameter()
    
    svm_c = FloatParameter(default=1.0)
    svm_kernel = Parameter(default='linear')
    svm_gamma = FloatParameter(default=0.1)
    svm_degree = IntParameter(default=1)
    svm_class_weight = Parameter(default='balanced')
    
    # vectorizer parameters
    weight = Parameter(default = 'frequency') # options: frequency, binary, tfidf
    prune = IntParameter(default = 5000) # after ranking the topfeatures in the training set, based on frequency or idf weighting
    balance = BoolParameter()
    
    def out_exp(self):
        return self.outputfrominput(inputformat='bins', stripextension='.bins.csv', addextension= 
            '.labels_' + self.in_labels().path.split('.')[-2] + self.classifier + 
            '.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.balance else 
            '.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz')

        
    def run(self):

        # make experiment directory
        self.setup_output_dir(self.out_exp().path)

        # for each fold
        for fold in range(self.n):
            yield RunFold(
                directory=self.out_exp().path, instances=self.in_instances().path, labels=self.in_labels().path, bins=self.in_bins().path, docs=self.in_docs().path, 
                i=fold, 
                classifier=self.classifier, ordinal=self.ordinal, jobs=self.jobs, iterations=self.iterations,
                nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
                svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight
            )                

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
    
    nb_alpha = FloatParameter()
    nb_fit_prior = BoolParameter()
    
    svm_c = FloatParameter()
    svm_kernel = Parameter()
    svm_gamma = FloatParameter()
    svm_degree = IntParameter()
    svm_class_weight = Parameter()
    
    # vectorizer parameters
    weight = Parameter() # options: frequency, binary, tfidf
    prune = IntParameter() # after ranking the topfeatures in the training set, based on frequency or idf weighting
    balance = BoolParameter()

    def in_vocabulary(self):
        return self.outputfrominput(inputformat='instances', stripextension='.' + self.in_instances().task.extension, addextension='.vocabulary.txt')   

    def out_fold(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i))    

    def out_train(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.' + self.in_instances().task.extension)        

    def out_test(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.' + self.in_instances().task.extension)

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.labels')

    def out_testlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.labels')

    def out_trainvocabulary(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.labels')

    def out_testvocabulary(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.labels')

    def out_testlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.labels')

    def out_testdocs(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.docs.txt')

    def run(self):

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
            train=self.out_train().path, trainlabels=self.out_trainlabels().path, test=self.out_test().path, testlabels=self.out_testlabels().path, docs=self.out_testdocs().path, 
            weight=self.weight, prune=self.prune, balance=self.balance, 
            classifier=self.classifier, ordinal=self.ordinal, jobs=self.jobs, iterations=self.iterations,
            nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
            svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight
        )

################################################################################
###Fold Wrapper
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
    
    nb_alpha = FloatParameter(default=1.0)
    nb_fit_prior = BoolParameter()
    
    svm_c = FloatParameter(default=1.0)
    svm_kernel = Parameter(default='linear')
    svm_gamma = FloatParameter(default=0.1)
    svm_degree = IntParameter(default=1)
    svm_class_weight = Parameter(default='balanced')

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
        ) ]
 
    def setup(self, workflow, input_feeds):

        run_fold = workflow.new_task(
            'run_fold', Fold, autopass=False, 
            i=self.i, 
            weight=self.weight, prune=self.prune, balance=self.balance, 
            classifier=self.classifier, ordinal=self.ordinal, jobs=self.jobs, iterations=self.iterations,
            nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
            svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight
        )
        run_fold.in_directory = input_feeds['directory']
        run_fold.in_instances = input_feeds['instances']
        run_fold.in_labels = input_feeds['labels']
        run_fold.in_docs = input_feeds['docs']
        run_fold.in_bins = input_feeds['bins']   

        return run_fold

#################################################################
### Component ###################################################
#################################################################

@registercomponent
class RunFolds(WorkflowComponent):

    instances = Parameter() 
    labels = Parameter()
    docs = Parameter()

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
    
    nb_alpha = FloatParameter(default=1.0)
    nb_fit_prior = BoolParameter()
    
    svm_c = FloatParameter(default=1.0)
    svm_kernel = Parameter(default='linear')
    svm_gamma = FloatParameter(default=0.1)
    svm_degree = IntParameter(default=1)
    svm_class_weight = Parameter(default='balanced')
    
    # vectorizer parameters
    weight = Parameter(default = 'frequency') # options: frequency, binary, tfidf
    prune = IntParameter(default = 5000) # after ranking the topfeatures in the training set, based on frequency or idf weighting
    balance = BoolParameter()
   
    def accepts(self):

        return [tuple(x) for x in numpy.array(numpy.meshgrid(*
            [
                (
                InputFormat(self, format_id='instances_csv',extension='.csv',inputparameter='instances'),
                InputFormat(self, format_id='instances',extension='.vectors.npz',inputparameter='instances'),                
                InputFormat(self, format_id='instances',extension='.features.npz',inputparameter='instances'),
                ),
                (
                InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels')
                ),
                (
                InputFormat(self, format_id='docs',extension='.txt',inputparameter='docs')
                )
            ]
            )).T.reshape(-1,3)]

    def setup(self, workflow, input_feeds):

        bin_maker = workflow.new_task('make_bins', MakeBins, autopass=True, n=self.n, teststart=self.teststart, testend=self.testend)
        bin_maker.in_labels = input_feeds['labels']

        if 'instances_csv' in input_feeds.keys():
            testvectorizer = workflow.new_task('vectorizer_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter)
            testvectorizer.in_csv = input_feeds['instances_csv']

            instances = testvectorizer.out_vectors

        else:
            instances = inputfeeds['instances']

        fold_runner = workflow.new_task(
            'nfold_cv', Folds, autopass=True, 
            n=self.n, 
            weight=self.weight, prune=self.prune, balance=self.balance, 
            classifier=self.classifier, ordinal=self.ordinal, jobs=self.jobs, iterations=self.iterations,
            nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
            svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight
        )
        fold_runner.in_bins = bin_maker.out_bins
        fold_runner.in_instances = instances
        fold_runner.in_labels = input_feeds['labels']
        fold_runner.in_docs = input_feeds['docs']            

        return fold_runner
