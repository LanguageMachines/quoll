
from luiginlp.engine import Task, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, IntParameter, BoolParameter
import numpy
from scipy import sparse
from collections import defaultdict

import quoll.classification_pipeline.functions.nfold_cv_functions as nfold_cv_functions
import quoll.classification_pipeline.functions.linewriter as linewriter
import quoll.classification_pipeline.functions.docreader as docreader

from quoll.classification_pipeline.modules.run_nfold_cv_vectors import ReportFolds
from quoll.classification_pipeline.modules.run_experiment import ExperimentComponent
from quoll.classification_pipeline.modules.make_bins import MakeBins 


################################################################################
###Component to thread the tasks together
################################################################################

@registercomponent
class NFoldCVSparse(WorkflowComponent):

    features = Parameter()
    labels = Parameter()
    vocabulary = Parameter()
    documents = Parameter()
    classifier_args = Parameter()

    n = IntParameter(default=10)
    weight = Parameter(default='frequency')
    prune = IntParameter(default=5000)
    balance = BoolParameter(default=False)
    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter(default=False)

    def accepts(self):
        return [ ( InputFormat(self,format_id='features',extension='.features.npz',inputparameter='features'), InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self, format_id='vocabulary', extension='.vocabulary.txt', inputparameter='vocabulary'), InputFormat(self,format_id='classifier_args',extension='.txt',inputparameter='classifier_args') ) ]

    def setup(self, workflow, input_feeds):

        bin_maker = workflow.new_task('make_bins', MakeBins, autopass=True, n=self.n)
        bin_maker.in_labels = input_feeds['labels']

        fold_runner = workflow.new_task('nfold_cv', RunFolds, autopass=True, n = self.n, weight=self.weight, prune=self.prune, balance=self.balance, classifier=self.classifier, ordinal=self.ordinal)
        fold_runner.in_bins = bin_maker.out_bins
        fold_runner.in_features = input_feeds['features']
        fold_runner.in_labels = input_feeds['labels']
        fold_runner.in_vocabulary = input_feeds['vocabulary']
        fold_runner.in_documents = input_feeds['documents']        
        fold_runner.in_classifier_args = input_feeds['classifier_args']

        folds_reporter = workflow.new_task('report_folds', ReportFolds, autopass = False)
        folds_reporter.in_expdirectory = fold_runner.out_exp

        return folds_reporter


################################################################################
###Experiment wrapper
################################################################################

class RunFolds(Task):

    in_bins = InputSlot()
    in_features = InputSlot()
    in_labels = InputSlot()
    in_vocabulary = InputSlot()
    in_documents = InputSlot()
    in_classifier_args = InputSlot()

    n = IntParameter()
    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()
    classifier = Parameter()
    ordinal = BoolParameter()

    def out_exp(self):
        return self.outputfrominput(inputformat='bins', stripextension='.bins.csv', addextension='.' + self.classifier + '.exp')
        
    def run(self):

        # make experiment directory
        self.setup_output_dir(self.out_exp().path)

        # for each fold
        for fold in range(self.n):
            yield Fold(directory=self.out_exp().path, features=self.in_features().path, labels=self.in_labels().path, vocabulary=self.in_vocabulary().path, bins=self.in_bins().path, documents=self.in_documents().path, classifier_args=self.in_classifier_args().path, i=fold, weight=self.weight, prune=self.prune, balance=self.balance, classifier=self.classifier, ordinal=self.ordinal)


################################################################################
###Fold Wrapper
################################################################################

@registercomponent
class Fold(WorkflowComponent):

    directory = Parameter()
    features = Parameter()
    labels = Parameter()
    vocabulary = Parameter()
    documents = Parameter()
    classifier_args = Parameter()
    bins = Parameter()

    i = IntParameter()
    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()
    classifier = Parameter()
    ordinal = BoolParameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='directory',extension='.exp',inputparameter='directory'), InputFormat(self,format_id='features',extension='.features.npz',inputparameter='features'), InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self, format_id='vocabulary', extension='.vocabulary.txt', inputparameter='vocabulary'), InputFormat(self,format_id='classifier_args',extension='.txt',inputparameter='classifier_args'), InputFormat(self,format_id='bins',extension='.bins.csv',inputparameter='bins') ) ]
 
    def setup(self, workflow, input_feeds):

        fold = workflow.new_task('run_fold', FoldTask, autopass=False, i=self.i, classifier=self.classifier, weight=self.weight, prune=self.prune, balance=self.balance, ordinal=self.ordinal)
        fold.in_directory = input_feeds['directory']
        fold.in_features = input_feeds['features']
        fold.in_vocabulary = input_feeds['vocabulary']
        fold.in_labels = input_feeds['labels']
        fold.in_documents = input_feeds['documents']
        fold.in_classifier_args = input_feeds['classifier_args']  
        fold.in_bins = input_feeds['bins']   

        return fold


class FoldTask(Task):

    in_directory = InputSlot()
    in_bins = InputSlot()
    in_features = InputSlot()
    in_labels = InputSlot()
    in_vocabulary = InputSlot()
    in_documents = InputSlot()
    in_classifier_args = InputSlot()
    
    i = IntParameter()
    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()
    classifier = Parameter()
    ordinal = BoolParameter()


    def out_fold(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i))    

    def out_trainfeatures(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.features.npz')        

    def out_testfeatures(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.features.npz')

    def out_trainvectors(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.vectors.npz')

    def out_testvectors(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.vectors.npz')

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.labels')

    def out_testlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.labels')

    def out_traindocuments(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.docs.txt')

    def out_testdocuments(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.docs.txt')

    def out_classifier_args(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/classifier_args.txt')

    def run(self):

        # make fold directory
        self.setup_output_dir(self.out_fold().path)

        # open bin indices
        dr = docreader.Docreader()
        bins_str = dr.parse_csv(self.in_bins().path)
        bins = [[int(x) for x in bin] for bin in bins_str]

        # open features
        loader = numpy.load(self.in_features().path)
        featurized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # open labels
        with open(self.in_labels().path,'r',encoding='utf-8') as infile:
            labels = numpy.array(infile.read().strip().split('\n'))

        # open documents
        with open(self.in_documents().path,'r',encoding='utf-8') as infile:
            documents = numpy.array(infile.read().strip().split('\n'))

        # open classifier args
        with open(self.in_classifier_args().path) as infile:
            classifier_args = infile.read().rstrip().split('\n')

        # set training and test data
        train_features = sparse.vstack([featurized_instances[indices,:] for j,indices in enumerate(bins) if j != self.i])
        train_labels = numpy.concatenate([labels[indices] for j,indices in enumerate(bins) if j != self.i])
        train_documents = numpy.concatenate([documents[indices] for j,indices in enumerate(bins) if j != self.i])
        test_features = featurized_instances[bins[self.i]]
        test_labels = labels[bins[self.i]]
        test_documents = documents[bins[self.i]]

        # write experiment data to files in fold directory
        numpy.savez(self.out_trainfeatures().path, data=train_features.data, indices=train_features.indices, indptr=train_features.indptr, shape=train_features.shape)
        numpy.savez(self.out_testfeatures().path, data=test_features.data, indices=test_features.indices, indptr=test_features.indptr, shape=test_features.shape)
        with open(self.out_trainlabels().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(train_labels))
        with open(self.out_testlabels().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_labels))
        with open(self.out_traindocuments().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(train_documents))
        with open(self.out_testdocuments().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_documents))
        with open(self.out_classifier_args().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(classifier_args))

        print('Running experiment for fold',self.i)

        yield ExperimentComponent(trainfeatures=self.out_trainfeatures().path, trainlabels=self.out_trainlabels().path, testfeatures=self.out_testfeatures().path, testlabels=self.out_testlabels().path, vocabulary=self.in_vocabulary().path, classifier_args=self.out_classifier_args().path, documents=self.out_testdocuments().path, weight=self.weight, prune=self.prune, balance=self.balance, classifier=self.classifier, ordinal=self.ordinal)
