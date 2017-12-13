
from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, IntParameter, BoolParameter
import numpy
from scipy import sparse
from collections import defaultdict
import glob

import quoll.classification_pipeline.functions.nfold_cv_functions as nfold_cv_functions
import quoll.classification_pipeline.functions.linewriter as linewriter
import quoll.classification_pipeline.functions.docreader as docreader
import quoll.classification_pipeline.functions.reporter as reporter

from quoll.classification_pipeline.modules.select_features import SelectFeatures
from quoll.classification_pipeline.modules.run_experiment import ExperimentComponentFilter
from quoll.classification_pipeline.modules.make_bins import MakeBins 
from quoll.classification_pipeline.modules.run_nfold_cv_vectors import ReportFolds

################################################################################
###Component to thread the tasks together
################################################################################

@registercomponent
class NFoldCVFilter(WorkflowComponent):
    
    vectors = Parameter()
    labels = Parameter()
    documents = Parameter()
    featurenames = Parameter()
    classifier_args = Parameter()

    threshold_strength = Parameter()
    threshold_correlation = Parameter()
    n = IntParameter(default=10)
    stepsize = IntParameter(default=1)
    teststart = IntParameter(default=0) # give the first index to indicate a range to fix these instances for testing (some instances are then always used in training and not in the testset; this is useful when you want to compare the addition of a certain dataset to train on)
    testend = IntParameter(default=-1)
    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter(default=False)

    def accepts(self):
        return [ ( InputFormat(self,format_id='vectors',extension='.vectors.npz',inputparameter='vectors'), InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self, format_id='featurenames', extension='.txt', inputparameter='featurenames'), InputFormat(self, format_id='classifier_args', extension='.txt', inputparameter='classifier_args') ) ]
    
    def setup(self, workflow, input_feeds):

        bin_maker = workflow.new_task('make_bins', MakeBins, autopass=True, n=self.n, steps=self.stepsize)
        bin_maker.in_labels = input_feeds['labels']

        fold_runner = workflow.new_task('run_folds_filter', RunFoldsFilter, autopass=True, n=self.n, threshold_strength=self.threshold_strength,threshold_correlation=self.threshold_correlation, teststart=self.teststart, testend=self.testend, classifier=self.classifier, ordinal=self.ordinal)
        fold_runner.in_bins = bin_maker.out_bins
        fold_runner.in_vectors = input_feeds['vectors']
        fold_runner.in_labels = input_feeds['labels']
        fold_runner.in_documents = input_feeds['documents']        
        fold_runner.in_featurenames = input_feeds['featurenames']
        fold_runner.in_classifier_args = input_feeds['classifier_args']

        folds_reporter = workflow.new_task('report_folds', ReportFolds, autopass = False)
        folds_reporter.in_expdirectory = fold_runner.out_exp

        return folds_reporter

    
################################################################################
###Experiment wrapper
################################################################################

class RunFoldsFilter(Task):

    in_bins = InputSlot()
    in_vectors = InputSlot()
    in_labels = InputSlot()
    in_documents = InputSlot()
    in_featurenames = InputSlot()
    in_classifier_args = InputSlot()

    threshold_strength = Parameter()
    threshold_correlation = Parameter()
    n = IntParameter()
    teststart = IntParameter()
    testend = IntParameter()
    classifier = Parameter()
    ordinal = BoolParameter()

    def out_exp(self):
        return self.outputfrominput(inputformat='bins', stripextension='.bins.csv', addextension='.filter_' + self.threshold_strength + '_' + self.threshold_correlation + '.' + self.classifier + '.exp')
        
    def run(self):

        # make experiment directory
        self.setup_output_dir(self.out_exp().path)

        # for each fold
        for fold in range(self.n):
            yield FoldFilter(directory=self.out_exp().path, vectors=self.in_vectors().path, labels=self.in_labels().path, bins=self.in_bins().path, documents=self.in_documents().path, featurenames=self.in_featurenames().path, classifier_args=self.in_classifier_args().path, i=fold, threshold_strength=self.threshold_strength, threshold_correlation=self.threshold_correlation, teststart=self.teststart, testend=self.testend, classifier=self.classifier, ordinal=self.ordinal)


################################################################################
###Fold Wrapper
################################################################################

@registercomponent
class FoldFilter(WorkflowComponent):

    directory = Parameter()
    vectors = Parameter()
    labels = Parameter()
    documents = Parameter()
    featurenames = Parameter()
    bins = Parameter()
    classifier_args = Parameter()

    i = IntParameter()
    threshold_strength = Parameter()
    threshold_correlation = Parameter()
    teststart = IntParameter()
    testend = IntParameter()
    classifier = Parameter()
    ordinal = BoolParameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='directory',extension='.exp',inputparameter='directory'), InputFormat(self,format_id='vectors',extension='.vectors.npz',inputparameter='vectors'), InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self, format_id='featurenames', extension='.txt', inputparameter='featurenames'), InputFormat(self,format_id='bins',extension='.bins.csv',inputparameter='bins'), InputFormat(self,format_id='classifier_args',extension='.txt',inputparameter='classifier_args')  ) ]
    
    def setup(self, workflow, input_feeds):

        fold = workflow.new_task('run_fold', FoldFilterTask, autopass=False, i=self.i, threshold_strength=self.threshold_strength,threshold_correlation=self.threshold_correlation, teststart=self.teststart, testend=self.testend, classifier=self.classifier, ordinal=self.ordinal)
        fold.in_directory = input_feeds['directory']
        fold.in_vectors = input_feeds['vectors']
        fold.in_labels = input_feeds['labels']
        fold.in_documents = input_feeds['documents']
        fold.in_featurenames = input_feeds['featurenames']
        fold.in_bins = input_feeds['bins']   
        fold.in_classifier_args = input_feeds['classifier_args']

        return fold

class FoldFilterTask(Task):

    in_directory = InputSlot()
    in_vectors = InputSlot()
    in_labels = InputSlot()
    in_documents = InputSlot()
    in_featurenames = InputSlot()
    in_bins = InputSlot()
    in_classifier_args = InputSlot()
    
    i = IntParameter()
    threshold_strength = Parameter()
    threshold_correlation = Parameter()
    teststart = IntParameter()
    testend = IntParameter()
    classifier = Parameter()
    ordinal = BoolParameter()

    def out_fold(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i))    

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

    def out_featurenames(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/featurenames.txt')

    def run(self):

        # make fold directory
        self.setup_output_dir(self.out_fold().path)

        # open bin indices
        dr = docreader.Docreader()
        bins_str = dr.parse_csv(self.in_bins().path)
        bins = [[int(x) for x in bin] for bin in bins_str]

        # open instances
        loader = numpy.load(self.in_vectors().path)
        instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # open labels
        with open(self.in_labels().path,'r',encoding='utf-8') as infile:
            labels = numpy.array(infile.read().strip().split('\n'))

        # open documents
        with open(self.in_documents().path,'r',encoding='utf-8') as infile:
            documents = numpy.array(infile.read().strip().split('\n'))

        # open featurenames
        with open(self.in_featurenames().path,'r',encoding='utf-8') as infile:
            featurenames = infile.read().strip().split('\n')

        # write data to files in fold directory
        train_vectors = sparse.vstack([instances[indices,:] for j,indices in enumerate(bins) if j != self.i])
        train_labels = numpy.concatenate([labels[indices] for j,indices in enumerate(bins) if j != self.i])
        train_documents = numpy.concatenate([documents[indices] for j,indices in enumerate(bins) if j != self.i])
        test_vectors = instances[bins[self.i]]
        test_labels = labels[bins[self.i]]
        test_documents = documents[bins[self.i]]
        numpy.savez(self.out_trainvectors().path, data=train_vectors.data, indices=train_vectors.indices, indptr=train_vectors.indptr, shape=train_vectors.shape)
        numpy.savez(self.out_testvectors().path, data=test_vectors.data, indices=test_vectors.indices, indptr=test_vectors.indptr, shape=test_vectors.shape)
        with open(self.out_trainlabels().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(train_labels))
        with open(self.out_testlabels().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_labels))
        with open(self.out_traindocuments().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(train_documents))
        with open(self.out_testdocuments().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_documents))
        with open(self.out_featurenames().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(featurenames))

        print('Running experiment for fold',self.i)
        yield ExperimentComponentFilter(train=self.out_trainvectors().path, trainlabels=self.out_trainlabels().path, test=self.out_testvectors().path, testlabels=self.out_testlabels().path, documents=self.out_testdocuments().path, featurenames=self.out_featurenames().path, classifier_args=self.in_classifier_args().path, threshold_strength=self.threshold_strength, threshold_correlation=self.threshold_correlation, classifier=self.classifier, ordinal=self.ordinal)

