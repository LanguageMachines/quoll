
from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, IntParameter, BoolParameter
import numpy
from scipy import sparse
from collections import defaultdict
import glob

import quoll.classification_pipeline.functions.nfold_cv_functions as nfold_cv_functions
import quoll.classification_pipeline.functions.linewriter as linewriter
import quoll.classification_pipeline.functions.docreader as docreader

from quoll.classification_pipeline.modules.select_features import SelectFeatures
from quoll.classification_pipeline.modules.run_experiment import ExperimentComponentLin
from quoll.classification_pipeline.modules.make_bins import MakeBins 

################################################################################
###Component to thread the tasks together
################################################################################

@registercomponent
class NFoldCVLin(WorkflowComponent):
    
    vectors = Parameter()
    labels = Parameter()
    documents = Parameter()
    featurenames = Parameter()
    featurecorrelation = Parameter()

    svorim_path = Parameter()
    feature_cutoff = IntParameter()
    n = IntParameter(default=10)
    stepsize = IntParameter(default=1)

    def accepts(self):
        return [ ( InputFormat(self,format_id='vectors',extension='.vectors.npz',inputparameter='vectors'), InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self, format_id='featurecorrelation', extension='.txt', inputparameter='featurecorrelation'), InputFormat(self, format_id='featurenames', extension='.txt', inputparameter='featurenames') ) ]
    
    def setup(self, workflow, input_feeds):

        bin_maker = workflow.new_task('make_bins', MakeBins, autopass=True, n=self.n, steps=self.stepsize)
        bin_maker.in_labels = input_feeds['labels']

        fold_runner = workflow.new_task('run_folds_lin', RunFoldsLin, autopass=True, n=self.n, cutoff=self.feature_cutoff, svorim_path=self.svorim_path)
        fold_runner.in_bins = bin_maker.out_bins
        fold_runner.in_vectors = input_feeds['vectors']
        fold_runner.in_labels = input_feeds['labels']
        fold_runner.in_documents = input_feeds['documents']        
        fold_runner.in_featurenames = input_feeds['featurenames']
        fold_runner.in_featurecorrelation = input_feeds['featurecorrelation']

        folds_reporter = workflow.new_task('report_folds', ReportFolds, autopass = False)
        folds_reporter.in_expdirectory = fold_runner.out_exp

        return folds_reporter

    
################################################################################
###Experiment wrapper
################################################################################

class RunFoldsLin(Task):

    in_bins = InputSlot()
    in_vectors = InputSlot()
    in_labels = InputSlot()
    in_documents = InputSlot()
    in_featurenames = InputSlot()
    in_featurecorrelation = InputSlot()

    n = IntParameter()
    svorim_path = Parameter()
    cutoff = IntParameter()

    def out_exp(self):
        return self.outputfrominput(inputformat='bins', stripextension='.bins.csv', addextension='.nocorr_ranked_' + str(self.cutoff) + '.exp')
        
    def run(self):

        # make experiment directory
        self.setup_output_dir(self.out_exp().path)

        # for each fold
        for fold in range(self.n):
            yield FoldLin(directory=self.out_exp().path, vectors=self.in_vectors().path, labels=self.in_labels().path, bins=self.in_bins().path, documents=self.in_documents().path, featurenames=self.in_featurenames().path, featurecorrelation=self.in_featurecorrelation().path, i=fold, svorim_path=self.svorim_path, cutoff=self.cutoff)


################################################################################
###Fold Wrapper
################################################################################

@registercomponent
class FoldLin(WorkflowComponent):

    directory = Parameter()
    vectors = Parameter()
    labels = Parameter()
    documents = Parameter()
    featurenames = Parameter()
    featurecorrelation = Parameter()
    bins = Parameter()

    i = IntParameter()
    svorim_path = Parameter()
    cutoff = IntParameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='directory',extension='.exp',inputparameter='directory'), InputFormat(self,format_id='vectors',extension='.vectors.npz',inputparameter='vectors'), InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self, format_id='featurecorrelation', extension='.txt', inputparameter='featurecorrelation'), InputFormat(self, format_id='featurenames', extension='.txt', inputparameter='featurenames'), InputFormat(self,format_id='bins',extension='.bins.csv',inputparameter='bins') ) ]
    
    def setup(self, workflow, input_feeds):

        fold = workflow.new_task('run_fold', FoldVectorsTask, autopass=False, i=self.i, svorim_path=self.svorim_path, cutoff=self.cutoff)
        fold.in_directory = input_feeds['directory']
        fold.in_vectors = input_feeds['vectors']
        fold.in_labels = input_feeds['labels']
        fold.in_documents = input_feeds['documents']
        fold.in_featurenames = input_feeds['featurenames']
        fold.in_featurecorrelation = input_feeds['featurecorrelation']
        fold.in_bins = input_feeds['bins']   

        return fold

class FoldVectorsTask(Task):

    in_directory = InputSlot()
    in_vectors = InputSlot()
    in_labels = InputSlot()
    in_documents = InputSlot()
    in_featurenames = InputSlot()
    in_featurecorrelation = InputSlot()
    in_bins = InputSlot()
    
    i = IntParameter()
    svorim_path = Parameter()
    cutoff = IntParameter()

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
        yield ExperimentComponentLin(train=self.out_trainvectors().path, trainlabels=self.out_trainlabels().path, test=self.out_testvectors().path, testlabels=self.out_testlabels().path, documents=self.out_testdocuments().path, featurenames=self.out_featurenames().path, featurecorrelation=self.in_featurecorrelation().path, feature_cutoff=self.cutoff, svorim_path=self.svorim_path)

################################################################################
###Reporter
################################################################################

@registercomponent
class ReportFoldsComponent(StandardWorkflowComponent):

    def accepts(self):
        return InputFormat(self, format_id='expdirectory', extension='.exp')
                    
    def autosetup(self):
        return ReportFolds

class ReportFolds(Task):

    in_expdirectory = InputSlot()

    def out_performance(self):
        return self.outputfrominput(inputformat='expdirectory', stripextension='.exp', addextension='.performance.csv')    

    def out_docpredictions(self):
        return self.outputfrominput(inputformat='expdirectory', stripextension='.exp', addextension='.docpredictions.csv')    
 
    def run(self):

        # gather fold reports
        print('gathering fold reports')
        performance_files = [ filename for filename in glob.glob(self.in_expdirectory().path + '/fold*/*.performance.csv') ]
        docprediction_files = [ filename for filename in glob.glob(self.in_expdirectory().path + '/fold*/*.docpredictions.csv') ]

        # calculate average performance
        dr = docreader.Docreader()
        performance_combined = [dr.parse_csv(performance_file) for performance_file in performance_files]
        all_performance = [performance_combined[0][0]] # headers
        label_performance = defaultdict(list)
        for p in performance_combined:
            for i in range(1,len(p)): # labels 
                performance = []
                label = p[i][0] # name of label
                for j in range(1,len(p[i])): # report values
                    performance.append(float(p[i][j]))
                label_performance[label].append(performance)

        # compute mean and sum per label
        if 'micro' in label_performance.keys():
            labels_order = [label for label in label_performance.keys() if label != 'micro'] + ['micro']
        else:
            labels_order = sorted(label_performance.keys())

        for label in labels_order:
            average_performance = [label]
            for j in range(0,len(label_performance[label][0])-3):
                average_performance.append(str(round(numpy.mean([float(p[j]) for p in label_performance[label]]),2)) + '(' + str(round(numpy.std([float(p[j]) for p in label_performance[label]]),2)) + ')')
            for j in range(len(label_performance[label][0])-3,len(label_performance[label][0])):
                average_performance.append(str(sum([int(p[j]) for p in label_performance[label]])))
            all_performance.append(average_performance)

        lw = linewriter.Linewriter(all_performance)
        lw.write_csv(self.out_performance().path)

        # write predictions per document
        docpredictions = sum([dr.parse_csv(docprediction_file) for docprediction_file in docprediction_files], [])
        lw = linewriter.Linewriter(docpredictions)
        lw.write_csv(self.out_docpredictions().path)
