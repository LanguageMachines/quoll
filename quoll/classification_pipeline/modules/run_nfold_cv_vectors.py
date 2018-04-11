
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
from quoll.classification_pipeline.modules.run_experiment import * 
from quoll.classification_pipeline.modules.make_bins import MakeBins 

################################################################################
###Component to thread the tasks together
################################################################################

@registercomponent
class NFoldCV(WorkflowComponent):
    
    vectors = Parameter()
    labels = Parameter()
    documents = Parameter()
    classifier_args = Parameter()
    featurenames = Parameter()

    n = IntParameter(default=10)
    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter(default=False)
    stepsize = IntParameter(default=1)
    raw_labels = Parameter(default=False)

    def accepts(self):
        return [ ( InputFormat(self,format_id='vectors',extension='.vectors.npz',inputparameter='vectors'), InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self,format_id='classifier_args',extension='.txt',inputparameter='classifier_args'), InputFormat(self,format_id='featurenames',extension='.txt',inputparameter='featurenames') ) ]
    
    def setup(self, workflow, input_feeds):

        bin_maker = workflow.new_task('make_bins', MakeBins, autopass=True, n=self.n, steps=self.stepsize)
        bin_maker.in_labels = input_feeds['labels']

        fold_runner = workflow.new_task('run_folds_vectors', RunFoldsVectors, autopass=True, n=self.n, classifier=self.classifier, ordinal=self.ordinal, raw_labels=self.raw_labels)
        fold_runner.in_bins = bin_maker.out_bins
        fold_runner.in_vectors = input_feeds['vectors']
        fold_runner.in_labels = input_feeds['labels']
        fold_runner.in_documents = input_feeds['documents']        
        fold_runner.in_classifier_args = input_feeds['classifier_args']
        fold_runner.in_featurenames = input_feeds['featurenames']

        folds_reporter = workflow.new_task('report_folds', ReportFolds, ordinal=self.ordinal, autopass = False)
        folds_reporter.in_expdirectory = fold_runner.out_exp

        return folds_reporter

    
################################################################################
###Experiment wrapper
################################################################################

class RunFoldsVectors(Task):

    in_bins = InputSlot()
    in_vectors = InputSlot()
    in_labels = InputSlot()
    in_documents = InputSlot()
    in_classifier_args = InputSlot()
    in_featurenames = InputSlot()

    n = IntParameter()
    classifier = Parameter()
    ordinal = BoolParameter()
    raw_labels = Parameter()

    def out_exp(self):
        return self.outputfrominput(inputformat='bins', stripextension='.bins.csv', addextension='.' + self.classifier + '.exp')
        
    def run(self):

        # make experiment directory
        self.setup_output_dir(self.out_exp().path)

        # for each fold
        for fold in range(self.n):
            yield FoldVectors(directory=self.out_exp().path, vectors=self.in_vectors().path, labels=self.in_labels().path, bins=self.in_bins().path, documents=self.in_documents().path, classifier_args=self.in_classifier_args().path, featurenames=self.in_featurenames().path, i=fold, classifier=self.classifier, ordinal=self.ordinal, raw_labels=self.raw_labels)


################################################################################
###Fold Wrapper
################################################################################

@registercomponent
class FoldVectors(WorkflowComponent):

    directory = Parameter()
    vectors = Parameter()
    labels = Parameter()
    documents = Parameter()
    classifier_args = Parameter()
    featurenames = Parameter()
    bins = Parameter()

    i = IntParameter()
    classifier = Parameter()
    ordinal = BoolParameter()
    raw_labels = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='directory',extension='.exp',inputparameter='directory'), InputFormat(self,format_id='vectors',extension='.vectors.npz',inputparameter='vectors'), InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self,format_id='classifier_args',extension='.txt',inputparameter='classifier_args'), InputFormat(self,format_id='bins',extension='.bins.csv',inputparameter='bins'), InputFormat(self,format_id='featurenames',extension='.txt',inputparameter='featurenames') ) ]
    
    def setup(self, workflow, input_feeds):

        fold = workflow.new_task('run_fold', FoldVectorsTask, autopass=False, i=self.i, classifier=self.classifier, ordinal=self.ordinal, raw_labels=self.raw_labels)
        fold.in_directory = input_feeds['directory']
        fold.in_vectors = input_feeds['vectors']
        fold.in_labels = input_feeds['labels']
        fold.in_documents = input_feeds['documents']
        fold.in_classifier_args = input_feeds['classifier_args']
        fold.in_featurenames = input_feeds['featurenames']  
        fold.in_bins = input_feeds['bins']   

        return fold

class FoldVectorsTask(Task):

    in_directory = InputSlot()
    in_vectors = InputSlot()
    in_labels = InputSlot()
    in_documents = InputSlot()
    in_classifier_args = InputSlot()
    in_featurenames = InputSlot()
    in_bins = InputSlot()
    
    i = IntParameter()
    classifier = Parameter()
    ordinal = BoolParameter()
    raw_labels = Parameter()

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

    def out_classifier_args(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/classifier_args.txt')

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

        # check for raw labels
        if self.raw_labels:
            with open(self.raw_labels,'r',encoding='utf-8') as infile:
                raw_labels = numpy.array(infile.read().strip().split('\n'))

        # open documents
        with open(self.in_documents().path,'r',encoding='utf-8') as infile:
            documents = numpy.array(infile.read().strip().split('\n'))

        # open classifier args
        with open(self.in_classifier_args().path) as infile:
            classifier_args = infile.read().rstrip().split('\n')

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
        if self.raw_labels:
            train_labels_raw = numpy.concatenate([raw_labels[indices] for j,indices in enumerate(bins) if j != self.i])
            raw_labels_out = self.out_fold().path + '/train.labels_raw.txt' 
            with open(raw_labels_out,'w',encoding='utf-8') as outfile:
                outfile.write('\n'.join(train_labels_raw))
        with open(self.out_testlabels().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_labels))
        with open(self.out_traindocuments().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(train_documents))
        with open(self.out_testdocuments().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_documents))
        with open(self.out_classifier_args().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(classifier_args))

        print('Running experiment for fold',self.i)
        if self.raw_labels:
            yield ExperimentComponentVector(train=self.out_trainvectors().path, trainlabels=self.out_trainlabels().path, test=self.out_testvectors().path, testlabels=self.out_testlabels().path, featurenames=self.in_featurenames().path, classifier_args=self.out_classifier_args().path, documents=self.out_testdocuments().path, classifier=self.classifier, ordinal=self.ordinal, raw_labels=raw_labels_out) 
        else:
            yield ExperimentComponentVector(train=self.out_trainvectors().path, trainlabels=self.out_trainlabels().path, test=self.out_testvectors().path, testlabels=self.out_testlabels().path, featurenames=self.in_featurenames().path, classifier_args=self.out_classifier_args().path, documents=self.out_testdocuments().path, classifier=self.classifier, ordinal=self.ordinal, raw_labels=self.raw_labels) 


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

    ordinal = BoolParameter()

    def out_report(self):
        return self.outputfrominput(inputformat='expdirectory', stripextension='.exp', addextension='.report')  
 
    def run(self):

        # make report directory
        self.setup_output_dir(self.out_report().path)
        
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
                no_float = []
                performance = []
                label = p[i][0] # name of label
                for j in range(1,len(p[i])): # report values
                    if j not in no_float:
                        try:
                            performance.append(float(p[i][j]))
                        except:
                            no_float.append(j)
                            performance.append('nan')
                            for lp in label_performance[label]:
                                lp[j] = 'nan'
                    else:
                        performance.append('nan')
                label_performance[label].append(performance)

        # compute mean and sum per label
        if 'micro' in label_performance.keys():
            labels_order = [label for label in label_performance.keys() if label != 'micro'] + ['micro']
        else:
            labels_order = sorted(label_performance.keys())

        for label in labels_order:
            average_performance = [label]
            for j in range(0,len(label_performance[label][0])-3):
                if label_performance[label][0][j] != 'nan':
                    average_performance.append(str(round(numpy.mean([float(p[j]) for p in label_performance[label]]),2)) + '(' + str(round(numpy.std([float(p[j]) for p in label_performance[label]]),2)) + ')')
                else:
                    average_performance.append('nan')
            for j in range(len(label_performance[label][0])-3,len(label_performance[label][0])):
                average_performance.append(str(sum([int(p[j]) for p in label_performance[label]])))
            all_performance.append(average_performance)

        lw = linewriter.Linewriter(all_performance)
        lw.write_csv(self.out_report().path + '/macro_performance.csv')

        # write predictions per document
        label_order = [x.split('prediction prob for ')[1] for x in dr.parse_csv(docprediction_files[0])[0][3:]]
        docpredictions = sum([dr.parse_csv(docprediction_file)[1:] for docprediction_file in docprediction_files], [])
        documents = [line[0] for line in docpredictions]
        labels = [line[1] for line in docpredictions]
        unique_labels = labels_order
        predictions = [line[2] for line in docpredictions]
        full_predictions = [line[3:] for line in docpredictions]

        # initiate reporter
        rp = reporter.Reporter(predictions, full_predictions, label_order, labels, unique_labels, self.ordinal, documents)
        predictions_by_document = rp.predictions_by_document()
        lw = linewriter.Linewriter(predictions_by_document)
        lw.write_csv(self.out_report().path + '/docpredictions.csv')

        # report performance
        if self.ordinal:
            performance = rp.assess_ordinal_performance()
        else:
            performance = rp.assess_performance()
        lw = linewriter.Linewriter(performance)
        lw.write_csv(self.out_report().path + '/performance.csv')

        # report fps per label
        self.setup_output_dir(self.out_report().path + '/ranked_fps')
        for label in unique_labels:
            try:
                ranked_fps = rp.return_ranked_fps(label)
                outfile = self.out_report().path + '/ranked_fps/' + label + '.csv'
                lw = linewriter.Linewriter(ranked_fps)
                lw.write_csv(outfile)
            except:
                continue

        # report tps per label
        self.setup_output_dir(self.out_report().path + '/ranked_tps')
        for label in unique_labels:
            try:
                ranked_tps = rp.return_ranked_tps(label)
                outfile = self.out_report().path + '/ranked_tps/' + label + '.csv'
                lw = linewriter.Linewriter(ranked_tps)
                lw.write_csv(outfile)
            except:
                continue
            
        # report fns per label
        self.setup_output_dir(self.out_report().path + '/ranked_fns')
        for label in unique_labels:
            try:
                ranked_fns = rp.return_ranked_fns(label)
                outfile = self.out_report().path + '/ranked_fns/' + label + '.csv'
                lw = linewriter.Linewriter(ranked_fns)
                lw.write_csv(outfile)
            except:
                continue
                
        # report tns per label
        self.setup_output_dir(self.out_report().path + '/ranked_tns')
        for label in unique_labels:
            try:
                ranked_tns = rp.return_ranked_tns(label)
                outfile = self.out_report().path + '/ranked_tns/' + label + '.csv'
                lw = linewriter.Linewriter(ranked_tns)
                lw.write_csv(outfile)
            except:
                continue
                
        # report confusion matrix
        if self.ordinal: # to make a confusion matrix, the labels should be formatted as string
            rp = reporter.Reporter(predictions, full_predictions, label_order, labels, unique_labels, False, documents)
        confusion_matrix = rp.return_confusion_matrix()
        with open(self.out_report().path + '/confusion_matrix.csv','w') as cm_out:
            cm_out.write(confusion_matrix)

        # report performance-at
        self.setup_output_dir(self.out_report().path + '/performance_at_dir')
        if len(unique_labels) >= 9:
            prat_opts = [3,5,7,9]
        elif len(unique_labels) >= 7:
            prat_opts = [3,5,7]
        elif len(unique_labels) >= 5:
            prat_opts = [3,5]
        elif len(unique_labels) >= 3:
            prat_opts = [3]
        else:
            prat_opts = []
        for po in prat_opts:
            outfile = self.out_report().path + '/performance_at_dir/performance_at_' + str(po) + '.txt'
            rp = reporter.Reporter(predictions, full_predictions, label_order, labels, unique_labels, self.ordinal, documents, po)
            if self.ordinal:
                performance = rp.assess_ordinal_performance()
            else:
                performance = rp.assess_performance()
            lw = linewriter.Linewriter(performance)
            lw.write_csv(outfile)

################################################################################
###Regression Reporter
################################################################################

@registercomponent
class ReportFoldsRegressionComponent(StandardWorkflowComponent):

    def accepts(self):
        return InputFormat(self, format_id='expdirectory', extension='.exp')
                    
    def autosetup(self):
        return ReportFoldsRegression

class ReportFoldsRegression(Task):

    in_expdirectory = InputSlot()

    def out_macro_performance(self):
        return self.outputfrominput(inputformat='expdirectory', stripextension='.exp', addextension='.macro_performance_regression.txt')  

    def run(self):

        # gather fold reports
        print('gathering fold reports')
        performance_files = [ filename for filename in glob.glob(self.in_expdirectory().path + '/fold*/*.regression_performance.csv') ]
        
        # calculate average performance
        performance_combined = []
        for pf in performance_files:
            with open(pf) as file_in:
                performance_combined.append(float(file_in.read().strip()))
        avg_performance = numpy.mean(performance_combined)

        # write to file
        with open(self.out_macro_performance().path,'w',encoding='utf-8') as file_out:
            file_out.write(str(avg_performance))
    
