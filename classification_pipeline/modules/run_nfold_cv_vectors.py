
from luiginlp.engine import Task, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, IntParameter, BoolParameter
import numpy
from scipy import sparse
from collections import defaultdict

import functions.nfold_cv_functions as nfold_cv_functions
import functions.linewriter as linewriter
import functions.docreader as docreader

from run_experiment import ExperimentComponentVector 
from report_performance import ReporterComponent

class RunFold(Task):

    in_vectors = InputSlot()
    in_labels = InputSlot()
    in_featurelabels = InputSlot()
    in_featurenames = InputSlot()
    in_bins = InputSlot()
    
    i = IntParameter()
    classifier = Parameter()
    documents = Parameter()
    
    def out_fold(self):
        return self.outputfrominput(inputformat='bins', stripextension='.bins.csv', addextension='.fold' + str(self.i))    
    
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
        try:
            with open(self.documents,'r',encoding='utf-8') as infile:
                documents = numpy.array(infile.read().split('\n'))
        except FileNotFoundError:
            documents = numpy.array(['-'] * len(labels))

        # set training and test data
        train_vectors = sparse.vstack([instances[indices,:] for j,indices in enumerate(bins) if j != self.i])
        train_labels = numpy.concatenate([labels[indices] for j,indices in enumerate(bins) if j != self.i])
        test_vectors = instances[bins[self.i]]
        test_labels = labels[bins[self.i]]
        test_documents = documents[bins[self.i]]

        # write experiment data to files in fold directory
        trv_out = self.out_fold().path + '/train.vectors.npz'
        numpy.savez(trv_out, data=train_vectors.data, indices=train_vectors.indices, indptr=train_vectors.indptr, shape=train_vectors.shape)        
        trl_out = self.out_fold().path + '/train.labels'
        with open(trl_out,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(train_labels))
        tev_out = self.out_fold().path + '/test.vectors.npz'
        numpy.savez(tev_out, data=test_vectors.data, indices=test_vectors.indices, indptr=test_vectors.indptr, shape=test_vectors.shape)        
        tel_out = self.out_fold().path + '/test.labels'
        with open(tel_out,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_labels))
        docs_out = self.out_fold().path + '/docs.txt'
        with open(docs_out,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_documents))

        print('Running experiment for fold',self.i)

        yield ExperimentComponentVector(train=trv_out, trainlabels=trl_out, test=tev_out, testlabels=tel_out, featurenames=self.in_featurenames().path, classifier=self.classifier, documents=docs_out) 
    

class Fold(WorkflowComponent):

    vectors = Parameter()
    labels = Parameter()
    featurenames = Parameter()
    bins = Parameter()
    
    i = IntParameter()
    classifier = Parameter()
    documents = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='vectors',extension='.vectors.npz',inputparameter='vectors'), InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), InputFormat(self, format_id='featurenames', extension='.featurenames.txt', inputparameter='featurenames'), InputFormat(self, format_id='bins', extension='.bins.csv', inputparameter='bins') ) ]
    
    def setup(self, workflow, input_feeds):
        
        fold = workflow.new_task('fold', RunFold, autopass=True, i=self.i, classifier=self.classifier, documents=self.documents)            
        fold.in_vectors = input_feeds['vectors']
        fold.in_labels = input_feeds['labels']
        fold.in_featurenames = input_feeds['featurenames']        
        fold.in_bins = input_feeds['bins']
        return fold
    
class Run_nfold_cv(Task):

    in_vectors = InputSlot()
    in_labels = InputSlot()
    in_featurenames = InputSlot()

    n = IntParameter()
    classifier = Parameter()
    documents = Parameter()

    def out_folds(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.' + str(self.n) + 'fold_cv')
        
    def run(self):

        # make nfold_cv directory
        self.setup_output_dir(self.out_folds().path)

        # open labels
        with open(self.in_labels().path,'r',encoding='utf-8') as infile:
            labels = numpy.array(infile.read().strip().split('\n'))

        # select rows per fold based on shape of the features
        num_instances = len(labels)
        fold_indices = nfold_cv_functions.return_fold_indices(num_instances, self.n)        
        
        # write indices of bins to file
        bins_out = self.out_folds().path + '/folds.bins.csv'
        lw = linewriter.Linewriter(fold_indices)
        lw.write_csv(bins_out)
        
        # run folds
        performance_files = []
        docprediction_files = []
        for fold in range(self.n):
            yield Fold(vectors=self.in_vectors().path, labels=self.in_labels().path, featurenames=self.in_featurenames().path, bins=bins_out, i=fold, classifier=self.classifier, documents=self.documents)
            
            performance_files.append(self.out_folds().path + '/folds.fold' + str(fold) + '/test.performance.csv')
            docprediction_files.append(self.out_folds().path + '/folds.fold' + str(fold) + '/test.docpredictions.csv')
            
        performance_out = self.out_folds().path + '/folds.performance.csv'
        docpredictions_out = self.out_folds().path + '/folds.docpredictions.csv'

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
        labels_order = [label for label in label_performance.keys() if label != 'micro'] + ['micro']
        for label in labels_order:
            average_performance = [label]
            for j in range(0,len(label_performance[label][0])-3):
                average_performance.append(str(round(numpy.mean([float(p[j]) for p in label_performance[label]]),2)) + '(' + str(round(numpy.std([float(p[j]) for p in label_performance[label]]),2)) + ')')
            for j in range(len(label_performance[label][0])-3,len(label_performance[label][0])):
                average_performance.append(str(sum([int(p[j]) for p in label_performance[label]])))
            all_performance.append(average_performance)

        lw = linewriter.Linewriter(all_performance)
        lw.write_csv(performance_out)

        # write predictions per document
        docpredictions = sum([dr.parse_csv(docprediction_file) for docprediction_file in docprediction_files], [])
        lw = linewriter.Linewriter(docpredictions)
        lw.write_csv(docpredictions_out)

@registercomponent
class NFoldCV(WorkflowComponent):
    
    vectors = Parameter()
    labels = Parameter()
    featurenames = Parameter()

    n = IntParameter(default=10)
    classifier = Parameter(default='naive_bayes')
    documents = Parameter(default=False)

    def accepts(self):
        return [ ( InputFormat(self,format_id='vectors',extension='.vectors.npz',inputparameter='vectors'), InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), InputFormat(self, format_id='featurenames', extension='.featurenames.txt', inputparameter='featurenames') ) ]
    
    def setup(self, workflow, input_feeds):

        nfold_cv = workflow.new_task('nfold_cv', Run_nfold_cv, autopass=True, n = self.n, classifier=self.classifier, documents=self.documents)
        nfold_cv.in_vectors = input_feeds['vectors']
        nfold_cv.in_labels = input_feeds['labels']
        nfold_cv.in_featurenames = input_feeds['featurenames']

        return nfold_cv


