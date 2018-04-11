
from luiginlp.engine import Task, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, IntParameter, BoolParameter
import numpy
from scipy import sparse
from collections import defaultdict
import glob

import quoll.classification_pipeline.functions.nfold_cv_functions as nfold_cv_functions
import quoll.classification_pipeline.functions.linewriter as linewriter
import quoll.classification_pipeline.functions.docreader as docreader

from quoll.classification_pipeline.modules.run_hierarchical_experiment import HierarchicalExperimentComponent
from quoll.classification_pipeline.modules.make_bins import MakeBins 


################################################################################
###Component to thread the tasks together
################################################################################

@registercomponent
class NFoldCVSparseHierarchical(WorkflowComponent):

    features = Parameter()
    labels_layer1 = Parameter()
    labels_layer2 = Parameter()
    vocabulary = Parameter()
    documents = Parameter()
    classifier_args = Parameter()

    n = IntParameter(default=10)
    steps = IntParameter(default=1) # useful to increase if closeby instances, for example sets of 2, are dependent
    teststart = IntParameter(default=0) # give the first index to indicate a range to fix these instances for testing (some instances are then always used in training and not in the testset; this is useful when you want to compare the addition of a certain dataset to train on)
    testend = IntParameter(default=-1)
    weight = Parameter(default='frequency')
    prune = IntParameter(default=5000)
    balance = BoolParameter(default=False)
    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter(default=False)
    no_label_encoding = BoolParameter(default=False)
    pca = BoolParameter(default=False)
    
    def accepts(self):
        return [ ( InputFormat(self,format_id='features',extension='.features.npz',inputparameter='features'), InputFormat(self, format_id='labels_layer1', extension='.labels', inputparameter='labels_layer1'), InputFormat(self,format_id='labels_layer2', extension='.labels', inputparameter='labels_layer2') InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self, format_id='vocabulary', extension='.vocabulary.txt', inputparameter='vocabulary'), InputFormat(self,format_id='classifier_args',extension='.txt',inputparameter='classifier_args') ) ]

    def setup(self, workflow, input_feeds):

        bin_maker = workflow.new_task('make_bins', MakeBins, autopass=True, n=self.n, teststart=self.teststart, testend=self.testend)
        bin_maker.in_labels = input_feeds['labels_layer1']

        fold_runner = workflow.new_task('nfold_cv_hierarchical', RunHierarchicalFolds, autopass=True, n = self.n, weight=self.weight, prune=self.prune, balance=self.balance, classifier=self.classifier, ordinal=self.ordinal, pca=self.pca, no_label_encoding=self.no_label_encoding)
        fold_runner.in_bins = bin_maker.out_bins
        fold_runner.in_features = input_feeds['features']
        fold_runner.in_labels_layer1 = input_feeds['labels_layer1']
        fold_runner.in_labels_layer2 = input_feeds['labels_layer2']
        fold_runner.in_vocabulary = input_feeds['vocabulary']
        fold_runner.in_documents = input_feeds['documents']        
        fold_runner.in_classifier_args = input_feeds['classifier_args']

        folds_reporter = workflow.new_task('report_folds_hierarchical', ReportFoldsHierarchical, autopass = False)
        folds_reporter.in_expdirectory = fold_runner.out_exp

        return folds_reporter


################################################################################
###Experiment wrapper
################################################################################

class RunHierarchicalFolds(Task):

    in_bins = InputSlot()
    in_features = InputSlot()
    in_labels_layer1 = InputSlot()
    in_labels_layer2 = InputSlot()
    in_vocabulary = InputSlot()
    in_documents = InputSlot()
    in_classifier_args = InputSlot()

    n = IntParameter()
    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()
    classifier = Parameter()
    ordinal = BoolParameter()
    pca = BoolParameter()
    no_label_encoding = BoolParameter()
    
    def out_exp(self):
        return self.outputfrominput(inputformat='bins', stripextension='.bins.csv', addextension='.hierarchical.' + self.classifier + '.args_' + self.in_classifier_args().path.split('/')[-1][:-4] + '.weight_' + self.weight + '.prune_' + str(self.prune) + '.balance_' + self.balance.__str__() + '.pca_' + self.pca.__str__() + '.exp')
        
    def run(self):

        # make experiment directory
        self.setup_output_dir(self.out_exp().path)

        # for each fold
        for fold in range(self.n):
            yield HierarchicalFold(directory=self.out_exp().path, features=self.in_features().path, labels_layer1=self.in_labels_layer1().path, labels_layer2=self.in_labels_layer2().path, vocabulary=self.in_vocabulary().path, bins=self.in_bins().path, documents=self.in_documents().path, classifier_args=self.in_classifier_args().path, i=fold, weight=self.weight, prune=self.prune, balance=self.balance, classifier=self.classifier, ordinal=self.ordinal, pca=self.pca, no_label_encoding=self.no_label_encoding)


################################################################################
###Fold Wrapper
################################################################################

@registercomponent
class HierarchicalFold(WorkflowComponent):

    directory = Parameter()
    features = Parameter()
    labels_layer1 = Parameter()
    labels_layer2 = Parameter()
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
    pca = BoolParameter()
    no_label_encoding = BoolParameter()
    
    def accepts(self):
        return [ ( InputFormat(self,format_id='directory',extension='.exp',inputparameter='directory'), InputFormat(self,format_id='features',extension='.features.npz',inputparameter='features'), InputFormat(self, format_id='labels_layer1', extension='.labels', inputparameter='labels_layer1'), InputFormat(self,format_id='labels_layer2', extension='labels', inputparameter='labels_layer2'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self, format_id='vocabulary', extension='.vocabulary.txt', inputparameter='vocabulary'), InputFormat(self,format_id='classifier_args',extension='.txt',inputparameter='classifier_args'), InputFormat(self,format_id='bins',extension='.bins.csv',inputparameter='bins') ) ]
 
    def setup(self, workflow, input_feeds):

        hfold = workflow.new_task('run_hierarchical_fold', HierarchicalFoldTask, autopass=False, i=self.i, classifier=self.classifier, weight=self.weight, prune=self.prune, balance=self.balance, ordinal=self.ordinal, pca=self.pca, no_label_encoding=self.no_label_encoding)
        hfold.in_directory = input_feeds['directory']
        hfold.in_features = input_feeds['features']
        hfold.in_vocabulary = input_feeds['vocabulary']
        hfold.in_labels_layer1 = input_feeds['labels_layer1']
        hfold.in_labels_layer2 = input_feeds['labels_layer2']
        hfold.in_documents = input_feeds['documents']
        hfold.in_classifier_args = input_feeds['classifier_args']  
        hfold.in_bins = input_feeds['bins']   

        return hfold

class HierarchicalFoldTask(Task):

    in_directory = InputSlot()
    in_bins = InputSlot()
    in_features = InputSlot()
    in_labels_layer1 = InputSlot()
    in_labels_layer2 = InputSlot()
    in_vocabulary = InputSlot()
    in_documents = InputSlot()
    in_classifier_args = InputSlot()
    
    i = IntParameter()
    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()
    classifier = Parameter()
    ordinal = BoolParameter()
    pca = BoolParameter()
    no_label_encoding = BoolParameter()

    def out_fold(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i))    

    def out_trainfeatures(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.features.npz')        

    def out_testfeatures(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.features.npz')

    def out_trainlabels_layer1(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.layer1.labels')

    def out_trainlabels_layer2(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.layer2.labels')

    def out_testlabels_layer1(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.layer1.labels')

    def out_testlabels_layer2(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.layer2.labels')

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
        bin_range = list(set(sum(bins,[])))
        bin_min = min(bin_range)
        bin_max = max(bin_range)

        # open features
        loader = numpy.load(self.in_features().path)
        featurized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # if applicable, set fixed train indices
        fixed_train_indices = []
        if bin_min > 0:
            fixed_train_indices.extend(list(range(bin_min)))
        if (bin_max+1) < featurized_instances.shape[0]:
            fixed_train_indices.extend(list(range(bin_max+1,featurized_instances.shape[0])))

        # open first layer labels
        with open(self.in_labels_layer1().path,'r',encoding='utf-8') as infile:
            labels_layer1 = numpy.array(infile.read().strip().split('\n'))

        # open second layer labels
        with open(self.in_labels_layer2().path,'r',encoding='utf-8') as infile:
            labels_layer2 = numpy.array(infile.read().strip().split('\n'))

        # open documents
        with open(self.in_documents().path,'r',encoding='utf-8') as infile:
            documents = numpy.array(infile.read().strip().split('\n'))

        # open classifier args
        with open(self.in_classifier_args().path) as infile:
            classifier_args = infile.read().rstrip().split('\n')

        # set training and test data
        train_features = sparse.vstack([featurized_instances[indices,:] for j,indices in enumerate(bins) if j != self.i] + [featurized_instances[fixed_train_indices,:]])
        train_labels_layer1 = numpy.concatenate([labels_layer1[indices] for j,indices in enumerate(bins) if j != self.i] + [labels_layer1[fixed_train_indices]])
        train_labels_layer2 = numpy.concatenate([labels_layer2[indices] for j,indices in enumerate(bins) if j != self.i] + [labels_layer2[fixed_train_indices]])
        train_documents = numpy.concatenate([documents[indices] for j,indices in enumerate(bins) if j != self.i] + [documents[fixed_train_indices]])
        test_features = featurized_instances[bins[self.i]]
        test_labels_layer1 = labels_layer1[bins[self.i]]
        test_labels_layer2 = labels_layer2[bins[self.i]]
        test_documents = documents[bins[self.i]]

        # write experiment data to files in fold directory
        numpy.savez(self.out_trainfeatures().path, data=train_features.data, indices=train_features.indices, indptr=train_features.indptr, shape=train_features.shape)
        numpy.savez(self.out_testfeatures().path, data=test_features.data, indices=test_features.indices, indptr=test_features.indptr, shape=test_features.shape)
        with open(self.out_trainlabels_layer1().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(train_labels_layer1))
        with open(self.out_trainlabels_layer2().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(train_labels_layer2))
        with open(self.out_testlabels_layer1().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_labels_layer1))
        with open(self.out_testlabels_layer2().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_labels_layer2))
        with open(self.out_testdocuments().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_documents))
        with open(self.out_classifier_args().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(classifier_args))

        print('Running experiment for fold',self.i)

        yield HierarchicalExperimentComponent(trainfeatures=self.out_trainfeatures().path, trainlabels_layer1=self.out_trainlabels_layer1().path, trainlabels_layer2=self.out_trainlabels_layer2().path, testfeatures=self.out_testfeatures().path, testlabels_layer1=self.out_testlabels_layer1().path, testlabels_layer2=self.out_testlabels_layer2().path, trainvocabulary=self.in_vocabulary().path, testvocabulary=self.in_vocabulary().path, classifier_args=self.out_classifier_args().path, documents=self.out_testdocuments().path, weight=self.weight, prune=self.prune, balance=self.balance, classifier=self.classifier, ordinal=self.ordinal, pca=self.pca, no_label_encoding=self.no_label_encoding)

################################################################################
###Reporter
################################################################################

@registercomponent
class ReportHierarchicalFoldsComponent(StandardWorkflowComponent):

    def accepts(self):
        return InputFormat(self, format_id='expdirectory', extension='.exp')
                    
    def autosetup(self):
        return ReportFolds

class ReportHierarchicalFolds(Task):

    in_expdirectory = InputSlot()

    ordinal = BoolParameter()

    def out_report(self):
        return self.outputfrominput(inputformat='expdirectory', stripextension='.exp', addextension='.report')  
 
    def run(self):

        # make report directory
        self.setup_output_dir(self.out_report().path)
        
        # gather fold reports
        performance_files = [ filename for filename in glob.glob(self.in_expdirectory().path + '/fold*/*.performance.csv') ]
        docprediction_files = [ filename for filename in glob.glob(self.in_expdirectory().path + '/fold*/*.docpredictions.csv') ]

        # gather fold reports layer 2
        print('gathering fold reports for layer 2')
        performance_files_layer2 = [ filename for filename in glob.glob(self.in_expdirectory().path + '/fold*/*.exp_second_layer.performance.csv') ]
        docprediction_files_layer2 = [ filename for filename in glob.glob(self.in_expdirectory().path + '/fold*/*.exp_second_layer.docpredictions.csv') ]

        # gather fold reports layer 1
        print('gathering fold reports for layer 1')
        performance_files_layer1 = list(set(performance_files) - set(performance_files_layer2))
        docprediction_files_layer1 = list(set(docprediction_files) - set(docprediction_files_layer2))

        ################
        ### Layer 1
        ################

        # calculate average performance
        dr = docreader.Docreader()
        performance_combined = [dr.parse_csv(performance_file) for performance_file in performance_files_layer1]
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

        # compute mean and sum per label for layer 1
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
        lw.write_csv(self.out_report().path + '/macro_performance.layer1.csv')

        # write predictions per document
        label_order = [x.split('prediction prob for ')[1] for x in dr.parse_csv(docprediction_files_layer1[0])[0][3:]]
        docpredictions = sum([dr.parse_csv(docprediction_file)[1:] for docprediction_file in docprediction_files_layer1], [])
        documents = [line[0] for line in docpredictions]
        labels = [line[1] for line in docpredictions]
        unique_labels = labels_order
        predictions = [line[2] for line in docpredictions]
        full_predictions = [line[3:] for line in docpredictions]

        # initiate reporter
        rp = reporter.Reporter(predictions, full_predictions, label_order, labels, unique_labels, self.ordinal, documents)
        predictions_by_document = rp.predictions_by_document()
        lw = linewriter.Linewriter(predictions_by_document)
        lw.write_csv(self.out_report().path + '/docpredictions.layer1.csv')

        # report performance
        if self.ordinal:
            performance = rp.assess_ordinal_performance()
        else:
            performance = rp.assess_performance()
        lw = linewriter.Linewriter(performance)
        lw.write_csv(self.out_report().path + '/performance.layer1.csv')

        # report fps per label
        self.setup_output_dir(self.out_report().path + '/ranked_fps.layer1')
        for label in unique_labels:
            try:
                ranked_fps = rp.return_ranked_fps(label)
                outfile = self.out_report().path + '/ranked_fps.layer1/' + label + '.csv'
                lw = linewriter.Linewriter(ranked_fps)
                lw.write_csv(outfile)
            except:
                continue

        # report tps per label
        self.setup_output_dir(self.out_report().path + '/ranked_tps.layer1')
        for label in unique_labels:
            try:
                ranked_tps = rp.return_ranked_tps(label)
                outfile = self.out_report().path + '/ranked_tps.layer1/' + label + '.csv'
                lw = linewriter.Linewriter(ranked_tps)
                lw.write_csv(outfile)
            except:
                continue
            
        # report fns per label
        self.setup_output_dir(self.out_report().path + '/ranked_fns.layer1')
        for label in unique_labels:
            try:
                ranked_fns = rp.return_ranked_fns(label)
                outfile = self.out_report().path + '/ranked_fns.layer1/' + label + '.csv'
                lw = linewriter.Linewriter(ranked_fns)
                lw.write_csv(outfile)
            except:
                continue
                
        # report tns per label
        self.setup_output_dir(self.out_report().path + '/ranked_tns.layer1')
        for label in unique_labels:
            try:
                ranked_tns = rp.return_ranked_tns(label)
                outfile = self.out_report().path + '/ranked_tns.layer1/' + label + '.csv'
                lw = linewriter.Linewriter(ranked_tns)
                lw.write_csv(outfile)
            except:
                continue
                
        # report confusion matrix
        if self.ordinal: # to make a confusion matrix, the labels should be formatted as string
            rp = reporter.Reporter(predictions, full_predictions, label_order, labels, unique_labels, False, documents)
        confusion_matrix = rp.return_confusion_matrix()
        with open(self.out_report().path + '/confusion_matrix.layer1.csv','w') as cm_out:
            cm_out.write(confusion_matrix)

        ################
        ### Layer 2
        ################

        # calculate average performance
        dr = docreader.Docreader()
        performance_combined = [dr.parse_csv(performance_file) for performance_file in performance_files_layer2]
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

        # compute mean and sum per label for layer 2
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
        lw.write_csv(self.out_report().path + '/macro_performance.layer2.csv')

        # write predictions per document
        label_order = [x.split('prediction prob for ')[1] for x in dr.parse_csv(docprediction_files_layer2[0])[0][3:]]
        docpredictions = sum([dr.parse_csv(docprediction_file)[1:] for docprediction_file in docprediction_files_layer2], [])
        documents = [line[0] for line in docpredictions]
        labels = [line[1] for line in docpredictions]
        unique_labels = labels_order
        predictions = [line[2] for line in docpredictions]
        full_predictions = [line[3:] for line in docpredictions]

        # initiate reporter
        rp = reporter.Reporter(predictions, full_predictions, label_order, labels, unique_labels, self.ordinal, documents)
        predictions_by_document = rp.predictions_by_document()
        lw = linewriter.Linewriter(predictions_by_document)
        lw.write_csv(self.out_report().path + '/docpredictions.layer2.csv')

        # report performance
        if self.ordinal:
            performance = rp.assess_ordinal_performance()
        else:
            performance = rp.assess_performance()
        lw = linewriter.Linewriter(performance)
        lw.write_csv(self.out_report().path + '/performance.layer2.csv')

        # report fps per label
        self.setup_output_dir(self.out_report().path + '/ranked_fps.layer2')
        for label in unique_labels:
            try:
                ranked_fps = rp.return_ranked_fps(label)
                outfile = self.out_report().path + '/ranked_fps.layer2/' + label + '.csv'
                lw = linewriter.Linewriter(ranked_fps)
                lw.write_csv(outfile)
            except:
                continue

        # report tps per label
        self.setup_output_dir(self.out_report().path + '/ranked_tps.layer2')
        for label in unique_labels:
            try:
                ranked_tps = rp.return_ranked_tps(label)
                outfile = self.out_report().path + '/ranked_tps.layer2/' + label + '.csv'
                lw = linewriter.Linewriter(ranked_tps)
                lw.write_csv(outfile)
            except:
                continue
            
        # report fns per label
        self.setup_output_dir(self.out_report().path + '/ranked_fns.layer2')
        for label in unique_labels:
            try:
                ranked_fns = rp.return_ranked_fns(label)
                outfile = self.out_report().path + '/ranked_fns.layer2/' + label + '.csv'
                lw = linewriter.Linewriter(ranked_fns)
                lw.write_csv(outfile)
            except:
                continue
                
        # report tns per label
        self.setup_output_dir(self.out_report().path + '/ranked_tns.layer2')
        for label in unique_labels:
            try:
                ranked_tns = rp.return_ranked_tns(label)
                outfile = self.out_report().path + '/ranked_tns.layer2/' + label + '.csv'
                lw = linewriter.Linewriter(ranked_tns)
                lw.write_csv(outfile)
            except:
                continue
                
        # report confusion matrix
        if self.ordinal: # to make a confusion matrix, the labels should be formatted as string
            rp = reporter.Reporter(predictions, full_predictions, label_order, labels, unique_labels, False, documents)
        confusion_matrix = rp.return_confusion_matrix()
        with open(self.out_report().path + '/confusion_matrix.layer2.csv','w') as cm_out:
            cm_out.write(confusion_matrix)

