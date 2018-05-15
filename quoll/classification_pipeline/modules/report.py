
import numpy
from scipy import sparse
import glob
from collections import defaultdict

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter, FloatParameter

from quoll.classification_pipeline.modules.classify import Train, Predict, VectorizeTrainTask, VectorizeTestTask
from quoll.classification_pipeline.modules.vectorize import Vectorize, VectorizeCsv, FeaturizeTask

from quoll.classification_pipeline.functions import reporter, nfold_cv_functions, linewriter, docreader


#################################################################
### Tasks #######################################################
#################################################################

class ReportPerformance(Task):

    in_predictions = InputSlot()
    in_testlabels = InputSlot()
    in_testdocuments = InputSlot()

    ordinal = BoolParameter()

    def in_full_predictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.full_predictions.txt')

    def out_report(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report')

    def out_performance(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report/performance.csv')

    def out_performance_at_dir(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report/performance_at_dir')

    def out_docpredictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report/docpredictions.csv')

    def out_confusionmatrix(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report/confusion_matrix.csv')

    def out_fps_dir(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report/ranked_fps')

    def out_tps_dir(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report/ranked_tps')

    def out_fns_dir(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report/ranked_fns')

    def out_tns_dir(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report/ranked_tns')

    def run(self):

        # setup reporter output directory
        self.setup_output_dir(self.out_report().path)
        self.setup_output_dir(self.out_performance_at_dir().path)
        
        # load predictions and full_predictions
        with open(self.in_predictions().path) as infile:
            predictions = infile.read().strip().split('\n')

        with open(self.in_full_predictions().path) as infile:
            lines = [line.split('\t') for line in infile.read().strip().split('\n')]
        label_order = lines[0]
        full_predictions = lines[1:]

        # load testlabels
        with open(self.in_testlabels().path) as infile:
            testlabels = infile.read().strip().split('\n')

        # load documents
        with open(self.in_testdocuments().path,'r',encoding='utf-8') as infile:
            testdocuments = infile.read().strip().split('\n')

        # initiate reporter
        rp = reporter.Reporter(predictions, full_predictions, label_order, testlabels, self.ordinal, testdocuments)

        # report performance
        if self.ordinal:
            performance = rp.assess_ordinal_performance()
        else:
            performance = rp.assess_performance()
        lw = linewriter.Linewriter(performance)
        lw.write_csv(self.out_performance().path)

        # report predictions by document
        predictions_by_document = rp.predictions_by_document()
        lw = linewriter.Linewriter(predictions_by_document)
        lw.write_csv(self.out_docpredictions().path)

        # report fps per label
        self.setup_output_dir(self.out_fps_dir().path)
        for label in label_order:
            ranked_fps = rp.return_ranked_fps(label)
            outfile = self.out_fps_dir().path + '/' + label + '.csv'
            lw = linewriter.Linewriter(ranked_fps)
            lw.write_csv(outfile)

        # report fps per label
        self.setup_output_dir(self.out_tps_dir().path)
        for label in label_order:
            ranked_tps = rp.return_ranked_tps(label)
            outfile = self.out_tps_dir().path + '/' + label + '.csv'
            lw = linewriter.Linewriter(ranked_tps)
            lw.write_csv(outfile)

        # report fns per label
        self.setup_output_dir(self.out_fns_dir().path)
        for label in label_order:
            ranked_fns = rp.return_ranked_fns(label)
            outfile = self.out_fns_dir().path + '/' + label + '.csv'
            lw = linewriter.Linewriter(ranked_fns)
            lw.write_csv(outfile)

        # report tns per label
        self.setup_output_dir(self.out_tns_dir().path)
        for label in label_order:
            ranked_tns = rp.return_ranked_tns(label)
            outfile = self.out_tns_dir().path + '/' + label + '.csv'
            lw = linewriter.Linewriter(ranked_tns)
            lw.write_csv(outfile)

        # report confusion matrix
        if self.ordinal: # to make a confusion matrix, the labels should be formatted as string
            rp = reporter.Reporter(predictions, full_predictions, label_order, testlabels, False, testdocuments)
        confusion_matrix = rp.return_confusion_matrix()
        with open(self.out_confusionmatrix().path,'w') as cm_out:
            cm_out.write(confusion_matrix)

        # report performance-at
        if len(label_order) >= 9:
            prat_opts = [3,5,7,9]
        elif len(label_order) >= 7:
            prat_opts = [3,5,7]
        elif len(label_order) >= 5:
            prat_opts = [3,5]
        elif len(label_order) >= 3:
            prat_opts = [3]
        else:
            prat_opts = []
        for po in prat_opts:
            outfile = self.out_performance_at_dir().path + '/performance_at_' + str(po) + '.txt'
            rp = reporter.Reporter(predictions, full_predictions, label_order, testlabels, self.ordinal, testdocuments, po)
            if self.ordinal:
                performance = rp.assess_ordinal_performance()
            else:
                performance = rp.assess_performance()
            lw = linewriter.Linewriter(performance)
            lw.write_csv(outfile)

            
class ReportDocpredictions(Task):

    in_predictions = InputSlot()
    in_testdocuments = InputSlot()

    def in_full_predictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.full_predictions.txt')

    def out_docpredictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.docpredictions.csv')

    def run(self):

        # load predictions and full_predictions
        with open(self.in_predictions().path) as infile:
            predictions = infile.read().strip().split('\n')

        with open(self.in_full_predictions().path) as infile:
            lines = [line.split('\t') for line in infile.read().strip().split('\n')]
        label_order = lines[0]
        full_predictions = lines[1:]
        
        # load documents
        with open(self.in_testdocuments().path,'r',encoding='utf-8') as infile:
            testdocuments = infile.read().strip().split('\n')

        # initiate reporter
        rp = reporter.Reporter(predictions, full_predictions, label_order, documents=testdocuments)

        # report predictions by document
        predictions_by_document = rp.predictions_by_document()
        lw = linewriter.Linewriter(predictions_by_document)
        lw.write_csv(self.out_docpredictions().path)


#################################################################
### Nfold CV Tasks #######################################################
#################################################################


class ReportFolds(Task):

    in_exp = InputSlot()

    def out_predictions(self):
        return self.outputfrominput(inputformat='exp', stripextension='.exp', addextension='.predictions.txt')  

    def out_labels(self):
        return self.outputfrominput(inputformat='exp', stripextension='.exp', addextension='.gs.labels')  

    def out_full_predictions(self):
        return self.outputfrominput(inputformat='exp', stripextension='.exp', addextension='.full_predictions.txt')        

    def out_macro_performance(self):
        return self.outputfrominput(inputformat='exp', stripextension='.exp', addextension='.macro_performance.csv')  
 
    def run(self):
       
        # gather fold reports
        print('gathering fold reports')
        performance_files = [ filename for filename in glob.glob(self.in_exp().path + '/fold*/*.report/performance.csv') ]
        docprediction_files = [ filename for filename in glob.glob(self.in_exp().path + '/fold*/*.report/docpredictions.csv') ]
        
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
        lw.write_csv(self.out_macro_performance().path)

        # write labels, predictions and full predictions
        label_order = [x.split('prediction prob for ')[1] for x in dr.parse_csv(docprediction_files[0])[0][3:]]
        docpredictions = sum([dr.parse_csv(docprediction_file)[1:] for docprediction_file in docprediction_files], [])
        labels = [line[1] for line in docpredictions]
        predictions = [line[2] for line in docpredictions]
        full_predictions = [label_order] + [line[3:] for line in docpredictions]

        with open(self.out_labels().path,'w',encoding='utf-8') as l_out:
            l_out.write('\n'.join(labels))

        with open(self.out_predictions().path,'w',encoding='utf-8') as pr_out:
            pr_out.write('\n'.join(predictions))

        with open(self.out_full_predictions().path,'w',encoding='utf-8') as fpr_out:
            fpr_out.write('\n'.join(['\t'.join([str(prob) for prob in full_prediction]) for full_prediction in full_predictions]))


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
    
    nb_alpha = Parameter(default=1.0)
    nb_fit_prior = BoolParameter()
    
    svm_c = Parameter(default=1.0)
    svm_kernel = Parameter(default='linear')
    svm_gamma = Parameter(default='0.1')
    svm_degree = Parameter(default='1')
    svm_class_weight = Parameter(default='balanced')
    
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
    
    nb_alpha = Parameter()
    nb_fit_prior = BoolParameter()
    
    svm_c = Parameter()
    svm_kernel = Parameter()
    svm_gamma = Parameter()
    svm_degree = Parameter()
    svm_class_weight = Parameter()
    
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
    
    nb_alpha = Parameter(default='1.0')
    nb_fit_prior = BoolParameter()
    
    svm_c = Parameter(default='1.0')
    svm_kernel = Parameter(default='linear')
    svm_gamma = Parameter(default='0.1')
    svm_degree = Parameter(default='1')
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
class Report(WorkflowComponent):

    train = Parameter() # only train for nfold cv
    test = Parameter(default = 'xxx.xxx')
    trainlabels = Parameter()
    testlabels = Parameter(default = 'xxx.xxx')
    docs = Parameter(default = 'xxx.xxx') # all docs for nfold cv, test docs for train and test

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
    
    nb_alpha = Parameter(default='1.0')
    nb_fit_prior = BoolParameter()
    
    svm_c = Parameter(default='1.0')
    svm_kernel = Parameter(default='linear')
    svm_gamma = Parameter(default='0.1')
    svm_degree = Parameter(default='1')
    svm_class_weight = Parameter(default='balanced')
    
    # vectorizer parameters
    weight = Parameter(default = 'frequency') # options: frequency, binary, tfidf
    prune = IntParameter(default = 5000) # after ranking the topfeatures in the training set, based on frequency or idf weighting
    balance = BoolParameter()
    delimiter = Parameter(default=',')

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
                InputFormat(self, format_id='modeled_train',extension ='.model.pkl',inputparameter='train'),
                InputFormat(self, format_id='vectorized_train',extension='.vectors.npz',inputparameter='train'),
                InputFormat(self, format_id='featurized_train',extension='.features.npz',inputparameter='train'),
                InputFormat(self, format_id='featurized_csv_train',extension='.csv',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txt',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txtdir',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.json',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.jsondir',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.txtdir',inputparameter='train'),
                InputFormat(self, format_id='docs_train',extension='.txt',inputparameter='train')
                ),
                (
                InputFormat(self, format_id='classified_test',extension='.predictions.txt',inputparameter='test'),
                InputFormat(self, format_id='vectorized_test',extension='.vectors.npz',inputparameter='test'),
                InputFormat(self, format_id='featurized_test',extension='.features.npz',inputparameter='test'),
                InputFormat(self, format_id='featurized_csv_test',extension='.csv',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.tok.txt',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.tok.txtdir',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.frog.json',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.frog.jsondir',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.txtdir',inputparameter='test'),
                InputFormat(self, format_id='docs_test',extension='.txt',inputparameter='test')
                ),
                (
                InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels')
                ),
                (
                InputFormat(self, format_id='labels_test',extension='.labels',inputparameter='testlabels')
                ),
                (
                InputFormat(self, format_id='docs',extension='.txt',inputparameter='docs')
                )
            ]
            )).T.reshape(-1,5)]

    def setup(self, workflow, input_feeds):
        
        ######################
        ### Training phase ###
        ######################

        trainlabels = input_feeds['labels_train']

        # make sure to have featurized train instances, needed for both the nfold-cv case and the train-test case

        featurized_train = False
        trainvectors = False
        
        if 'docs_train' in input_feeds.keys() or 'pre_featurized_train' in input_feeds.keys():

            if 'pre_featurized_train' in input_feeds.keys():
                pre_featurized = input_feeds['pre_featurized_train']

            else: # docs (.txt)
                docs = input_feeds['docs_train']
                pre_featurized = input_feeds['docs_train']

            trainfeaturizer = workflow.new_task('featurize_train',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
            trainfeaturizer.in_pre_featurized = pre_featurized

            featurized_train = trainfeaturizer.out_featurized

        elif 'featurized_train' in input_feeds.keys(): 
            featurized_train = input_feeds['featurized_train']

        elif 'featurized_csv_train' in input_feeds.keys():
            trainvectorizer = workflow.new_task('vectorize_train_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter)
            trainvectorizer.in_csv = input_feeds['featurized_csv_train']

            trainvectors = trainvectorizer.out_vectors

        elif 'vectorized_train' in input_feeds.keys():
            trainvectors = input_feeds['vectorized_train']

        if not 'test' in [x.split('_')[-1] for x in input_feeds.keys()]: # only train input --> nfold-cv

        ######################
        ### Nfold CV #########
        ######################

            if 'docs' in input_feeds.keys():
                docs = input_feeds['docs']

            if featurized_train:
                instances = featurized_train

            elif trainvectors:
                instances = trainvectors

            else:
                print('Invalid \'train\' input for Nfold CV; ' + 
                    'give either \'.txt\', \'.tok.txt\', \'frog.json\', \'.txtdir\', \'.tok.txtdir\', \'.frog.jsondir\', \'.features.npz\', \'.vectors.npz\' or \'.csv\'')
                exit()

            bin_maker = workflow.new_task('make_bins', MakeBins, autopass=True, n=self.n, teststart=self.teststart, testend=self.testend)
            bin_maker.in_labels = trainlabels

            fold_runner = workflow.new_task('nfold_cv', Folds, autopass=True, 
                n=self.n, 
                weight=self.weight, prune=self.prune, balance=self.balance, 
                classifier=self.classifier, ordinal=self.ordinal, jobs=self.jobs, iterations=self.iterations,
                nb_alpha=self.nb_alpha, nb_fit_prior=self.nb_fit_prior,
                svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight
            )
            fold_runner.in_bins = bin_maker.out_bins
            fold_runner.in_instances = instances
            fold_runner.in_labels = trainlabels
            fold_runner.in_docs = docs      

            foldreporter = workflow.new_task('report_folds', ReportFolds, autopass=True)
            foldreporter.in_exp = fold_runner.out_exp

            labels = foldreporter.out_labels
            predictions = foldreporter.out_predictions

        else:

            if 'modeled_train' in input_feeds.keys():
                trainmodel = input_feeds['modeled_train']

            else:

                if not trainvectors:

                    if featurized_train:

                        trainvectorizer = workflow.new_task('vectorize_train',VectorizeTrainTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
                        trainvectorizer.in_trainfeatures = featurized_train
                        trainvectorizer.in_trainlabels = trainlabels
            
                        trainvectors = trainvectorizer.out_train
                        trainlabels = trainvectorizer.out_trainlabels              

                    else:
                        print('Invalid train input, exiting program')
                        exit()

                trainer = workflow.new_task('train',Train,autopass=True,classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,
                    nb_alpha=self.nb_alpha,nb_fit_prior=self.nb_fit_prior,
                    svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight)
                trainer.in_train = trainvectors
                trainer.in_trainlabels = trainlabels    

                trainmodel = trainer.out_model 

            if 'classified_test' in input_feeds.keys(): # reporter can be started
                predictions = input_feeds['classified_test']

            else: 

        ######################
        ### Testing phase ####
        ######################

                if 'vectorized_test' in input_feeds.keys():
                    testvectors = input_feeds['vectorized_test']

                elif 'featurized_csv_test' in input_feeds.keys():
                    testvectorizer = workflow.new_task('vectorize_test_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter)
                    testvectorizer.in_csv = input_feeds['featurized_csv_test']

                    testvectors = testvectorizer.out_vectors

                else:
                    
                    if 'docs_test' in input_feeds.keys() or 'pre_featurized_test' in input_feeds.keys():

                        if 'pre_featurized_test' in input_feeds.keys():
                            pre_featurized = input_feeds['pre_featurized_test']

                        else:
                            docs = input_feeds['docs_test']
                            pre_featurized = input_feeds['docs_test']

                        testfeaturizer = workflow.new_task('featurize_test',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                        testfeaturizer.in_pre_featurized = pre_featurized

                        featurized_test = testfeaturizer.out_featurized

                    else:
                        featurized_test = input_feeds['featurized_test']

                    testvectorizer = workflow.new_task('vectorize_test',VectorizeTestTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
                    testvectorizer.in_trainvectors = trainvectors
                    testvectorizer.in_trainlabels = trainlabels
                    testvectorizer.in_testfeatures = featurized_test

                    testvectors = testvectorizer.out_vectors

                predictor = workflow.new_task('predictor',Predict,autopass=True,classifier=self.classifier,ordinal=self.ordinal)
                predictor.in_test = testvectors
                predictor.in_trainlabels = trainlabels
                predictor.in_model = trainmodel

                predictions = predictor.out_predictions

            if 'docs' in input_feeds.keys():
                docs = input_feeds['docs']
                
            if 'labels_test' in input_feeds.keys(): # full performance reporter
                labels = input_feeds['labels_test']

            else:
                labels = False

        ######################
        ### Reporting phase ##
        ######################

        if labels:

            reporter = workflow.new_task('report_performance',ReportPerformance,autopass=True,ordinal=self.ordinal)
            reporter.in_predictions = predictions
            reporter.in_testlabels = labels
            reporter.in_testdocuments = docs

        else: # report docpredictions

            reporter = workflow.new_task('report_docpredictions',ReportDocpredictions,autopass=True)
            reporter.in_predictions = predictions
            reporter.in_testdocuments = docs

        return reporter
