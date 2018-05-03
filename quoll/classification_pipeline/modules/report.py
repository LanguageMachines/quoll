
import numpy
from scipy import sparse
from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.functions import reporter, linewriter


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
            documents = infile.read().strip().split('\n')

        # initiate reporter
        rp = reporter.Reporter(predictions, full_predictions, label_order, documents=testdocuments)

        # report predictions by document
        predictions_by_document = rp.predictions_by_document()
        lw = linewriter.Linewriter(predictions_by_document)
        lw.write_csv(self.out_docpredictions().path)


#################################################################
### Component ##################################################
#################################################################

@registercomponent
class Report(WorkflowComponent):

    train = Parameter(default = 'xxx.xxx') # only train for nfold cv
    test = Parameter(default = 'xxx.xxx')
    trainlabels = Parameter()
    testlabels = Parameter(default = 'xxx.xxx')
    docs = Parameter(default = 'xxx.xxx') # all docs for nfold cv, test docs for train and test

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
                InputFormat(self, format_id='docs_train',extension='.txt',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.txtdir',inputparameter='train')
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
                InputFormat(self, format_id='docs_test',extension='.txt',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.txtdir',inputparameter='test')
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

        trainlabels = input_feeds['trainlabels']

        if 'test' in [x.split('_')[-1] for x in input_feeds.keys()]: # work towards reporting testpredictions

            if 'classified_test' in input_feeds.keys(): # reporter can be started
                testpredictions = input_feeds['classified_test']

            else: # need classifier, running train pipeline

                ######################
                ### Training phase ###
                ######################

                if 'modeled_train' in input_feeds.keys():
                    trainmodel = input_feeds['modeled_train']

                else:

                    if 'vectorized_train' in input_feeds.keys():
                        trainvectors = input_feeds['vectorized_train']

                    elif 'featurized_csv_train' in input_feeds.keys():
                        trainvectorizer = workflow.new_task('vectorize_train_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter)
                        trainvectorizer.in_csv = input_feeds['featurized_csv_train']
                
                        trainvectors = trainvectorizer.out_vectors

                    else:

                        if 'docs_train' in input_feeds.keys() or if 'pre_featurized_train' in input_feeds.keys():

                            if 'pre_featurized_train' in input_feeds.keys():
                                pre_featurized = input_feeds['pre_featurized_train']

                            else:
                                traindocs = input_feeds['docs_train']
                                pre_featurized = input_feeds['docs_train']

                            trainfeaturizer = workflow.new_task('featurize_train',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                            trainfeaturizer.in_pre_featurized = pre_featurized

                            featurized_train = trainfeaturizer.out_featurized

                        else: # can only be featurized train
                            featurized_train = input_feeds['featurized_train']

                        trainvectorizer = workflow.new_task('vectorize_train',VectorizeTrainTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
                        trainvectorizer.in_trainfeatures = featurized_train
                        trainvectorizer.in_trainlabels = trainlabels
                    
                        trainvectors = trainvectorizer.out_train
                        trainlabels = trainvectorizer.out_trainlabels

                    trainer = workflow.new_task('train',Train,autopass=True,classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,nb_alpha=self.nb_alpha,nb_fit_prior=self.nb_fit_prior,svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight)
                    trainer.in_train = trainvectors
                    trainer.in_trainlabels = trainlabels

                    trainmodel = trainer.out_model

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
                    
                    if 'docs_test' in input_feeds.keys() or if 'pre_featurized_test' in input_feeds.keys():

                        if 'pre_featurized_test' in input_feeds.keys():
                            pre_featurized = input_feeds['pre_featurized_test']

                        else:
                            testdocs = input_feeds['docs_test']
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

                testpredictions = predictor.out_predictions


            ######################
            ### Reporting phase ##
            ######################

            if 'docs' in input_feeds.keys():
                testdocs = input_feeds['docs']

            if 'test_labels' in input_feeds.keys(): # full performance reporter

                testlabels = input_feeds['test_labels']

                reporter = workflow.new_task('report_performance',ReportPerformance,autopass=True,ordinal=self.ordinal)
                reporter.in_predictions = testpredictions
                reporter.in_testlabels = testlabels
                reporter.in_testdocs = testdocs

            else: # report docpredictions

                reporter = workflow.new_task('report_docpredictions',ReportDocpredictions,autopass=True)

            return reporter


        #if 'train' in [x.split('_')[-1] for x in input_feeds.keys() if x != 'labels_train']: #


# @registercomponent
# class RegressionReporterComponent(WorkflowComponent):

#     predictions = Parameter()
#     labels = Parameter()

#     def accepts(self):
#         return [ ( InputFormat(self, format_id='predictions', extension='.predictions.txt',inputparameter='predictions'), InputFormat(self, format_id='labels', extension='.txt', inputparameter='labels') ) ]

#     def setup(self, workflow, input_feeds):

#         regreporter = workflow.new_task('report_regression_performance', ReportRegressionPerformance, autopass=True)
#         regreporter.in_predictions = input_feeds['predictions']
#         regreporter.in_labels = input_feeds['labels']

#         return regreporter