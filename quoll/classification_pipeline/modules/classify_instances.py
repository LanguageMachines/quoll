
import os
import numpy
from scipy import sparse
import pickle
import math
from collections import defaultdict

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.functions.classifier import *
from quoll.classification_pipeline.functions.lcs_classifier import LCS_classifier
from quoll.classification_pipeline.functions.decisiontree_continuous import DecisionTreeContinuous

class TrainClassifier(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_vocab = InputSlot()
    in_classifier_args = InputSlot()

    classifier = Parameter()
    no_label_encoding = BoolParameter()
    ordinal = BoolParameter()

    def out_model(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension=self.classifier + '.model.pkl')

    def out_label_encoding(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension=self.classifier + '.le')

    def out_model_insights(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension=self.classifier + '.model_insights')

    def run(self):

        # initiate directory with model insights
        self.setup_output_dir(self.out_model_insights().path)

        # initiate classifier
        # classifierdict = {'naive_bayes':NaiveBayesClassifier(), 'svm':SVMClassifier(), 'tree':TreeClassifier(), 'continuous_tree':DecisionTreeContinuous(), 'perceptron':PerceptronLClassifier(), 'logistic_regression':LogisticRegressionClassifier(), 'lreg':LinearRegressionClassifier(), 'ordinal_ridge':OrdinalRidge(), 'ordinal_at':OrdinalLogisticAT(), 'ordinal_se':OrdinalLogisticSE(), 'ordinal_it':OrdinalLogisticIT()}
        classifierdict = {'naive_bayes':NaiveBayesClassifier(), 'svm':SVMClassifier(), 'tree':TreeClassifier(), 'continuous_tree':DecisionTreeContinuous(), 'perceptron':PerceptronLClassifier(), 'logistic_regression':LogisticRegressionClassifier(), 'lreg':LinearRegressionClassifier()}
        clf = classifierdict[self.classifier]

        # load vectorized instances
        loader = numpy.load(self.in_train().path)
        vectorized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load trainlabels
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')
        if self.ordinal:
            trainlabels = [float(x) for x in trainlabels]

        # load vocabulary
        with open(self.in_vocab().path,'r',encoding='utf-8') as infile:
            vocab = infile.read().strip().split('\n')

        # transform trainlabels
        if not self.no_label_encoding:
            clf.set_label_encoder(sorted(trainlabels))

        # load classifier arguments        
        with open(self.in_classifier_args().path,'r',encoding='utf-8') as infile:
            classifier_args_str = infile.read().rstrip()
            classifier_args = classifier_args_str.split('\n')

        # train classifier
        clf.train_classifier(vectorized_instances, trainlabels, self.no_label_encoding, *classifier_args)

        # save classifier and insights
        model = clf.return_classifier()
        with open(self.out_model().path, 'wb') as fid:
            pickle.dump(model, fid)
        model_insights = clf.return_model_insights(vocab)
        for mi in model_insights:
            with open(self.out_model_insights().path + '/' + mi[0],'w',encoding='utf-8') as outfile:
                outfile.write(mi[1])

        # save label encoding
        if not self.no_label_encoding:
            label_encoding = clf.return_label_encoding(trainlabels)
            with open(self.out_label_encoding().path,'w',encoding='utf-8') as le_out:
                le_out.write('\n'.join([' '.join(le) for le in label_encoding]))
        else:
            with open(self.out_label_encoding().path,'w',encoding='utf-8') as le_out:
                le_out.write('Not applicable')

class ApplyClassifier(Task):

    in_test = InputSlot()
    in_labels = InputSlot()
    in_model = InputSlot()

    classifier = Parameter()
    no_label_encoding = BoolParameter()
    ordinal = BoolParameter()

    def out_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension=self.classifier + '.predictions.txt')

    def out_full_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension=self.classifier + '.full_predictions.txt')

    def run(self):

        # load vectorized instances
        loader = numpy.load(self.in_test().path)
        vectorized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load classifier
        with open(self.in_model().path, 'rb') as fid:
            model = pickle.load(fid)

        # load labels (for the label encoder)
        with open(self.in_labels().path,'r',encoding='utf-8') as infile:
            labels = infile.read().strip().split('\n')
        if self.ordinal:
            labels = [float(x) for x in labels]

        # inititate classifier
        clf = AbstractSKLearnClassifier()

        # transform labels
        if not self.no_label_encoding:
            clf.set_label_encoder(labels)

        # apply classifier
        predictions, full_predictions = clf.apply_model(model,vectorized_instances,self.no_label_encoding)
        if self.no_label_encoding and self.ordinal:
            predictions = [str(x) for x in predictions]

        # write predictions to file
        with open(self.out_predictions().path,'w',encoding='utf-8') as pr_out:
            pr_out.write('\n'.join(predictions))

        # write full predictions to file
        with open(self.out_full_predictions().path,'w',encoding='utf-8') as fpr_out:
            fpr_out.write('\n'.join(['\t'.join([str(prob) for prob in full_prediction]) for full_prediction in full_predictions]))

@registercomponent
class Train(WorkflowComponent):

    trainvectors = Parameter()
    trainlabels = Parameter()
    classifier_args = Parameter()

    classifier = Parameter()
    ordinal = BoolParameter()
    no_label_encoding = BoolParameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.vectorized.labels', inputparameter='trainlabels'), InputFormat(self, format_id='classifier_args', extension='.txt', inputparameter='classifier_args') ) ]

    def setup(self, workflow, input_feeds):

        trainer = workflow.new_task('train_classifier', TrainClassifier, autopass=True, classifier=self.classifier, ordinal=self.ordinal, label_encoding=self.label_encoding)
        trainer.in_train = input_feeds['trainvectors']
        trainer.in_trainlabels = input_feeds['trainlabels']
        trainer.in_classifier_args = input_feeds['classifier_args']

        return trainer

@registercomponent
class TrainApply(WorkflowComponent):

    trainvectors = Parameter()
    trainlabels = Parameter()
    vocab = Parameter()
    classifier_args = Parameter()
    testvectors = Parameter()

    classifier = Parameter()
    ordinal = BoolParameter()
    no_label_encoding = BoolParameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.vectorized.labels', inputparameter='trainlabels'), InputFormat(self, format_id='vocab', extension='.topfeatures.txt', inputparameter='vocab'), InputFormat(self, format_id='classifier_args', extension='.txt', inputparameter='classifier_args'), InputFormat(self, format_id='testvectors', extension='.vectors.npz',inputparameter='testvectors') ) ]

    def setup(self, workflow, input_feeds):

        trainer = workflow.new_task('train_classifier', TrainClassifier, autopass=True, classifier=self.classifier, ordinal=self.ordinal, no_label_encoding=self.no_label_encoding)
        trainer.in_train = input_feeds['trainvectors']
        trainer.in_trainlabels = input_feeds['trainlabels']
        trainer.in_vocab = input_feeds['vocab']
        trainer.in_classifier_args = input_feeds['classifier_args']

        predictor = workflow.new_task('apply_classifier', ApplyClassifier, autopass=True, ordinal=self.ordinal, no_label_encoding=self.no_label_encoding)
        predictor.in_test = input_feeds['testvectors']
        predictor.in_labels = input_feeds['trainlabels']
        predictor.in_model = trainer.out_model

        return predictor

@registercomponent
class Apply(WorkflowComponent):

    testvectors = Parameter()
    model = Parameter()
    trainlabels = Parameter()

    ordinal = BoolParameter()
    no_label_encoding = BoolParameter()

    def accepts(self):
        return [ ( InputFormat(self, format_id='labels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='testvectors', extension='.vectors.npz',inputparameter='testvectors'), InputFormat(self, format_id='model', extension='.model.pkl',inputparameter='model') ) ]

    def setup(self, workflow, input_feeds):

        predictor = workflow.new_task('apply_classifier', ApplyClassifier, autopass=True, ordinal=self.ordinal, no_label_encoding=self.no_label_encoding)
        predictor.in_test = input_feeds['testvectors']
        predictor.in_labels = input_feeds['labels']
        predictor.in_model = input_feeds['model']

        return predictor


#################################################
######## Raw label transformation
#################################################

class Transform_labels_pre_ml(Task):

    in_labels = InputSlot()
    
    raw_labels = Parameter()

    def out_raw(self):
        return self.outputfrominput(inputformat='labels', stripextension='.labels', addextension='.raw.labels')

    def out_translator(self):
        return self.outputfrominput(inputformat='labels', stripextension='.labels', addextension='.labeltranslator.txt')

    def run(self):
        
        # open labels
        with open(self.in_labels().path,'r',encoding='utf-8') as labels_in:
            labels = labels_in.read().strip().split('\n')
            
        # open raw labels
        with open(self.raw_labels,'r',encoding='utf-8') as raw_labels_in:
            raw_labels = raw_labels_in.read().strip().split('\n')

        # check if both lists have the same length
        if len(labels) != len(raw_labels):
            print('Labels (',len(labels),') and raw labels (',len(raw_labels),') do not have the same length; exiting program...')
            quit()

        # generate dictionary (1 0 14.4\n2 14.4 15.6)
        transformed_labels = []
        for label in sorted(list(set(labels))):
            fitting_raw_labels = [float(x) for i,x in enumerate(raw_labels) if labels[i] == label]
            transformed_labels.append([label,min(fitting_raw_labels),max(fitting_raw_labels)])
        # sort dictionary
        transformed_labels_sorted = sorted(transformed_labels,key = lambda k : k[1])
        transformed_labels_final = []
        for i,tl in enumerate(transformed_labels_sorted):
            if i == 0:
                new_min = transformed_labels_sorted[i][1] - 5000
            else:
                new_min = transformed_labels_sorted[i-1][2]
            if i == len(transformed_labels_sorted)-1:
                new_max = transformed_labels_sorted[i][2] * 50
            else:
                new_max = transformed_labels_sorted[i][2]
            new_tl = [str(transformed_labels_sorted[i][0]),str(new_min),str(new_max)]
            transformed_labels_final.append(new_tl)

        # write output
        with open(self.out_raw().path,'w',encoding='utf-8') as out:
            out.write('\n'.join(raw_labels))

        with open(self.out_translator().path,'w',encoding='utf-8') as out:
            out.write('\n'.join([' '.join(line) for line in transformed_labels_final]))

class Transform_labels_post_ml(Task):

    in_translator = InputSlot()
    in_predictions = InputSlot()

    def out_predictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.translated.predictions.txt')

    def out_full_predictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.translated.full_predictions.txt')

    def run(self):
        
        # open translator
        with open(self.in_translator().path,'r',encoding='utf-8') as translator_in:
            translator = []
            for line in translator_in.read().strip().split('\n'):
                tokens = line.split()
                translator.append([tokens[0],float(tokens[1]),float(tokens[2])])
            
        # open predictions
        with open(self.in_predictions().path,'r',encoding='utf-8') as predictions_in:
            predictions = [float(x) for x in predictions_in.read().strip().split('\n')]

        # translate predictions
        translated_predictions = []
        for prediction in predictions:
            options = []
            for candidate in translator:
                if prediction > candidate[1] and prediction < candidate[2]:
                    options.append(candidate[0])
            if len(options) == 1:
                translated_predictions.append(options[0])
            else:
                print('multiple translations possible for prediction',prediction,':',options,'exiting program...')
                quit()
                
        # write translated predictions to file
        with open(self.out_predictions().path,'w',encoding='utf-8') as out:
            out.write('\n'.join(translated_predictions))

        with open(self.out_full_predictions().path,'w',encoding='utf-8') as out:
            classes = sorted([x[0] for x in translator])
            full_predictions = [classes]
            for i,x in enumerate(translated_predictions):
                full_predictions.append(['-'] * len(classes))
            out.write('\n'.join(['\t'.join([prob for prob in full_prediction]) for full_prediction in full_predictions]))

#################################################
######## Svorim classification (command line tool)
#################################################

@registercomponent
class TrainApplySvorim(WorkflowComponent):

    trainvectors = Parameter()
    trainlabels = Parameter()
    testvectors = Parameter()
    classifier_args = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='testvectors', extension='.vectors.npz',inputparameter='testvectors'), InputFormat(self, format_id='classifier_args', extension='.txt',inputparameter='classifier_args') ) ]

    def setup(self, workflow, input_feeds):

        classifier = workflow.new_task('svorim_classifier', SvorimClassifier, autopass=True, ga_catch=self.ga_catch)
        classifier.in_train = input_feeds['trainvectors']
        classifier.in_labels = input_feeds['trainlabels']
        classifier.in_test = input_feeds['testvectors']
        classifier.in_classifier_args = input_feeds['classifier_args']

        return classifier

class SvorimClassifier(Task):

    in_train = InputSlot()
    in_labels = InputSlot()
    in_test = InputSlot()
    in_classifier_args = InputSlot()

    ga_catch = BoolParameter()

    def out_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='svorim.predictions.txt')

    def out_full_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='svorim.full_predictions.txt')

    def run(self):

        # open train
        loader = numpy.load(self.in_train().path)
        sparse_train_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        array_train_instances = sparse_train_instances.toarray()
        list_train_instances = array_train_instances.tolist()

        # open labels
        with open(self.in_labels().path) as infile:
            labels = infile.read().strip().split('\n')

        # open test
        loader = numpy.load(self.in_test().path)
        sparse_test_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        array_test_instances = sparse_test_instances.toarray()
        list_test_instances = array_test_instances.tolist()

        # open svorim path
        with open(self.in_classifier_args().path) as infile:
            svorim_path = infile.read().strip()

        # set files
        train_instances_labels = []
        for i,instance in enumerate(list_train_instances):
            train_instances_labels.append(instance + [labels[i]])
        expdir = '/'.join(self.in_train().path.split('/')[:-1]) + '/'
        with open(expdir+'svorim_train.0','w',encoding='utf-8') as out:
            out.write('\n'.join([' '.join([str(x) for x in instance]) for instance in train_instances_labels]))
        with open(expdir+'svorim_test.0','w',encoding='utf-8') as out:
            out.write('\n'.join([' '.join([str(x) for x in instance]) + ' ' for instance in list_test_instances]))

        # perform classification
        os.system(svorim_path + ' -i ' + expdir+'svorim_train.0')

        # read in predictions and probabilities
        try:
            predictionfile = expdir + 'svorim_cguess.0'
            probfile = expdir + 'svorim_cguess.0.svm.conf'
            with open(predictionfile) as infile:
                predictions = infile.read().strip().split('\n')
            with open(probfile) as infile:
                probs = infile.read().strip().split('\n')
                full_predictions = [sorted(list(set(labels)))]
                for i,prob in enumerate(probs):
                    template = ['0'] * len(full_predictions[0])
                    index = full_predictions[0].index(predictions[i])
                    template[index] = str(prob)
                    full_predictions.append(template)
        except:
            predictions = ['0'] * len(list_test_instances)
            probs = ['0'] * len(list_test_instances)

        # write predictions to file
        with open(self.out_predictions().path,'w',encoding='utf-8') as pr_out:
            pr_out.write('\n'.join(predictions))

        # write full predictions to file
        with open(self.out_full_predictions().path,'w',encoding='utf-8') as fpr_out:
            fpr_out.write('\n'.join(['\t'.join(full_prediction) for full_prediction in full_predictions]))


#####################################################################
######## Linear regression (continuous labels)
#####################################################################

@registercomponent
class TrainApplyLinearRegression(WorkflowComponent):

    trainvectors = Parameter()
    trainlabels_raw = Parameter()
    trainlabels = Parameter()
    testvectors = Parameter()
    testlabels_raw = Parameter()
    testlabels = Parameter()
    classifier_args = Parameter()
    featurenames = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels_raw', extension='.txt', inputparameter='trainlabels_raw'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='testvectors', extension='.vectors.npz',inputparameter='testvectors'), InputFormat(self, format_id='testlabels_raw', extension='.txt', inputparameter='testlabels_raw'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'),InputFormat(self, format_id='featurenames',extension='.txt',inputparameter='featurenames'), InputFormat(self, format_id='classifier_args', extension='.txt', inputparameter='classifier_args') ) ]

    def setup(self, workflow, input_feeds):

        classifier = workflow.new_task('linreg_classifier', LinearRegression, autopass=True)
        classifier.in_train = input_feeds['trainvectors']
        classifier.in_trainlabels_raw = input_feeds['trainlabels_raw']
        classifier.in_trainlabels = input_feeds['trainlabels']
        classifier.in_test = input_feeds['testvectors']
        classifier.in_testlabels_raw = input_feeds['trainlabels_raw']
        classifier.in_testlabels = input_feeds['trainlabels']
        classifier.in_featurenames = input_feeds['featurenames']
        classifier.in_classifier_args = input_feeds['classier_args']

        return classifier

class LinearRegression(Task):

    in_train = InputSlot()
    in_trainlabels_raw = InputSlot()
    in_trainlabels = InputSlot()
    in_test = InputSlot()
    in_testlabels_raw = InputSlot()
    in_testlabels = InputSlot()
    in_featurenames = InputSlot()
    in_classifier_args = InputSlot()

    def out_raw_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.lreg.raw_predictions.txt')       

    def out_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.lreg.predictions.txt')

    def out_full_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.lreg.full_predictions.txt')

    def out_model(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.lreg.model.pkl')

    def out_model_insights(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.lreg.model_insights')

    def out_label_encoding(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.lreg.le')

    def run(self):

        # initiate directory with model insights
        self.setup_output_dir(self.out_model_insights().path)

        # load vectorized train instances
        loader = numpy.load(self.in_train().path)
        vectorized_traininstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load vectorized test instances
        loader = numpy.load(self.in_test().path)
        vectorized_testinstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load raw trainlabels
        with open(self.in_trainlabels_raw().path,'r',encoding='utf-8') as infile:
            trainlabels_raw = [float(x) for x in infile.read().strip().split('\n')]

        # load raw testlabels
        with open(self.in_testlabels_raw().path,'r',encoding='utf-8') as infile:
            testlabels_raw = [float(x) for x in infile.read().strip().split('\n')]

        # load trainlabels
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')

        # load testlabels
        with open(self.in_testlabels().path,'r',encoding='utf-8') as infile:
            testlabels = infile.read().strip().split('\n')

        # concatenate labelfiles
        raw_labels = trainlabels_raw + testlabels_raw
        nominal_labels = trainlabels + testlabels

        # load vocabulary
        with open(self.in_featurenames().path,'r',encoding='utf-8') as infile:
            vocab = infile.read().strip().split('\n')

        # load classifier arguments        
        with open(self.in_classifier_args().path,'r',encoding='utf-8') as infile:
            classifier_args_str = infile.read().rstrip()
            classifier_args = classifier_args_str.split('\n')

        # train classifier
        lreg = LinearRegressionClassifier(raw_labels,nominal_labels)
        lreg.train_classifier(vectorized_traininstances,trainlabels_raw,classifier_args)

        # apply classifier
        predictions_raw, predictions, full_predictions = lreg.apply_classifier(vectorized_testinstances)

        # save classifier and insights
        model = clf.return_classifier()
        with open(self.out_model().path, 'wb') as fid:
            pickle.dump(model, fid)
        model_insights = clf.return_model_insights(vocab)
        for mi in model_insights:
            with open(self.out_model_insights().path + '/' + mi[0],'w',encoding='utf-8') as outfile:
                outfile.write(mi[1])

        # save label encoding
        label_encoding = clf.return_label_encoding(trainlabels)
        with open(self.out_label_encoding().path,'w',encoding='utf-8') as le_out:
            le_out.write('\n'.join([' '.join(le) for le in label_encoding.items()]))

        # write predictions to file
        with open(self.out_predictions().path,'w',encoding='utf-8') as rpr_out:
            rpr_out.write('\n'.join(predictions_raw))

        # write predictions to file
        with open(self.out_raw_predictions().path,'w',encoding='utf-8') as pr_out:
            pr_out.write('\n'.join(predictions))

        # write full predictions to file
        with open(self.out_full_predictions().path,'w',encoding='utf-8') as fpr_out:
            fpr_out.write('\n'.join(['\t'.join([str(prob) for prob in full_prediction]) for full_prediction in full_predictions]))


#####################################################################
######## Decision tree classification (continuous segmentation)
#####################################################################

@registercomponent
class TrainApplyDTC(WorkflowComponent):

    trainvectors = Parameter()
    trainlabels = Parameter()
    testvectors = Parameter()
    featurenames = Parameter()

    minimum_per_class = IntParameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='testvectors', extension='.vectors.npz',inputparameter='testvectors'), InputFormat(self, format_id='featurenames',extension='.txt',inputparameter='featurenames') ) ]

    def setup(self, workflow, input_feeds):

        classifier = workflow.new_task('dtc_classifier', DTCClassifier, autopass=True, minimum_per_class=self.minimum_per_class)
        classifier.in_train = input_feeds['trainvectors']
        classifier.in_labels = input_feeds['trainlabels']
        classifier.in_test = input_feeds['testvectors']
        classifier.in_featurenames = input_feeds['featurenames']

        return classifier

class DTCClassifier(Task):

    in_train = InputSlot()
    in_labels = InputSlot()
    in_test = InputSlot()
    in_featurenames = InputSlot()

    minimum_per_class = IntParameter()
    minimum_IG = Parameter()

    def out_classifications(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.dtc.classifications.txt')

    def out_tree(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.dtc.tree.txt')

    def out_best_features(self):
        return self.outputfrominput(inputformat='train',stripextension='.vectors.npz', addextension='.best_features.txt')

    def run(self):

        # open train
        loader = numpy.load(self.in_train().path)
        sparse_train_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        array_train_instances = sparse_train_instances.toarray()
        # list_train_instances = array_train_instances.tolist()

        # open labels
        with open(self.in_labels().path) as infile:
            labels = infile.read().strip().split('\n')

        # open test
        loader = numpy.load(self.in_test().path)
        sparse_test_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        array_test_instances = sparse_test_instances.toarray()
        # list_test_instances = array_test_instances.tolist()

        # open features
        with open(self.in_featurenames().path) as infile:
            featurenames = infile.read().strip().split('\n')

        # train classifier
        dtc = DecisionTreeContinuous()
        dtc.fit(array_train_instances,labels,self.minimum_per_class,float(self.minimum_IG))

        # apply classifier
        predictions = dtc.transform(array_test_instances)
        # print('PREDICTIONS',predictions)
        predictions = [x[1] for x in sorted(predictions.items(),key = lambda k : k[0])]

        # write tree to file
        best_features = []
        with open(self.out_tree().path,'w',encoding='utf-8') as tree_out:
            tree = dtc.Tree
            for item in sorted(tree.items(),key=lambda k : k[0]):
                node = str(item[0])
                feature_index = item[1][0]
                fi = str(feature_index) if item[0] else 'LAST'
                if feature_index:
                    best_features.append(feature_index)
                featurename = featurenames[feature_index] if item[0] else 'LAST'
                segmentation = item[1][1]
                # st = '\t'.join([' - '.join([str(round(x[0],10)),str(round(x[1],10))]) for x in segmentation]) if segmentation else 'LAST'
                st = str(segmentation) if segmentation else 'LAST'
                label = item[1][2]
                tree_out.write('\n'.join([node,featurename,st,label]) + '\n\n')

        # write classifications to file
        with open(self.out_classifications().path,'w',encoding='utf-8') as cl_out:
            cl_out.write('\n'.join(['\t'.join([str(prediction),'-']) for prediction in predictions])) 

        # write best features to file
        best_featurenames = list(numpy.array(featurenames)[best_features])
        with open(self.out_best_features().path,'w') as outfile:
            outfile.write('\n'.join(best_featurenames))

#####################################################################
######## Balanced Winnow classification (using LCS)
#####################################################################

@registercomponent
class TrainApplyBalancedWinnow(WorkflowComponent):

    trainvectors = Parameter()
    trainlabels = Parameter()
    testvectors = Parameter()
    testlabels = Parameter()
    vocabulary = Parameter()

    lcs_path = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='testvectors', extension='.vectors.npz',inputparameter='testvectors'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self, format_id='vocabulary',extension='.txt',inputparameter='vocabulary') ) ]

    def setup(self, workflow, input_feeds):

        bw_classifier = workflow.new_task('bw_classifier', BalancedWinnowClassifier, lcs_path=self.lcs_path, autopass=True)
        bw_classifier.in_train = input_feeds['trainvectors']
        bw_classifier.in_trainlabels = input_feeds['trainlabels']
        bw_classifier.in_test = input_feeds['testvectors']
        bw_classifier.in_testlabels = input_feeds['testlabels']
        bw_classifier.in_vocabulary = input_feeds['vocabulary']

        return bw_classifier

class BalancedWinnowClassifier(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_test = InputSlot()
    in_testlabels = InputSlot()
    in_testdocs = InputSlot()
    in_vocabulary = InputSlot()

    lcs_path = Parameter()

    def out_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.lcs.predictions.txt')

    def out_full_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.lcs.full_predictions.txt')
    
    def out_test_indices(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.lcs.doc_indices.txt')
    
    def out_docs(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.lcs.docs.txt')

    def out_labels(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.lcs.labels')

    def out_model_insights(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='lcs.model_insights')

    def out_lcsdir(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.lcsdir')

    def run(self):

        #Set up the output directories, will create it and tear it down on failure automatically
        self.setup_output_dir(self.out_lcsdir().path)
        self.setup_output_dir(self.out_model_insights().path)

        # open train
        loader = numpy.load(self.in_train().path)
        sparse_train_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # open trainlabels
        with open(self.in_trainlabels().path) as infile:
            trainlabels = infile.read().strip().split('\n')

        # open test
        loader = numpy.load(self.in_test().path)
        sparse_test_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # open testlabels
        with open(self.in_testlabels().path) as infile:
            testlabels = infile.read().strip().split('\n')

        # open testdocs
        with open(self.in_testdocs().path,'r',encoding='utf-8') as infile:
            testdocs = infile.read().strip().split('\n')

        # open vocabulary
        with open(self.in_vocabulary().path,'r',encoding='utf-8') as infile:
            vocabulary = [feature.strip().split('\t')[0] for feature in infile.read().strip().split('\n')]

        # train classifier
        lcsc = LCS_classifier(self.out_lcsdir().path)
        lcsc.experiment(sparse_train_instances,trainlabels,vocabulary,sparse_test_instances,testlabels)
        predictions = lcsc.predictions
        full_predictions = lcsc.full_predictions
        docs = lcsc.docs
        model_insights = lcsc.model
        with open(self.out_lcsdir().path + '/testfile_index.txt','r',encoding='utf-8') as tfi_in:
            testfile_index = {}
            for line in tfi_in.read().strip().split('\n'):
                tf_i = line.strip().split()
                testfile_index[tf_i[0]] = tf_i[1]
        testindices = [testfile_index[d] for d in docs]

        testdocs_lcs = [testdocs[int(ti)] for ti in testindices]
        testlabels_lcs = [testlabels[int(ti)] for ti in testindices]

        # write predictions to file
        with open(self.out_predictions().path,'w',encoding='utf-8') as pr_out:
            pr_out.write('\n'.join(predictions))
    
        # write full predictions to file
        with open(self.out_full_predictions().path,'w',encoding='utf-8') as fpr_out:
            fpr_out.write('\n'.join(['\t'.join(full_prediction) for full_prediction in full_predictions]))

        # write model insights to files
        for category in model_insights.keys():
            outfile = self.out_model_insights().path + '/' + category + '_model.txt'
            with open(outfile,'w',encoding='utf-8') as out:
                mi = model_insights[category]
                out.write('\n'.join(['\t'.join([str(x) for x in line]) for line in mi]))

        # write testindices to file
        with open(self.out_test_indices().path,'w',encoding='utf-8') as ti_out:
            ti_out.write('\n'.join(testindices))

        # write succesfully classified test documents to file
        with open(self.out_docs().path,'w',encoding='utf-8') as td_out:
            td_out.write('\n'.join(testdocs_lcs))

        # write succesfully classified test labels to file
        with open(self.out_labels().path,'w',encoding='utf-8') as tl_out:
            tl_out.write('\n'.join(testlabels_lcs))

