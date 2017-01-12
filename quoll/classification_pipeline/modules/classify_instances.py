
import numpy
from scipy import sparse
import pickle

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.functions.classifier import *

class TrainClassifier(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_classifier_args = InputSlot()

    classifier = Parameter()

    def out_model(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.model.pkl')

    def out_label_encoding(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.le')

    def run(self):

        # initiate classifier
        classifierdict = {'naive_bayes':NaiveBayesClassifier(), 'svm':SVMClassifier(), 'ordinal_ridge':OrdinalRidge(), 'ordinal_at':OrdinalLogisticAT(), 'ordinal_se':OrdinalLogisticSE(), 'ordinal_it':OrdinalLogisticIT()}
        clf = classifierdict[self.classifier]

        # load vectorized instances
        loader = numpy.load(self.in_train().path)
        vectorized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load trainlabels
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')

        # transform trainlabels
        clf.set_label_encoder(trainlabels)

        # load classifier arguments        
        with open(self.in_classifier_args().path,'r',encoding='utf-8') as infile:
            classifier_args = infile.read().strip().split('\n')

        # train classifier
        clf.train_classifier(vectorized_instances, trainlabels, *classifier_args)
        model = clf.return_classifier()

        # save classifier
        with open(self.out_model().path, 'wb') as fid:
            pickle.dump(model, fid)

        # save label encoding
        label_encoding = clf.return_label_encoding(trainlabels)
        with open(self.out_label_encoding().path,'w',encoding='utf-8') as le_out:
            le_out.write('\n'.join([' '.join(le) for le in label_encoding]))

class ApplyClassifier(Task):

    in_test = InputSlot()
    in_labels = InputSlot()
    in_model = InputSlot()

    def out_classifications(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.classifications.txt')

    def run(self):

        # load vectorized instances
        loader = numpy.load(self.in_test().path)
        vectorized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load classifier
        with open(self.in_model().path, 'rb') as fid:
            model = pickle.load(fid)

        # load labels (for the label encoder)
        with open(self.in_labels().path,'r',encoding='utf-8') as infile:
            labels = infile.read().split('\n')

        # inititate classifier
        clf = AbstractSKLearnClassifier()

        # transform labels
        clf.set_label_encoder(labels)

        # apply classifier
        classifications = clf.apply_model(model,vectorized_instances)

        # write classifications to file
        with open(self.out_classifications().path,'w',encoding='utf-8') as cl_out:
            cl_out.write('\n'.join(['\t'.join([str(field) for field in classification]) for classification in classifications]))

@registercomponent
class Train(WorkflowComponent):

    trainvectors = Parameter()
    trainlabels = Parameter()
    classifier_args = Parameter()

    classifier = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.vectorlabels', inputparameter='trainlabels'), InputFormat(self, format_id='classifier_args', extension='.txt', inputparameter='classifier_args') ) ]

    def setup(self, workflow, input_feeds):

        trainer = workflow.new_task('train_classifier', TrainClassifier, autopass=True, classifier=self.classifier, classifier_args=self.classifier_args)
        trainer.in_train = input_feeds['trainvectors']
        trainer.in_trainlabels = input_feeds['trainlabels']
        trainer.in_classifier_args = input_feeds['classifier_args']

        return trainer

@registercomponent
class TrainApply(WorkflowComponent):

    trainvectors = Parameter()
    trainlabels = Parameter()
    classifier_args = Parameter()
    testvectors = Parameter()

    classifier = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.vectorlabels', inputparameter='trainlabels'), InputFormat(self, format_id='classifier_args', extension='.txt', inputparameter='classifier_args'), InputFormat(self, format_id='testvectors', extension='.vectors.npz',inputparameter='testvectors') ) ]

    def setup(self, workflow, input_feeds):

        trainer = workflow.new_task('train_classifier', TrainClassifier, autopass=True, classifier=self.classifier)
        trainer.in_train = input_feeds['trainvectors']
        trainer.in_trainlabels = input_feeds['trainlabels']
        trainer.in_classifier_args = input_feeds['classifier_args']

        predictor = workflow.new_task('apply_classifier', ApplyClassifier, autopass=True)
        predictor.in_test = input_feeds['testvectors']
        predictor.in_labels = input_feeds['trainlabels']
        predictor.in_model = trainer.out_model

        return predictor
