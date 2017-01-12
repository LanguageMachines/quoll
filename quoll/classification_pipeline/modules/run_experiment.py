
import numpy
from scipy import sparse
import pickle

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.modules import vectorize_instances, vectorize_sparse_instances, classify_instances, report_performance

@registercomponent
class ExperimentComponent(WorkflowComponent):

    trainfeatures = Parameter()
    trainlabels = Parameter()
    testfeatures = Parameter()
    testlabels = Parameter()
    vocabulary = Parameter()
    documents = Parameter()

    weight = Parameter(default='frequency')
    prune = IntParameter(default=5000)
    balance = BoolParameter(default=False)
    classifier = Parameter(default='naive_bayes')
    classifier_args = Parameter(default='')
    ordinal = BoolParameter(default=False)

    def accepts(self):
        return [ ( InputFormat(self,format_id='train',extension='.features.npz',inputparameter='trainfeatures'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='test', extension='.features.npz',inputparameter='testfeatures'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self, format_id='vocabulary', extension='.vocabulary.txt', inputparameter='vocabulary'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents') ) ]

    def setup(self, workflow, input_feeds):

        train_vectors = workflow.new_task('vectorize_traininstances', vectorize_sparse_instances.Vectorize_traininstances, autopass=True, weight=self.weight, prune=self.prune, balance=self.balance)
        train_vectors.in_train = input_feeds['train']
        train_vectors.in_trainlabels = input_feeds['trainlabels']
        train_vectors.in_vocabulary = input_feeds['vocabulary']

        test_vectors = workflow.new_task('vectorize_testinstances', vectorize_sparse_instances.Vectorize_testinstances, autopass=True, weight=self.weight)
        test_vectors.in_test = input_feeds['test']
        test_vectors.in_sourcevocabulary = input_feeds['vocabulary']
        test_vectors.in_topfeatures = train_vectors.out_topfeatures

        trainer = workflow.new_task('train_classifier', classify_instances.TrainClassifier, autopass=True, classifier=self.classifier, classifier_args=self.classifier_args)
        trainer.in_train = train_vectors.out_train
        trainer.in_trainlabels = train_vectors.out_labels

        predictor = workflow.new_task('apply_classifier', classify_instances.ApplyClassifier, autopass=True)
        predictor.in_test = test_vectors.out_test
        predictor.in_labels = train_vectors.out_labels
        predictor.in_model = trainer.out_model

        reporter = workflow.new_task('report_performance', report_performance.ReportPerformance, autopass=True)
        reporter.in_predictions = predictor.out_classifications
        reporter.in_labels = input_feeds['testlabels']
        reporter.in_documents = input_feeds['documents']

        return reporter

@registercomponent
class ExperimentComponentVector(WorkflowComponent):

    train = Parameter()
    trainlabels = Parameter()
    test = Parameter()
    testlabels = Parameter()
    documents = Parameter()
    classifier_args = Parameter()

    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter(default=False)

    def accepts(self):
        return [ ( InputFormat(self,format_id='train',extension='.vectors.npz',inputparameter='train'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='test', extension='.vectors.npz',inputparameter='test'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self,format_id='classifier_args',extension='.txt',inputparameter='classifier_args') ) ]

    def setup(self, workflow, input_feeds):

        trainer = workflow.new_task('train_classifier', classify_instances.TrainClassifier, autopass=True, classifier=self.classifier)
        trainer.in_train = input_feeds['train']
        trainer.in_trainlabels = input_feeds['trainlabels']
        trainer.in_classifier_args = input_feeds['classifier_args']

        predictor = workflow.new_task('apply_classifier', classify_instances.ApplyClassifier, autopass=True)
        predictor.in_test = input_feeds['test']
        predictor.in_labels = input_feeds['trainlabels']
        predictor.in_model = trainer.out_model

        reporter = workflow.new_task('report_performance', report_performance.ReportPerformance, autopass=True, ordinal=self.ordinal)
        reporter.in_predictions = predictor.out_classifications
        reporter.in_labels = input_feeds['testlabels']
        reporter.in_documents = input_feeds['documents']

        return reporter

