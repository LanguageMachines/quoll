
import numpy
from scipy import sparse
import pickle

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.modules import vectorize_instances, vectorize_sparse_instances, classify_instances, report_performance, run_second_layer_classification

@registercomponent
class HierarchicalExperimentComponent(WorkflowComponent):

    trainfeatures = Parameter()
    testfeatures = Parameter()
    trainvocabulary = Parameter()
    testvocabulary = Parameter()
    documents = Parameter()
    trainlabels_layer1 = Parameter()
    trainlabels_layer2 = Parameter()
    testlabels_layer1 = Parameter()
    testlabels_layer2 = Parameter()
    classifier_args = Parameter()

    weight = Parameter(default='frequency')
    prune = IntParameter(default=5000)
    balance = BoolParameter(default=False)
    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter(default=False)
    no_label_encoding = BoolParameter(default=False)
    pca = BoolParameter(default=False)

    def accepts(self):
        return [ ( InputFormat(self,format_id='train',extension='.features.npz',inputparameter='trainfeatures'), InputFormat(self, format_id='trainlabels_layer1', extension='.labels', inputparameter='trainlabels_layer1'), InputFormat(self, format_id='trainlabels_layer2', extension='.labels', inputparameter='trainlabels_layer2'), InputFormat(self, format_id='test', extension='.features.npz',inputparameter='testfeatures'), InputFormat(self, format_id='testlabels_layer1', extension='.labels', inputparameter='testlabels_layer1'), InputFormat(self, format_id='testlabels_layer2', extension='.labels', inputparameter='testlabels_layer2'), InputFormat(self, format_id='trainvocabulary', extension='.vocabulary.txt', inputparameter='trainvocabulary'), InputFormat(self, format_id='testvocabulary', extension='.vocabulary.txt', inputparameter='testvocabulary'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self,format_id='classifier_args',extension='.txt',inputparameter='classifier_args') ) ]

    def setup(self, workflow, input_feeds):

        # layer 1
        if self.balance:
            balancer = workflow.new_task('balance instances', vectorize_sparse_instances.Balance_instances, autopass=True)
            balancer.in_train = input_feeds['train']
            balancer.in_trainlabels = input_feeds['trainlabels_layer1']

            train_vectors = workflow.new_task('vectorize_traininstances_balanced', vectorize_sparse_instances.Vectorize_traininstances, autopass=True, weight=self.weight, prune=self.prune)
            train_vectors.in_train = balancer.out_train
            train_vectors.in_trainlabels = balancer.out_labels
            train_vectors.in_vocabulary = input_feeds['trainvocabulary']
            
        else:
            train_vectors = workflow.new_task('vectorize_traininstances', vectorize_sparse_instances.Vectorize_traininstances, autopass=True, weight=self.weight, prune=self.prune)
            train_vectors.in_train = input_feeds['train']
            train_vectors.in_trainlabels = input_feeds['trainlabels_layer1']
            train_vectors.in_vocabulary = input_feeds['trainvocabulary']

        test_vectors = workflow.new_task('vectorize_testinstances', vectorize_sparse_instances.Vectorize_testinstances, autopass=True, weight=self.weight)
        test_vectors.in_test = input_feeds['test']
        test_vectors.in_sourcevocabulary = input_feeds['testvocabulary']
        test_vectors.in_topfeatures = train_vectors.out_topfeatures
       
        if self.classifier == 'balanced_winnow':
            predictor = workflow.new_task('classify_lcs',  classify_instances.BalancedWinnowClassifier, autopass=True)
            predictor.in_train = train_vectors.out_train
            predictor.in_trainlabels = train_vectors.out_labels
            predictor.in_test = test_vectors.out_test
            predictor.in_testlabels = input_feeds['testlabels_layer1']
            predictor.in_vocabulary = train_vectors.out_topfeatures
            predictor.in_classifier_args = input_feeds['classifier_args']

        else:
            if self.pca:
                dimensionality_reducer = workflow.new_task('train_apply_pca', vectorize_sparse_instances.TrainApply_PCA, autopass=True)
                dimensionality_reducer.in_train = train_vectors.out_train
                dimensionality_reducer.in_test = test_vectors.out_test

                trainer = workflow.new_task('train_classifier_pca', classify_instances.TrainClassifier, autopass=True, classifier=self.classifier, ordinal=self.ordinal, no_label_encoding=self.no_label_encoding)
                trainer.in_train = dimensionality_reducer.out_train
                trainer.in_trainlabels = train_vectors.out_labels
                trainer.in_vocab = dimensionality_reducer.out_vocab
                trainer.in_classifier_args = input_feeds['classifier_args']

                predictor = workflow.new_task('apply_classifier_pca', classify_instances.ApplyClassifier, autopass=True, ordinal=self.ordinal, no_label_encoding=self.no_label_encoding)
                predictor.in_test = dimensionality_reducer.out_test
                predictor.in_labels = train_vectors.out_labels
                predictor.in_model = trainer.out_model
                
            else:
                trainer = workflow.new_task('train_classifier', classify_instances.TrainClassifier, autopass=True, classifier=self.classifier, ordinal=self.ordinal, no_label_encoding=self.no_label_encoding)
                trainer.in_train = train_vectors.out_train
                trainer.in_trainlabels = train_vectors.out_labels
                trainer.in_vocab = train_vectors.out_topfeatures
                trainer.in_classifier_args = input_feeds['classifier_args']

                predictor = workflow.new_task('apply_classifier', classify_instances.ApplyClassifier, autopass=True, ordinal=self.ordinal, no_label_encoding=self.no_label_encoding)
                predictor.in_test = test_vectors.out_test
                predictor.in_labels = train_vectors.out_labels
                predictor.in_model = trainer.out_model

        reporter1 = workflow.new_task('report_performance1', report_performance.ReportPerformance, autopass=False, ordinal=self.ordinal, no_label_encoding=self.no_label_encoding)
        reporter1.in_predictions = predictor.out_predictions
        reporter1.in_full_predictions = predictor.out_full_predictions
        reporter1.in_labels = input_feeds['testlabels_layer1']
        reporter1.in_trainlabels = input_feeds['trainlabels_layer1']
        reporter1.in_documents = input_feeds['documents']

        # layer 2

        # prepare label-bins for second layer based on predictions after layer one
        layer2_bin_maker = workflow.new_task('setup_second_layer_bins', run_second_layer_classification.SetupSecondLayerBins, autopass=True, weight=self.weight, prune=self.prune, classifier=self.classifier, pca=self.pca, ordinal=self.ordinal, balance=self.balance, no_label_encoing=self.no_label_encoding)
        layer2_bin_maker.in_trainlabels_layer1 = input_feeds['trainlabels_layer1']
        layer2_bin_maker.in_docpredictions_layer1 = reporter1.out_docpredictions
        

        # run second layer
        second_layer = workflow.new_task('run_second_layer', run_second_layer_classification.RunSecondLayer, autopass=True, weight=self.weight, prune=self.prune, classifier=self.classifier, pca=self.pca, ordinal=self.ordinal, balance=self.balance, no_label_encoing=self.no_label_encoding)
        second_layer.in_bins = layer2_bin_maker.out_layer2_bins
        second_layer.in_trainfeatures = input_feeds['train']
        second_layer.in_trainlabels_first_layer = input_feeds['trainlabels_layer1']
        second_layer.in_trainlabels_second_layer = input_feeds['trainlabels_layer2']
        second_layer.in_testfeatures = input_feeds['test']
        second_layer.in_testlabels_second_layer = input_feeds['testlabels_layer2']
        second_layer.in_testdocuments = input_feeds['documents']
        second_layer.in_trainvocabulary = input_feeds['trainvocabulary']
        second_layer.in_testvocabulary = input_feeds['testvocabulary']
        second_layer.in_classifier_args = input_feeds['classifier_args']

        # extract predictions from second layer
        layer2_classifications_2_predictions = workflow.new_task('classifications_2_predictions', run_second_layer_classification.SecondLayerClassifications2Predictions, autopass=True)
        layer2_classifications_2_predictions.in_exp_layer2 = second_layer.out_exp_layer2
        layer2_classifications_2_predictions.in_bins = layer2_bin_maker.out_layer2_bins
        layer2_classifications_2_predictions.in_trainlabels_layer2 = input_feeds['trainlabels_layer2']
        layer2_classifications_2_predictions.in_testlabels_layer2 = input_feeds['testlabels_layer2']

        # report performance
        reporter2 = workflow.new_task('report_performance2', report_performance.ReportPerformance, autopass=False, ordinal=self.ordinal, no_label_encoding=self.no_label_encoding)
        reporter2.in_predictions = layer2_classifications_2_predictions.out_predictions
        reporter2.in_full_predictions = layer2_classifications_2_predictions.out_full_predictions
        reporter2.in_labels = input_feeds['testlabels_layer2']
        reporter2.in_trainlabels = input_feeds['trainlabels_layer2']
        reporter2.in_documents = input_feeds['documents']

        return reporter2
