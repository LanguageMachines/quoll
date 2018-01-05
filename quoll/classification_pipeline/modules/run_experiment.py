
import numpy
from scipy import sparse
import pickle

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.modules import vectorize_instances, vectorize_sparse_instances, classify_instances, report_performance, filter_features_correlation, rank_features_ordinal_correlation, select_features_fcbf

@registercomponent
class ExperimentComponent(WorkflowComponent):

    trainfeatures = Parameter()
    trainlabels = Parameter()
    testfeatures = Parameter()
    testlabels = Parameter()
    trainvocabulary = Parameter()
    testvocabulary = Parameter()
    documents = Parameter()
    classifier_args = Parameter()

    weight = Parameter(default='frequency')
    prune = IntParameter(default=5000)
    balance = BoolParameter(default=False)
    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter(default=False)
    no_label_encoding = BoolParameter(default=False)

    def accepts(self):
        return [ ( InputFormat(self,format_id='train',extension='.features.npz',inputparameter='trainfeatures'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='test', extension='.features.npz',inputparameter='testfeatures'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self, format_id='trainvocabulary', extension='.vocabulary.txt', inputparameter='trainvocabulary'), InputFormat(self, format_id='testvocabulary', extension='.vocabulary.txt', inputparameter='testvocabulary'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self,format_id='classifier_args',extension='.txt',inputparameter='classifier_args') ) ]

    def setup(self, workflow, input_feeds):

        if self.balance:
            balancer = workflow.new_task('balance instances', vectorize_sparse_instances.Balance_instances, autopass=True)
            balancer.in_train = input_feeds['train']
            balancer.in_trainlabels = input_feeds['trainlabels']
            balancer.in_test = input_feeds['test']

            train_vectors = workflow.new_task('vectorize_traininstances_balanced', vectorize_sparse_instances.Vectorize_traininstances, autopass=True, weight=self.weight, prune=self.prune)
            train_vectors.in_train = balancer.out_train
            train_vectors.in_trainlabels = balancer.out_labels
            train_vectors.in_vocabulary = input_feeds['trainvocabulary']

            test_vectors = workflow.new_task('vectorize_testinstances_balanced', vectorize_sparse_instances.Vectorize_testinstances, autopass=True, weight=self.weight)
            test_vectors.in_test = balancer.out_test
            test_vectors.in_sourcevocabulary = input_feeds['testvocabulary']
            test_vectors.in_topfeatures = train_vectors.out_topfeatures

        else:
            train_vectors = workflow.new_task('vectorize_traininstances', vectorize_sparse_instances.Vectorize_traininstances, autopass=True, weight=self.weight, prune=self.prune)
            train_vectors.in_train = input_feeds['train']
            train_vectors.in_trainlabels = input_feeds['trainlabels']
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
            predictor.in_testlabels = input_feeds['testlabels']
            predictor.in_vocabulary = train_vectors.out_topfeatures
            predictor.in_classifier_args = input_feeds['classifier_args']

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

        reporter = workflow.new_task('report_performance', report_performance.ReportPerformance, autopass=True, ordinal=self.ordinal, no_label_encoding=self.no_label_encoding)
        reporter.in_predictions = predictor.out_predictions
        reporter.in_full_predictions = predictor.out_full_predictions
        reporter.in_labels = input_feeds['testlabels']
        reporter.in_trainlabels = input_feeds['trainlabels']
        reporter.in_documents = input_feeds['documents']

        return reporter

@registercomponent
class ExperimentComponentVector(WorkflowComponent):

    train = Parameter()
    trainlabels = Parameter()
    test = Parameter()
    testlabels = Parameter()
    documents = Parameter()
    featurenames = Parameter()
    classifier_args = Parameter()

    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter(default=False)
    raw_labels = Parameter(default=False)

    def accepts(self):
        return [ ( InputFormat(self,format_id='train',extension='.vectors.npz',inputparameter='train'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='test', extension='.vectors.npz',inputparameter='test'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self,format_id='featurenames',extension='.txt',inputparameter='featurenames'), InputFormat(self,format_id='classifier_args',extension='.txt',inputparameter='classifier_args') ) ]

    def setup(self, workflow, input_feeds):

        if self.classifier == 'svorim':
            predictor = workflow.new_task('classifier', classify_instances.SvorimClassifier, autopass=False)
            predictor.in_train = input_feeds['train']
            predictor.in_labels = input_feeds['trainlabels']
            predictor.in_test = input_feeds['test']
            predictor.in_classifier_args = input_feeds['classifier_args']
        elif self.classifier == 'balanced_winnow':
            predictor = workflow.new_task('classify_lcs', classify_instances.BalancedWinnowClassifier, autopass=True)
            predictor.in_train = input_feeds['train'] 
            predictor.in_trainlabels = input_feeds['trainlabels']
            predictor.in_test = input_feeds['test']
            predictor.in_testlabels = input_feeds['testlabels']
            predictor.in_vocabulary = input_feeds['featurenames']
            predictor.in_classifier_args = input_feeds['classifier_args']
        else:
            if self.raw_labels != 'false':
                pre_ml_label_transformer = workflow.new_task('Tranform labels pre ML', classify_instances.Transform_labels_pre_ml, autopass=True, raw_labels=self.raw_labels)
                pre_ml_label_transformer.in_labels = input_feeds['trainlabels']
                
                trainer = workflow.new_task('train_classifier', classify_instances.TrainClassifier, autopass=True, classifier=self.classifier, ordinal=self.ordinal, no_label_encoding=True)
                trainer.in_train = input_feeds['train']
                trainer.in_trainlabels = pre_ml_label_transformer.out_raw
                trainer.in_vocab = input_feeds['featurenames']
                trainer.in_classifier_args = input_feeds['classifier_args']

                predictor = workflow.new_task('apply_classifier', classify_instances.ApplyClassifier, autopass=True, ordinal=self.ordinal, no_label_encoding=True)
                predictor.in_test = input_feeds['test']
                predictor.in_labels = pre_ml_label_transformer.out_raw
                predictor.in_model = trainer.out_model
            else:
                trainer = workflow.new_task('train_classifier', classify_instances.TrainClassifier, autopass=True, classifier=self.classifier, ordinal=self.ordinal, no_label_encoding=self.no_label_encoding)
                trainer.in_train = input_feeds['train']
                trainer.in_trainlabels = input_feeds['trainlabels']
                trainer.in_vocab = input_feeds['featurenames']
                trainer.in_classifier_args = input_feeds['classifier_args']

                predictor = workflow.new_task('apply_classifier', classify_instances.ApplyClassifier, autopass=True, ordinal=self.ordinal, no_label_encoding=self.no_label_encoding)
                predictor.in_test = input_feeds['test']
                predictor.in_labels = input_feeds['trainlabels']
                predictor.in_model = trainer.out_model

        if self.raw_labels != 'false':
            post_ml_label_transformer = workflow.new_task('transform labels post ML', classify_instances.Transform_labels_post_ml, autopass=True)
            post_ml_label_transformer.in_translator = pre_ml_label_transformer.out_translator
            post_ml_label_transformer.in_predictions = predictor.out_predictions

            reporter = workflow.new_task('report_performance', report_performance.ReportPerformance, autopass=True, ordinal=self.ordinal)
            reporter.in_predictions = post_ml_label_transformer.out_predictions
            reporter.in_full_predictions = post_ml_label_transformer.out_full_predictions
            reporter.in_labels = input_feeds['testlabels']
            reporter.in_trainlabels = input_feeds['trainlabels']
            reporter.in_documents = input_feeds['documents']

        else:
            reporter = workflow.new_task('report_performance', report_performance.ReportPerformance, autopass=True, ordinal=self.ordinal, no_label_encoding=self.no_label_encoding)
            reporter.in_predictions = predictor.out_predictions
            reporter.in_full_predictions = predictor.out_full_predictions
            reporter.in_labels = input_feeds['testlabels']
            reporter.in_trainlabels = input_feeds['trainlabels']
            reporter.in_documents = input_feeds['documents']

        return reporter


# @registercomponent
# class ExperimentComponentLRegVector(WorkflowComponent):

#     train = Parameter()
#     trainlabels = Parameter()
#     trainlabels_raw = Parameter()
#     test = Parameter()
#     testlabels = Parameter()
#     testlabels_raw = Parameter()
#     documents = Parameter()
#     classifier_args = Parameter()
#     featurenames = Parameter()
#     classifier_args = Parameter()

#     def accepts(self):
#         return [ ( InputFormat(self,format_id='train',extension='.vectors.npz',inputparameter='train'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='trainlabels_raw', extension='.txt', inputparameter='trainlabels_raw'), InputFormat(self, format_id='test', extension='.vectors.npz',inputparameter='test'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self, format_id='testlabels_raw', extension='.txt', inputparameter='testlabels_raw'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self,format_id='featurenames',extension='.txt',inputparameter='featurenames'), InputFormat(self,format_id='classifier_args',extension='.txt',inputparameter='classifier_args') ) ]

#     def setup(self, workflow, input_feeds):

#         classifier = workflow.new_task('lreg_classifier', classify_instances.TrainApplyLinearRegression, autopass=False)
#         classifier.in_train = input_feeds['train']
#         classifier.in_trainlabels = input_feeds['trainlabels']
#         classifier.in_trainlabels_raw = input_feeds['trainlabels_raw']
#         classifier.in_test = input_feeds['test']
#         classifier.in_testlabels = input_feeds['testlabels']
#         classifier.in_testlabels_raw = input_feeds['testlabels_raw']

#         reporter = workflow.new_task('report_performance', report_performance.ReportPerformance, autopass=True, ordinal=True)
#         reporter.in_predictions = classifier.out_predictions
#         reporter.in_full_predictions = classifier.out_full_predictions
#         reporter.in_labels = input_feeds['testlabels']
#         reporter.in_trainlabels = input_feeds['trainlabels']
#         reporter.in_documents = input_feeds['documents']

#         return reporter

@registercomponent
class ExperimentComponentDTCVector(WorkflowComponent):

    train = Parameter()
    trainlabels = Parameter()
    test = Parameter()
    testlabels = Parameter()
    documents = Parameter()
    featurenames = Parameter()

    ordinal = BoolParameter()
    minimum_per_class = IntParameter()
    minimum_IG = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='train',extension='.vectors.npz',inputparameter='train'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='test', extension='.vectors.npz',inputparameter='test'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self,format_id='featurenames',extension='.txt',inputparameter='featurenames') ) ]

    def setup(self, workflow, input_feeds):

        classifier = workflow.new_task('dtc_classifier', classify_instances.DTCClassifier, minimum_per_class=self.minimum_per_class, minimum_IG=self.minimum_IG, autopass=False)
        classifier.in_train = input_feeds['train']
        classifier.in_labels = input_feeds['trainlabels']
        classifier.in_test = input_feeds['test']
        classifier.in_featurenames = input_feeds['featurenames']

        reporter = workflow.new_task('report_performance', report_performance.ReportPerformance, autopass=True, ordinal=self.ordinal)
        reporter.in_predictions = classifier.out_classifications
        reporter.in_labels = input_feeds['testlabels']
        reporter.in_trainlabels = input_feeds['trainlabels']
        reporter.in_documents = input_feeds['documents']

        return reporter

@registercomponent
class ExperimentComponentLin(WorkflowComponent):

    train = Parameter()
    trainlabels = Parameter()
    test = Parameter()
    testlabels = Parameter()
    documents = Parameter()
    featurenames = Parameter()
    featurecorrelation = Parameter()

    feature_cutoff = IntParameter()
    svorim_path = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='train',extension='.vectors.npz',inputparameter='train'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='test', extension='.vectors.npz',inputparameter='test'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self, format_id='featurecorrelation', extension='.txt', inputparameter='featurecorrelation'), InputFormat(self, format_id='featurenames', extension='.txt', inputparameter='featurenames') ) ]

    def setup(self, workflow, input_feeds):

        feature_ranker = workflow.new_task('rank_features',rank_features_ordinal_correlation.RankFeaturesOrdinalTask,autopass=True)
        feature_ranker.in_vectors = input_feeds['train']
        feature_ranker.in_labels = input_feeds['trainlabels']
        feature_ranker.in_featurenames = input_feeds['featurenames']

        feature_filter = workflow.new_task('filter_features',filter_features_correlation.FilterFeaturesTask,autopass=True,cutoff=self.feature_cutoff)
        feature_filter.in_featurenames = input_feeds['featurenames']
        feature_filter.in_featrank = feature_ranker.out_ranked_features
        feature_filter.in_featcorr = input_feeds['featurecorrelation']

        train_vector_transformer = workflow.new_task('transform_vectors',vectorize_instances.TransformVectorsTask,autopass=True)
        train_vector_transformer.in_vectors = input_feeds['train']
        train_vector_transformer.in_selection = feature_filter.out_filtered_features_index

        test_vector_transformer = workflow.new_task('transform_test_vectors',vectorize_instances.TransformVectorsTask,autopass=True)
        test_vector_transformer.in_vectors = input_feeds['test']
        test_vector_transformer.in_selection = feature_filter.out_filtered_features_index

        classifier = workflow.new_task('svorim_classifier', classify_instances.SvorimClassifier, autopass=False, svorim_path=self.svorim_path)
        classifier.in_train = train_vector_transformer.out_vectors
        classifier.in_labels = input_feeds['trainlabels']
        classifier.in_test = test_vector_transformer.out_vectors

        reporter = workflow.new_task('report_performance', report_performance.ReportPerformance, autopass=True, ordinal=True)
        reporter.in_predictions = classifier.out_classifications
        reporter.in_labels = input_feeds['testlabels']
        reporter.in_trainlabels = input_feeds['trainlabels']
        reporter.in_documents = input_feeds['documents']

        return reporter

@registercomponent
class ExperimentComponentLin2(WorkflowComponent):

    train = Parameter()
    trainlabels = Parameter()
    test = Parameter()
    testlabels = Parameter()
    documents = Parameter()
    featurenames = Parameter()
    featuregroups = Parameter()

    svorim_path = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='train',extension='.vectors.npz',inputparameter='train'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='test', extension='.vectors.npz',inputparameter='test'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self, format_id='featuregroups', extension='.txt', inputparameter='featuregroups'), InputFormat(self, format_id='featurenames', extension='.txt', inputparameter='featurenames') ) ]

    def setup(self, workflow, input_feeds):

        feature_ranker = workflow.new_task('rank_features',rank_features_ordinal_correlation.RankFeaturesOrdinalTask,autopass=True)
        feature_ranker.in_vectors = input_feeds['train']
        feature_ranker.in_labels = input_feeds['trainlabels']
        feature_ranker.in_featurenames = input_feeds['featurenames']

        feature_filter = workflow.new_task('filter_features',filter_features_correlation.FilterFeaturesGroupsTask,autopass=True)
        feature_filter.in_featurenames = input_feeds['featurenames']
        feature_filter.in_featrank = feature_ranker.out_ranked_features
        feature_filter.in_featgroups = input_feeds['featuregroups']

        train_vector_transformer = workflow.new_task('transform_vectors',vectorize_instances.TransformVectorsTask,autopass=True)
        train_vector_transformer.in_vectors = input_feeds['train']
        train_vector_transformer.in_selection = feature_filter.out_filtered_features_index

        test_vector_transformer = workflow.new_task('transform_test_vectors',vectorize_instances.TransformVectorsTask,autopass=True)
        test_vector_transformer.in_vectors = input_feeds['test']
        test_vector_transformer.in_selection = feature_filter.out_filtered_features_index

        classifier = workflow.new_task('svorim_classifier', classify_instances.SvorimClassifier, autopass=False, svorim_path=self.svorim)
        classifier.in_train = train_vector_transformer.out_vectors
        classifier.in_labels = input_feeds['trainlabels']
        classifier.in_test = test_vector_transformer.out_vectors

        reporter = workflow.new_task('report_performance', report_performance.ReportPerformance, autopass=True, ordinal=True)
        reporter.in_predictions = classifier.out_classifications
        reporter.in_labels = input_feeds['testlabels']
        reporter.in_trainlabels = input_feeds['trainlabels']
        reporter.in_documents = input_feeds['documents']

        return reporter

@registercomponent
class ExperimentComponentFilter(WorkflowComponent):

    train = Parameter()
    trainlabels = Parameter()
    test = Parameter()
    testlabels = Parameter()
    documents = Parameter()
    featurenames = Parameter()
    classifier_args = Parameter()

    threshold_strength = Parameter()
    threshold_correlation = Parameter()
    classifier = Parameter()
    ordinal = BoolParameter()
    raw_labels = Parameter()
    
    def accepts(self):
        return [ ( InputFormat(self,format_id='train',extension='.vectors.npz',inputparameter='train'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='test', extension='.vectors.npz',inputparameter='test'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self, format_id='featurenames', extension='.txt', inputparameter='featurenames'), InputFormat(self,format_id='classifier_args',extension='.txt',inputparameter='classifier_args') ) ]

    def setup(self, workflow, input_feeds):

        if self.raw_labels != 'false':
            pre_ml_label_transformer = workflow.new_task('Tranform labels pre ML', classify_instances.Transform_labels_pre_ml, autopass=True, raw_labels=self.raw_labels)
            pre_ml_label_transformer.in_labels = input_feeds['trainlabels']

            feature_ranker = workflow.new_task('rank_features',rank_features_ordinal_correlation.RankFeaturesOrdinalTask,autopass=True)
            feature_ranker.in_vectors = input_feeds['train']
            feature_ranker.in_labels = pre_ml_label_transformer.out_raw
            feature_ranker.in_featurenames = input_feeds['featurenames']

            feature_correlation = workflow.new_task('feature_correlation',rank_features_ordinal_correlation.CalculateFeatureCorrelationTask,autopass=True)
            feature_correlation.in_vectors = input_feeds['train']
            feature_correlation.in_featurenames = input_feeds['featurenames']

            feature_filter = workflow.new_task('filter_features',filter_features_correlation.FilterFeaturesFTask,autopass=True, threshold_strength=self.threshold_strength,threshold_correlation=self.threshold_correlation)
            feature_filter.in_featurenames = input_feeds['featurenames']
            feature_filter.in_featstrength = feature_ranker.out_ranked_features
            feature_filter.in_featcorr = feature_correlation.out_feature_correlation

            train_vector_transformer = workflow.new_task('transform_vectors',vectorize_instances.TransformVectorsTask,autopass=True)
            train_vector_transformer.in_vectors = input_feeds['train']
            train_vector_transformer.in_selection = feature_filter.out_filtered_features_index

            test_vector_transformer = workflow.new_task('transform_test_vectors',vectorize_instances.TransformVectorsTask,autopass=True)
            test_vector_transformer.in_vectors = input_feeds['test']
            test_vector_transformer.in_selection = feature_filter.out_filtered_features_index
                
            trainer = workflow.new_task('train_classifier', classify_instances.TrainClassifier, autopass=True, classifier=self.classifier, ordinal=self.ordinal, no_label_encoding=True)
            trainer.in_train = train_vector_transformer.out_vectors
            trainer.in_trainlabels = pre_ml_label_transformer.out_raw
            trainer.in_vocab = input_feeds['featurenames']
            trainer.in_classifier_args = input_feeds['classifier_args']

            predictor = workflow.new_task('apply_classifier', classify_instances.ApplyClassifier, autopass=True, ordinal=self.ordinal, no_label_encoding=True)
            predictor.in_test = test_vector_transformer.out_vectors
            predictor.in_labels = pre_ml_label_transformer.out_raw
            predictor.in_model = trainer.out_model

            post_ml_label_transformer = workflow.new_task('transform labels post ML', classify_instances.Transform_labels_post_ml, autopass=True)
            post_ml_label_transformer.in_translator = pre_ml_label_transformer.out_translator
            post_ml_label_transformer.in_predictions = predictor.out_predictions

            reporter = workflow.new_task('report_performance', report_performance.ReportPerformance, autopass=True, ordinal=self.ordinal)
            reporter.in_predictions = post_ml_label_transformer.out_predictions
            reporter.in_full_predictions = post_ml_label_transformer.out_full_predictions
            reporter.in_labels = input_feeds['testlabels']
            reporter.in_trainlabels = input_feeds['trainlabels']
            reporter.in_documents = input_feeds['documents']
        
        else:

            feature_ranker = workflow.new_task('rank_features',rank_features_ordinal_correlation.RankFeaturesOrdinalTask,autopass=True)
            feature_ranker.in_vectors = input_feeds['train']
            feature_ranker.in_labels = input_feeds['trainlabels']
            feature_ranker.in_featurenames = input_feeds['featurenames']

            feature_correlation = workflow.new_task('feature_correlation',rank_features_ordinal_correlation.CalculateFeatureCorrelationTask,autopass=True)
            feature_correlation.in_vectors = input_feeds['train']
            feature_correlation.in_featurenames = input_feeds['featurenames']

            feature_filter = workflow.new_task('filter_features',filter_features_correlation.FilterFeaturesFTask,autopass=True, threshold_strength=self.threshold_strength,threshold_correlation=self.threshold_correlation)
            feature_filter.in_featurenames = input_feeds['featurenames']
            feature_filter.in_featstrength = feature_ranker.out_ranked_features
            feature_filter.in_featcorr = feature_correlation.out_feature_correlation

            train_vector_transformer = workflow.new_task('transform_vectors',vectorize_instances.TransformVectorsTask,autopass=True)
            train_vector_transformer.in_vectors = input_feeds['train']
            train_vector_transformer.in_selection = feature_filter.out_filtered_features_index

            test_vector_transformer = workflow.new_task('transform_test_vectors',vectorize_instances.TransformVectorsTask,autopass=True)
            test_vector_transformer.in_vectors = input_feeds['test']
            test_vector_transformer.in_selection = feature_filter.out_filtered_features_index

            if self.classifier == 'svorim':
                predictor = workflow.new_task('classifier', classify_instances.SvorimClassifier, autopass=False)
                predictor.in_train = train_vector_transformer.out_vectors
                predictor.in_labels = input_feeds['trainlabels']
                predictor.in_test = test_vector_transformer.out_vectors
                predictor.in_classifier_args = input_feeds['classifier_args']
            elif self.classifier == 'balanced_winnow':
                predictor = workflow.new_task('classify_lcs',  classify_instances.BalancedWinnowClassifier, autopass=True)
                predictor.in_train = train_vector_transformer.out_vectors
                predictor.in_trainlabels = input_feeds['trainlabels']
                predictor.in_test = test_vector_transformer.out_vectors
                predictor.in_testlabels = input_feeds['testlabels']
                predictor.in_vocabulary = input_feeds['featurenames']
                predictor.in_classifier_args = input_feeds['classifier_args']
            else:
                trainer = workflow.new_task('train_classifier', classify_instances.TrainClassifier, autopass=True, classifier=self.classifier)
                trainer.in_train = train_vector_transformer.out_vectors
                trainer.in_trainlabels = input_feeds['trainlabels']
                trainer.in_vocab = input_feeds['featurenames']
                trainer.in_classifier_args = input_feeds['classifier_args']

                predictor = workflow.new_task('apply_classifier', classify_instances.ApplyClassifier, autopass=True)
                predictor.in_test = test_vector_transformer.out_vectors
                predictor.in_labels = input_feeds['trainlabels']
                predictor.in_model = trainer.out_model

            reporter = workflow.new_task('report_performance', report_performance.ReportPerformance, autopass=True, ordinal=True)
            reporter.in_predictions = predictor.out_predictions
            reporter.in_full_predictions = predictor.out_full_predictions
            reporter.in_labels = input_feeds['testlabels']
            reporter.in_trainlabels = input_feeds['trainlabels']
            reporter.in_documents = input_feeds['documents']

        return reporter


@registercomponent
class ExperimentComponentFCBF(WorkflowComponent):

    train = Parameter()
    trainlabels = Parameter()
    test = Parameter()
    testlabels = Parameter()
    documents = Parameter()
    featurenames = Parameter()

    fcbf_threshold = Parameter()
    svorim_path = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='train',extension='.vectors.npz',inputparameter='train'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='test', extension='.vectors.npz',inputparameter='test'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self, format_id='featurenames', extension='.txt', inputparameter='featurenames') ) ]

    def setup(self, workflow, input_feeds):

        file_formatter = workflow.new_task('trainvectors_2_fcbf_input', select_features_fcbf.Trainvectors2FCBFInput, autopass=False)
        file_formatter.in_vectors = input_feeds['train']
        file_formatter.in_labels = input_feeds['trainlabels']

        feature_selector = workflow.new_task('select_features_fcbf', select_features_fcbf.FCBFTask, autopass=False, threshold=self.fcbf_threshold)
        feature_selector.in_instances = file_formatter.out_instances
        feature_selector.in_featurenames = input_feeds['featurenames']

        train_vector_transformer = workflow.new_task('transform_vectors',vectorize_instances.TransformVectorsTask,autopass=True)
        train_vector_transformer.in_vectors = input_feeds['train']
        train_vector_transformer.in_selection = feature_selector.out_feature_indices

        test_vector_transformer = workflow.new_task('transform_test_vectors',vectorize_instances.TransformVectorsTask,autopass=True)
        test_vector_transformer.in_vectors = input_feeds['test']
        test_vector_transformer.in_selection = feature_selector.out_feature_indices

        classifier = workflow.new_task('svorim_classifier', classify_instances.SvorimClassifier, autopass=False, svorim_path=self.svorim_path)
        classifier.in_train = train_vector_transformer.out_vectors
        classifier.in_labels = input_feeds['trainlabels']
        classifier.in_test = test_vector_transformer.out_vectors

        reporter = workflow.new_task('report_performance', report_performance.ReportPerformance, autopass=True, ordinal=True)
        reporter.in_predictions = classifier.out_classifications
        reporter.in_labels = input_feeds['testlabels']
        reporter.in_trainlabels = input_feeds['trainlabels']
        reporter.in_documents = input_feeds['documents']

        return reporter
