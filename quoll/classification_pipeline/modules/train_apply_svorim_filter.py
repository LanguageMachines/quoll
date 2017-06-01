
from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.modules import vectorize_instances, classify_instances, report_performance, filter_features_correlation, rank_features_ordinal_correlation

@registercomponent
class TrainApplySvorimFilter(WorkflowComponent):

    train = Parameter()
    trainlabels = Parameter()
    test = Parameter()
    featurenames_train = Parameter()
    featurenames_test = Parameter()t
    documents_test = Parameter()

    threshold_strength = Parameter()
    threshold_correlation = Parameter()
    svorim_path = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='train',extension='.vectors.npz',inputparameter='train'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='test', extension='.vectors.npz',inputparameter='test'), InputFormat(self,format_id='documents_test',extension='.txt',inputparameter='documents_test'), InputFormat(self, format_id='featurenames_train', extension='.txt', inputparameter='featurenames_train'), InputFormat(self, format_id='featurenames_test', extension='.txt', inputparameter='featurenames_test') ) ]

    def setup(self, workflow, input_feeds):

        feature_ranker = workflow.new_task('rank_features',rank_features_ordinal_correlation.RankFeaturesOrdinalTask,autopass=True)
        feature_ranker.in_vectors = input_feeds['train']
        feature_ranker.in_labels = input_feeds['trainlabels']
        feature_ranker.in_featurenames = input_feeds['featurenames_train']

        feature_correlation = workflow.new_task('feature_correlation',rank_features_ordinal_correlation.CalculateFeatureCorrelationTask,autopass=True)
        feature_correlation.in_vectors = input_feeds['train']
        feature_correlation.in_featurenames = input_feeds['featurenames_train']

        feature_filter = workflow.new_task('filter_features',filter_features_correlation.FilterFeaturesFTask,autopass=True, threshold_strength=self.threshold_strength,threshold_correlation=self.threshold_correlation)
        feature_filter.in_featurenames = input_feeds['featurenames_train']
        feature_filter.in_featstrength = feature_ranker.out_ranked_features
        feature_filter.in_featcorr = feature_correlation.out_feature_correlation

        train_vector_transformer = workflow.new_task('transform_vectors',vectorize_instances.TransformVectorsTask,autopass=True)
        train_vector_transformer.in_vectors = input_feeds['train']
        train_vector_transformer.in_selection = feature_filter.out_filtered_features_index

        index_generator = workflow.new_task('generate_feature_indices',GenerateFeatureIndices,autopass=True)
        index_generator.in_all_featurenames = input_feeds['featurenames_test']
        index_generator.in_selected_featurenames = feature_filter.out_filtered_features

        test_vector_transformer = workflow.new_task('transform_test_vectors',vectorize_instances.TransformVectorsTask,autopass=True)
        test_vector_transformer.in_vectors = input_feeds['test']
        test_vector_transformer.in_selection = index_generator.out_features_index

        classifier = workflow.new_task('svorim_classifier', classify_instances.SvorimClassifier, autopass=False, svorim_path=self.svorim_path)
        classifier.in_train = train_vector_transformer.out_vectors
        classifier.in_labels = input_feeds['trainlabels']
        classifier.in_test = test_vector_transformer.out_vectors

        reporter = workflow.new_task('report_performance', report_performance.ReportDocpredictions, autopass=True)
        reporter.in_predictions = classifier.out_classifications
        reporter.in_documents = input_feeds['documents_test']

        return reporter

class GenerateFeatureIndices(Task):

    in_all_featurenames = InputSlot()
    in_selected_featurenames = InputSlot()

    def out_features_index(self):
        return self.outputfrominput(inputformat='selected_featurenames', stripextension='.txt', addextension='.indices_test.txt')

    def run(self):

        with open(self.in_all_featurenames().path,'r',encoding='utf-8') as infile:
            all_featurenames = infile.read().strip().split('\n')

        with open(self.in_selected_featurenames().path,'r',encoding='utf-8') as infile:
            selected_featurenames = infile.read().strip().split('\n')

        print('all_featurenames:',all_featurenames)
        print('selected_featurenames:',selected_featurenames)
        indices = [i for i,fn in enumerate(all_featurenames) if fn in selected_featurenames]
        print('Indices:',indices)

        with open(self.out_features_index().path, 'w', encoding='utf-8') as out:
            out.write(' '.join([str(x) for x in indices]))
