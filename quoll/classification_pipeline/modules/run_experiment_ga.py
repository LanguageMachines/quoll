
import numpy
from scipy import sparse
import pickle

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.modules import vectorize_instances, vectorize_sparse_instances, classify_instances, report_performance, make_bins, select_features

@registercomponent
class ExperimentComponentVectorFeatureSelection(WorkflowComponent):

    train = Parameter()
    trainlabels = Parameter()
    parameter_options = Parameter()
    test = Parameter()
    testlabels = Parameter()
    feature_names = Parameter()
    traindocuments = Parameter()
    testdocuments = Parameter()

    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter(default=False)
    training_split = IntParameter(default=10)
    num_iterations = IntParameter(default=300)
    population_size = IntParameter(default=100)
    crossover_probability = Parameter(default='0.9')
    mutation_rate = Parameter(default='0.3')
    tournament_size = IntParameter(default=2)
    n_crossovers = IntParameter(default=1)
    fitness_metric = Parameter(default='microF1')
    stop_condition = IntParameter(default=5)

    def accepts(self):
        return [ ( InputFormat(self,format_id='train',extension='.vectors.npz',inputparameter='train'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='test', extension='.vectors.npz',inputparameter='test'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self, format_id='parameter_options', extension='.txt', inputparameter='parameter_options'), InputFormat(self, format_id='feature_names', extension='.txt', inputparameter='feature_names'), InputFormat(self,format_id='traindocuments',extension='.txt',inputparameter='traindocuments'), InputFormat(self,format_id='testdocuments',extension='.txt',inputparameter='testdocuments') ) ]

    def setup(self, workflow, input_feeds):

        binner = workflow.new_task('make_bins', make_bins.MakeBins, autopass=True, n=self.training_split)
        binner.in_labels = input_feeds['trainlabels']

        foldrunner = workflow.new_task('run_folds', select_features.RunFoldsGA, autopass=True, n=self.training_split, num_iterations=self.num_iterations, population_size=self.population_size, crossover_probability=self.crossover_probability, mutation_rate=self.mutation_rate, tournament_size=self.tournament_size, n_crossovers=self.n_crossovers, classifier=self.classifier, ordinal=self.ordinal, fitness_metric=self.fitness_metric, stop_condition=self.stop_condition)
        foldrunner.in_bins = binner.out_bins
        foldrunner.in_vectors = input_feeds['train']
        foldrunner.in_labels = input_feeds['trainlabels']
        foldrunner.in_parameter_options = input_feeds['parameter_options']
        foldrunner.in_documents = input_feeds['traindocuments']

        foldreporter = workflow.new_task('report_folds', select_features.ReportFoldsGA, autopass=True)
        foldreporter.in_feature_selection_directory = foldrunner.out_feature_selection
        foldreporter.in_trainvectors = input_feeds['train']
        foldreporter.in_parameter_options = input_feeds['parameter_options']
        foldreporter.in_feature_names = input_feeds['feature_names']

        trainer = workflow.new_task('train_classifier', classify_instances.TrainClassifier, autopass=True, classifier=self.classifier)
        trainer.in_train = foldreporter.out_best_trainvectors
        trainer.in_trainlabels = input_feeds['trainlabels']
        trainer.in_classifier_args = foldreporter.out_best_parameters

        testtransformer = workflow.new_task('transform testvectors', vectorize_instances.TransformVectorsTask, autopass=True)
        testtransformer.in_vectors = input_feeds['test']
        testtransformer.in_selection = foldreporter.out_best_vectorsolution

        predictor = workflow.new_task('apply_classifier', classify_instances.ApplyClassifier, autopass=True)
        predictor.in_test = testtransformer.out_vectors
        predictor.in_labels = input_feeds['trainlabels']
        predictor.in_model = trainer.out_model

        reporter = workflow.new_task('report_performance', report_performance.ReportPerformance, autopass=True, ordinal=self.ordinal)
        reporter.in_predictions = predictor.out_classifications
        reporter.in_labels = input_feeds['testlabels']
        reporter.in_documents = input_feeds['testdocuments']

        return reporter
