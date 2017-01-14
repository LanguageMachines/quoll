
from luiginlp.engine import Task, WorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, IntParameter, BoolParameter
import numpy
from scipy import sparse
from collections import defaultdict
import glob

import quoll.classification_pipeline.functions.nfold_cv_functions as nfold_cv_functions
import quoll.classification_pipeline.functions.linewriter as linewriter
import quoll.classification_pipeline.functions.docreader as docreader

from quoll.classification_pipeline.modules.select_features import SelectFeatures
from quoll.classification_pipeline.modules.run_experiment_ga import ExperimentComponentVectorFeatureSelection 
from quoll.classification_pipeline.modules.make_bins import MakeBins
from quoll.classification_pipeline.modules.run_nfold_cv_vectors import ReportFolds  

################################################################################
###Component to thread the tasks together
################################################################################

@registercomponent
class NFoldCVGA(WorkflowComponent):
    
    vectors = Parameter()
    labels = Parameter()
    parameter_options = Parameter()
    feature_names = Parameter()
    documents = Parameter()

    n = IntParameter(default=10)
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
        return [ ( InputFormat(self,format_id='vectors',extension='.vectors.npz',inputparameter='vectors'), InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), InputFormat(self, format_id='parameter_options', extension='.txt', inputparameter='parameter_options'), InputFormat(self, format_id='feature_names', extension='.txt', inputparameter='feature_names'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents') ) ]
    
    def setup(self, workflow, input_feeds):

        bin_maker = workflow.new_task('make_bins', MakeBins, autopass=True, n=self.n)
        bin_maker.in_labels = input_feeds['labels']

        fold_runner = workflow.new_task('run_folds_vectors_ga', RunFoldsVectorsGA, autopass=True, n=self.n, classifier=self.classifier, training_split=self.training_split, num_iterations=self.num_iterations, population_size=self.population_size, crossover_probability=self.crossover_probability, mutation_rate=self.mutation_rate, tournament_size = self.tournament_size, n_crossovers=self.n_crossovers, ordinal=self.ordinal, fitness_metric=self.fitness_metric, stop_condition=self.stop_condition)
        fold_runner.in_bins = bin_maker.out_bins
        fold_runner.in_vectors = input_feeds['vectors']
        fold_runner.in_labels = input_feeds['labels']
        fold_runner.in_parameter_options = input_feeds['parameter_options']
        fold_runner.in_feature_names = input_feeds['feature_names']
        fold_runner.in_documents = input_feeds['documents']        

        folds_reporter = workflow.new_task('report_folds', ReportFolds, autopass = False)
        folds_reporter.in_expdirectory = fold_runner.out_exp

        return folds_reporter

    
################################################################################
###Experiment wrapper
################################################################################

class RunFoldsVectorsGA(Task):

    in_bins = InputSlot()
    in_vectors = InputSlot()
    in_labels = InputSlot()
    in_parameter_options = InputSlot()
    in_feature_names = InputSlot()
    in_documents = InputSlot()

    n = IntParameter()
    classifier = Parameter()
    ordinal = BoolParameter()
    training_split = IntParameter()
    num_iterations = IntParameter()
    population_size = IntParameter()
    crossover_probability = Parameter()
    mutation_rate = Parameter()
    tournament_size = IntParameter()
    n_crossovers = IntParameter()
    fitness_metric = Parameter()
    stop_condition = IntParameter()

    def out_exp(self):
        return self.outputfrominput(inputformat='bins', stripextension='.bins.csv', addextension='.' + self.classifier + '.exp')
        
    def run(self):

        # make experiment directory
        self.setup_output_dir(self.out_exp().path)

        # for each fold
        for fold in range(self.n):
            yield FoldVectorsGA(directory=self.out_exp().path, vectors=self.in_vectors().path, labels=self.in_labels().path, bins=self.in_bins().path, parameter_options=self.in_parameter_options().path, feature_names=self.in_feature_names().path, documents=self.in_documents().path, i=fold, classifier=self.classifier, training_split=self.training_split, num_iterations=self.num_iterations, population_size=self.population_size, crossover_probability=self.crossover_probability, mutation_rate=self.mutation_rate, tournament_size = self.tournament_size, n_crossovers=self.n_crossovers, ordinal=self.ordinal, fitness_metric=self.fitness_metric, stop_condition=self.stop_condition)


################################################################################
###Fold Wrapper
################################################################################

@registercomponent
class FoldVectorsGA(WorkflowComponent):

    directory = Parameter()
    vectors = Parameter()
    labels = Parameter()
    parameter_options = Parameter()
    feature_names = Parameter()
    documents = Parameter()
    bins = Parameter()

    i = IntParameter()
    classifier = Parameter()
    ordinal = BoolParameter()
    training_split = IntParameter()
    num_iterations = IntParameter()
    population_size = IntParameter()
    crossover_probability = Parameter()
    mutation_rate = Parameter()
    tournament_size = IntParameter()
    n_crossovers = IntParameter()
    fitness_metric = Parameter()
    stop_condition = IntParameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='directory',extension='.exp',inputparameter='directory'), InputFormat(self,format_id='vectors',extension='.vectors.npz',inputparameter='vectors'), InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), InputFormat(self, format_id='parameter_options', extension='.txt', inputparameter='parameter_options'), InputFormat(self, format_id='feature_names', extension='.txt', inputparameter='feature_names'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self,format_id='bins',extension='.bins.csv',inputparameter='bins') ) ]
    
    def setup(self, workflow, input_feeds):

        fold = workflow.new_task('run_fold', FoldVectorsGATask, autopass=False, i=self.i, classifier=self.classifier, training_split=self.training_split, num_iterations=self.num_iterations, population_size=self.population_size, crossover_probability=self.crossover_probability, mutation_rate=self.mutation_rate, tournament_size = self.tournament_size, n_crossovers=self.n_crossovers, ordinal=self.ordinal, fitness_metric=self.fitness_metric, stop_condition=self.stop_condition)
        fold.in_directory = input_feeds['directory']
        fold.in_vectors = input_feeds['vectors']
        fold.in_labels = input_feeds['labels']
        fold.in_parameter_options = input_feeds['parameter_options']
        fold.in_feature_names = input_feeds['feature_names']
        fold.in_documents = input_feeds['documents']     
        fold.in_bins = input_feeds['bins']   

        return fold

class FoldVectorsGATask(Task):

    in_directory = InputSlot()
    in_vectors = InputSlot()
    in_labels = InputSlot()
    in_parameter_options = InputSlot()
    in_feature_names = InputSlot()
    in_documents = InputSlot()
    in_bins = InputSlot()
    
    i = IntParameter()
    classifier = Parameter()
    ordinal = BoolParameter()
    training_split = IntParameter()
    num_iterations = IntParameter()
    population_size = IntParameter()
    crossover_probability = Parameter()
    mutation_rate = Parameter()
    tournament_size = IntParameter()
    n_crossovers = IntParameter()
    fitness_metric = Parameter()
    stop_condition = IntParameter()


    def out_fold(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i))    

    def out_trainvectors(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.vectors.npz')

    def out_testvectors(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.vectors.npz')

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.labels')

    def out_testlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.labels')

    def out_traindocuments(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/train.docs.txt')

    def out_testdocuments(self):
        return self.outputfrominput(inputformat='directory', stripextension='.exp', addextension='.exp/fold' + str(self.i) + '/test.docs.txt')

    def run(self):

        # make fold directory
        self.setup_output_dir(self.out_fold().path)

        # open bin indices
        dr = docreader.Docreader()
        bins_str = dr.parse_csv(self.in_bins().path)
        bins = [[int(x) for x in bin] for bin in bins_str]

        # open instances
        loader = numpy.load(self.in_vectors().path)
        instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # open labels
        with open(self.in_labels().path,'r',encoding='utf-8') as infile:
            labels = numpy.array(infile.read().strip().split('\n'))

        # open documents
        with open(self.in_documents().path,'r',encoding='utf-8') as infile:
            documents = numpy.array(infile.read().split('\n'))

        # write data to files in fold directory
        train_vectors = sparse.vstack([instances[indices,:] for j,indices in enumerate(bins) if j != self.i])
        train_labels = numpy.concatenate([labels[indices] for j,indices in enumerate(bins) if j != self.i])
        train_documents = numpy.concatenate([documents[indices] for j,indices in enumerate(bins) if j != self.i])
        test_vectors = instances[bins[self.i]]
        test_labels = labels[bins[self.i]]
        test_documents = documents[bins[self.i]]
        numpy.savez(self.out_trainvectors().path, data=train_vectors.data, indices=train_vectors.indices, indptr=train_vectors.indptr, shape=train_vectors.shape)
        numpy.savez(self.out_testvectors().path, data=test_vectors.data, indices=test_vectors.indices, indptr=test_vectors.indptr, shape=test_vectors.shape)
        with open(self.out_trainlabels().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(train_labels))
        with open(self.out_testlabels().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_labels))
        with open(self.out_traindocuments().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(train_documents))
        with open(self.out_testdocuments().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_documents))

        print('Running experiment for fold',self.i)
        yield ExperimentComponentVectorFeatureSelection(train=self.out_trainvectors().path, trainlabels=self.out_trainlabels().path, test=self.out_testvectors().path, testlabels=self.out_testlabels().path, parameter_options=self.in_parameter_options().path, feature_names=self.in_feature_names().path, traindocuments=self.out_traindocuments().path, testdocuments=self.out_testdocuments().path, training_split=self.training_split, num_iterations=self.num_iterations, population_size=self.population_size, crossover_probability=self.crossover_probability, mutation_rate=self.mutation_rate, tournament_size=self.tournament_size, n_crossovers=self.n_crossovers, classifier=self.classifier, ordinal=self.ordinal, fitness_metric=self.fitness_metric, stop_condition=self.stop_condition) 
