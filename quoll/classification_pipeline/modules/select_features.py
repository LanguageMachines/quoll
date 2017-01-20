
import numpy
from scipy import sparse
import glob
import random

from luiginlp.engine import Task, WorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, IntParameter, BoolParameter

from quoll.classification_pipeline.modules import run_ga, make_bins
from quoll.classification_pipeline.functions import ga_functions, vectorizer, docreader, linewriter

################################################################################
###Component to thread the tasks together
################################################################################

@registercomponent
class SelectFeatures(WorkflowComponent):

    trainvectors = Parameter()
    trainlabels = Parameter()
    parameter_options = Parameter()
    feature_names = Parameter()
    documents = Parameter()

    training_split = IntParameter(default=10)
    num_iterations = IntParameter(default=300)
    population_size = IntParameter(default=100)
    crossover_probability = Parameter(default='0.9')
    mutation_rate = Parameter(default='0.3')
    tournament_size = IntParameter(default=2)
    n_crossovers = IntParameter(default=1)
    classifier = Parameter(default='svm')
    ordinal = BoolParameter(default=False)
    fitness_metric = Parameter(default='microF1')
    stop_condition = IntParameter(default=5)

    def accepts(self):
        return [ ( InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='parameter_options', extension='.txt', inputparameter='parameter_options'), InputFormat(self, format_id='feature_names', extension='.txt', inputparameter='feature_names'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents') ) ]

    def setup(self, workflow, input_feeds):

        binner = workflow.new_task('make_bins', make_bins.MakeBins, autopass=True, n=self.training_split)
        binner.in_labels = input_feeds['trainlabels']

        foldrunner = workflow.new_task('run_folds', RunFoldsGA, autopass=True, n=self.training_split, num_iterations=self.num_iterations, population_size=self.population_size, crossover_probability=self.crossover_probability, mutation_rate=self.mutation_rate, tournament_size=self.tournament_size, n_crossovers=self.n_crossovers, classifier=self.classifier, ordinal=self.ordinal, fitness_metric=self.fitness_metric, stop_condition=self.stop_condition)
        foldrunner.in_bins = binner.out_bins
        foldrunner.in_vectors = input_feeds['trainvectors']
        foldrunner.in_labels = input_feeds['trainlabels']
        foldrunner.in_parameter_options = input_feeds['parameter_options']
        foldrunner.in_documents = input_feeds['documents']

        foldreporter = workflow.new_task('report_folds', ReportFoldsGA, autopass=False, fitness_metric=self.fitness_metric)
        foldreporter.in_feature_selection_directory = foldrunner.out_feature_selection
        foldreporter.in_trainvectors = input_feeds['trainvectors']
        foldreporter.in_parameter_options = input_feeds['parameter_options']
        foldreporter.in_feature_names = input_feeds['feature_names']

        return foldreporter


################################################################################
###Feature selection wrapper
################################################################################

class RunFoldsGA(Task):

    in_bins = InputSlot()
    in_vectors = InputSlot()
    in_labels = InputSlot()
    in_parameter_options = InputSlot()
    in_documents = InputSlot()

    n = IntParameter()
    num_iterations = IntParameter()
    population_size = IntParameter()
    crossover_probability = Parameter()
    mutation_rate = Parameter()
    tournament_size = IntParameter()
    n_crossovers = IntParameter()
    classifier = Parameter()
    ordinal = BoolParameter()
    fitness_metric = Parameter()
    stop_condition = IntParameter()

    def out_feature_selection(self):
        return self.outputfrominput(inputformat='bins', stripextension='.bins.csv', addextension='.' + self.classifier + '.feature_selection')

    def run(self):

        # make feature selection directory
        self.setup_output_dir(self.out_feature_selection().path)

        # for each fold
        for fold in range(self.n):
            yield FoldGA(directory=self.out_feature_selection().path, vectors=self.in_vectors().path, labels=self.in_labels().path, bins=self.in_bins().path, parameter_options=self.in_parameter_options().path, documents=self.in_documents().path, i=fold, num_iterations=self.num_iterations, population_size=self.population_size, crossover_probability=self.crossover_probability, mutation_rate=self.mutation_rate, tournament_size=self.tournament_size, n_crossovers=self.n_crossovers, classifier=self.classifier, ordinal=self.ordinal, fitness_metric=self.fitness_metric, stop_condition=self.stop_condition)


################################################################################
###Fold Wrapper
################################################################################

@registercomponent
class FoldGA(WorkflowComponent):

    directory = Parameter()
    vectors = Parameter()
    labels = Parameter()
    parameter_options = Parameter()
    documents = Parameter()
    bins = Parameter()

    i = IntParameter()
    num_iterations = IntParameter()
    population_size = IntParameter()
    crossover_probability = Parameter()
    mutation_rate = Parameter()
    tournament_size = IntParameter()
    n_crossovers = IntParameter()
    classifier = Parameter()
    ordinal = BoolParameter()
    fitness_metric = Parameter()
    stop_condition = IntParameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='directory',extension='.feature_selection',inputparameter='directory'), InputFormat(self,format_id='vectors',extension='.vectors.npz',inputparameter='vectors'), InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), InputFormat(self, format_id='parameter_options', extension='.txt', inputparameter='parameter_options'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self,format_id='bins',extension='.bins.csv',inputparameter='bins') ) ]

    def setup(self, workflow, input_feeds):

        fold = workflow.new_task('run_fold', FoldGATask, autopass=False, i=self.i, num_iterations=self.num_iterations, population_size=self.population_size, crossover_probability=self.crossover_probability, mutation_rate=self.mutation_rate, tournament_size=self.tournament_size, n_crossovers=self.n_crossovers, classifier=self.classifier, ordinal=self.ordinal, fitness_metric=self.fitness_metric, stop_condition=self.stop_condition)
        fold.in_directory = input_feeds['directory']
        fold.in_vectors = input_feeds['vectors']
        fold.in_labels = input_feeds['labels']
        fold.in_parameter_options = input_feeds['parameter_options']
        fold.in_documents = input_feeds['documents']
        fold.in_bins = input_feeds['bins']

        return fold

class FoldGATask(Task):

    in_directory = InputSlot()
    in_vectors = InputSlot()
    in_labels = InputSlot()
    in_parameter_options = InputSlot()
    in_documents = InputSlot()
    in_bins = InputSlot()

    i = IntParameter()
    num_iterations = IntParameter()
    population_size = IntParameter()
    crossover_probability = Parameter()
    mutation_rate = Parameter()
    tournament_size = IntParameter()
    n_crossovers = IntParameter()
    classifier = Parameter()
    ordinal = BoolParameter()
    fitness_metric = Parameter()
    stop_condition = IntParameter()

    def out_fold(self):
        return self.outputfrominput(inputformat='directory', stripextension='.feature_selection', addextension='.feature_selection/fold' + str(self.i))

    def out_trainvectors(self):
        return self.outputfrominput(inputformat='directory', stripextension='.feature_selection', addextension='.feature_selection/fold' + str(self.i) + '/train.vectors.npz')

    def out_testvectors(self):
        return self.outputfrominput(inputformat='directory', stripextension='.feature_selection', addextension='.feature_selection/fold' + str(self.i) + '/test.vectors.npz')

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.feature_selection', addextension='.feature_selection/fold' + str(self.i) + '/train.labels')

    def out_testlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.feature_selection', addextension='.feature_selection/fold' + str(self.i) + '/test.labels')

    def out_traindocuments(self):
        return self.outputfrominput(inputformat='directory', stripextension='.feature_selection', addextension='.feature_selection/fold' + str(self.i) + '/train.docs.txt')

    def out_testdocuments(self):
        return self.outputfrominput(inputformat='directory', stripextension='.feature_selection', addextension='.feature_selection/fold' + str(self.i) + '/test.docs.txt')

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


        # set training and test data
        train_vectors = sparse.vstack([instances[indices,:] for j,indices in enumerate(bins) if j != self.i])
        train_labels = numpy.concatenate([labels[indices] for j,indices in enumerate(bins) if j != self.i])
        train_documents = numpy.concatenate([documents[indices] for j,indices in enumerate(bins) if j != self.i])
        test_vectors = instances[bins[self.i]]
        test_labels = labels[bins[self.i]]
        test_documents = documents[bins[self.i]]

        # write experiment data to files in fold directory
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

        print('Running feature selection for fold',self.i)

        yield run_ga.RunGA(trainvectors=self.out_trainvectors().path, trainlabels=self.out_trainlabels().path, testvectors=self.out_testvectors().path, testlabels=self.out_testlabels().path, parameter_options=self.in_parameter_options().path, documents=self.out_testdocuments().path, num_iterations=self.num_iterations, population_size=self.population_size, crossover_probability=self.crossover_probability, mutation_rate=self.mutation_rate, tournament_size=self.tournament_size, n_crossovers=self.n_crossovers, classifier=self.classifier, ordinal=self.ordinal, fitness_metric=self.fitness_metric, stop_condition=self.stop_condition)


################################################################################
###Reporter
################################################################################

class ReportFoldsGA(Task):

    in_feature_selection_directory = InputSlot()
    in_trainvectors = InputSlot()
    in_parameter_options = InputSlot()
    in_feature_names = InputSlot()

    fitness_metric = Parameter()

    def out_folds_report(self):
        return self.outputfrominput(inputformat='feature_selection_directory', stripextension='.feature_selection', addextension='.foldsreport.txt')

    def out_best_trainvectors(self):
        return self.outputfrominput(inputformat='feature_selection_directory', stripextension='.feature_selection', addextension='.best_train.vectors.npz')

    def out_best_vectorsolution(self):
        return self.outputfrominput(inputformat='feature_selection_directory', stripextension='.feature_selection', addextension='.best_vectorsolution.txt')

    def out_best_features(self):
        return self.outputfrominput(inputformat='feature_selection_directory', stripextension='.feature_selection', addextension='.best_featurecombi.txt')

    def out_best_parameters(self):
        return self.outputfrominput(inputformat='feature_selection_directory', stripextension='.feature_selection', addextension='.best_classifier_parameters.txt')

    def run(self):

        # gather fold reports
        print('gathering fold reports')
        fold_reports = [ filename for filename in glob.glob(self.in_feature_selection_directory().path + '/fold*/train.ga.report.txt') ]
        fold_best_vectorsolutions = [ filename for filename in glob.glob(self.in_feature_selection_directory().path + '/fold*/train.ga.best_vectorsolution.txt') ]
        fold_best_parametersolutions = [ filename for filename in glob.glob(self.in_feature_selection_directory().path + '/fold*/train.ga.best_parametersolution.txt') ]

        # summarize reports
        highest = True if self.fitness_metric in ['microPrecision','microRecall','microF1','FPR','AUC','AAC'] else False
        dr = docreader.Docreader()
        best_fitness = 0 if highest else 1000
        candidate_iterations_solutions = []
        fold_best_fitness = []
        for i,report in enumerate(fold_reports):
            parsed = dr.parse_txt(report,delimiter='\t',header=True)
            iteration, solution = parsed[-1][0].split(',')
            iteration_index = int(iteration.split()[2])
            solution_index = int(solution.split()[2])
            best_fitness_iteration = float(parsed[iteration_index][2])
            fold_best_fitness.append(best_fitness_iteration)
            if (highest and best_fitness_iteration > best_fitness) or (not highest and best_fitness_iteration < best_fitness):
                best_fitness = best_fitness_iteration
                candidate_iterations_solutions = [[i,iteration_index,solution_index]]
            elif best_fitness_iteration == best_fitness:
                candidate_iterations_solutions.append([i,iteration_index,solution_index])
        avg = numpy.mean(fold_best_fitness)
        median = numpy.median(fold_best_fitness)
        best = max(fold_best_fitness) if highest else min(fold_best_fitness)
        selection_best = random.choice(candidate_iterations_solutions)
        fold_best = selection_best[0]
        fold_iteration_best = selection_best[1]
        fold_iteration_index_best = selection_best[2]
        final_report = [['Average fold best fitness:',str(avg)],['Median fold best fitness',str(median)],['Best fitness',str(best)],['Best fitness fold',str(fold_best)],['Best fitness iteration',str(fold_iteration_best)],['Best fitness index',str(fold_iteration_index_best)]]
        lw = linewriter.Linewriter(final_report)
        lw.write_csv(self.out_folds_report().path)

        ### manage output best training vectors

        # extract best solution 
        best_vectorsolution_file = fold_best_vectorsolutions[fold_best]
        with open(best_vectorsolution_file) as infile:
            best_solution = infile.read().strip().split()
        best_solution_features = [int(x) for x in best_solution]

        # write best solution
        with open(self.out_best_vectorsolution().path,'w',encoding='utf-8') as outfile:
            outfile.write(' '.join(best_solution))

        # extract training vectors
        loader = numpy.load(self.in_trainvectors().path)
        train_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # transform training vectors based on best solution
        transformed_traininstances = vectorizer.compress_vectors(train_instances,best_solution_features)

        # write to file
        numpy.savez(self.out_best_trainvectors().path, data=transformed_traininstances.data, indices=transformed_traininstances.indices, indptr=transformed_traininstances.indptr, shape=transformed_traininstances.shape)

        ### manage output best feature combination
        
        # extract feature names
        with open(self.in_feature_names().path,'r',encoding='utf-8') as infile:
            feature_names = numpy.array(infile.read().strip().split('\n'))

        # select feature names of best solution
        best_feature_combi = list(feature_names[best_solution_features])

        # write to file
        with open(self.out_best_features().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(feature_names))

        ### manage output best parameters
        
        # extract indices best parameter solution
        best_parametersolution_file = fold_best_parametersolutions[fold_best]
        with open(best_parametersolution_file) as infile:
            best_solution_parameters = [int(x) for x in infile.read().strip().split()]

        # load parameter values
        with open(self.in_parameter_options().path) as infile:
            lines = infile.read().strip().split('\n')
            parameter_options = [line.split() for line in lines]

        # extract best parameters
        best_parameter_combi = [variable[best_solution_parameters[i]] for i,variable in enumerate(parameter_options)]
        
        # write to file
        with open(self.out_best_parameters().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(best_parameter_combi))
