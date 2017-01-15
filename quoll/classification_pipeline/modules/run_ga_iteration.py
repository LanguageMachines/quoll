
import os
import glob
import numpy
from scipy import sparse

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.modules import run_experiment
from quoll.classification_pipeline.functions import ga_functions, vectorizer, docreader

################################################################################
###Component to thread the tasks together
################################################################################

@registercomponent
class RunGAIteration(WorkflowComponent):

    dir_latest_iter = Parameter()
    iteration_cursor = Parameter()
    trainvectors = Parameter()
    trainlabels = Parameter()
    testvectors = Parameter()
    testlabels = Parameter()
    parameter_options = Parameter()
    documents = Parameter()

    iteration = IntParameter()
    population_size = IntParameter()
    crossover_probability = Parameter(default='0.9')
    mutation_rate = Parameter(default='0.3')
    tournament_size = IntParameter(default=2)
    n_crossovers = IntParameter(default=1)
    classifier = Parameter(default='svm')
    ordinal = BoolParameter(default=False)
    fitness_metric = Parameter(default='microF1')

    def accepts(self):
        return [ ( InputFormat(self,format_id='dir_latest_iter',extension='.iteration',inputparameter='dir_latest_iter'), InputFormat(self,format_id='iteration_cursor',extension='.txt',inputparameter='iteration_cursor'), InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='testvectors', extension='.vectors.npz',inputparameter='testvectors'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self, format_id='parameter_options', extension='.txt', inputparameter='parameter_options'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents') ) ]

    def setup(self, workflow, input_feeds):

        offspring_generator = workflow.new_task('generate_offspring', GenerateOffspring, autopass=False, iteration=self.iteration, crossover_probability=self.crossover_probability, mutation_rate=self.mutation_rate, tournament_size=self.tournament_size, n_crossovers=self.n_crossovers, fitness_metric=self.fitness_metric)
        offspring_generator.in_dir_latest_iter = input_feeds['dir_latest_iter']

        fitness_manager = workflow.new_task('score_fitness_population', ScoreFitnessPopulation, autopass=False, population_size=self.population_size, classifier=self.classifier, ordinal=self.ordinal, fitness_metric=self.fitness_metric)
        fitness_manager.in_vectorpopulation = offspring_generator.out_vectoroffspring
        fitness_manager.in_parameterpopulation = offspring_generator.out_parameteroffspring
        fitness_manager.in_trainvectors = input_feeds['trainvectors']
        fitness_manager.in_trainlabels = input_feeds['trainlabels']
        fitness_manager.in_testvectors = input_feeds['testvectors']
        fitness_manager.in_testlabels = input_feeds['testlabels']
        fitness_manager.in_parameter_options = input_feeds['parameter_options']
        fitness_manager.in_documents = input_feeds['documents']

        fitness_reporter = workflow.new_task('report_fitness_population', ReportFitnessPopulation, autopass=False, fitness_metric=self.fitness_metric)
        fitness_reporter.in_fitness_exp = fitness_manager.out_fitness_exp

        iteration_monitor = workflow.new_task('monitor_iteration', MonitorIteration, autopass=False, fitness_metric=self.fitness_metric)
        iteration_monitor.in_iteration_cursor = input_feeds['iteration_cursor']
        iteration_monitor.in_current_iterdir = offspring_generator.out_iterationdir
        iteration_monitor.in_fitnessreport = fitness_reporter.out_fitnessreport 

        return iteration_monitor


################################################################################
###Offspring generator
################################################################################

class GenerateOffspring(Task):

    in_dir_latest_iter = InputSlot()

    iteration = IntParameter()
    crossover_probability = Parameter()
    mutation_rate = Parameter()
    tournament_size = IntParameter()
    n_crossovers = IntParameter()
    fitness_metric = Parameter()

    def out_iterationdir(self):
        return self.outputfrominput(inputformat='dir_latest_iter', stripextension='.' + str(self.iteration-1) + '.iteration', addextension='.' + str(self.iteration) + '.iteration')

    def out_vectoroffspring(self):
        return self.outputfrominput(inputformat='dir_latest_iter', stripextension='.' + str(self.iteration-1) + '.iteration', addextension='.' + str(self.iteration) + '.iteration/vectorpopulation.npz')

    def out_parameteroffspring(self):
        return self.outputfrominput(inputformat='dir_latest_iter', stripextension='.' + str(self.iteration-1) + '.iteration', addextension='.' + str(self.iteration) + '.iteration/parameterpopulation.npz')

    def run(self):

        # create output directory
        self.setup_output_dir(self.out_iterationdir().path)

        # load vectorpopulation
        vectorpopulationfile = self.in_dir_latest_iter().path + '/vectorpopulation.npz'
        loader = numpy.load(vectorpopulationfile)
        vectorpopulation = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load parameterpopulation
        parameterpopulationfile = self.in_dir_latest_iter().path + '/parameterpopulation.npz'
        loader = numpy.load(parameterpopulationfile)
        parameterpopulation = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # combine into population
        population = sparse.hstack([vectorpopulation,parameterpopulation]).tocsr()

        # load fitness
        fitnessfile = self.in_dir_latest_iter().path + '/vectorpopulation.fitness.txt'
        with open(fitnessfile,'r',encoding='utf-8') as infile:
            fitness = [float(value) for value in infile.read().split('\n')]

        # decide win_condition based on fitness metric
        win_condition = 'highest' if self.fitness_metric in ['microPrecision','microRecall','microF1','FPR','AUC','AAC'] else 'lowest'
        # generate offspring
        offspring = ga_functions.generate_offspring(population,fitness,tournament_size=self.tournament_size,crossover_prob=float(self.crossover_probability),n_crossovers=self.n_crossovers,mutation_rate=float(self.mutation_rate),win_condition=win_condition)

        ### unwind offspring

        # extract number of parameter values
        vector_length = vectorpopulation.shape[1]
        
        # extract vectorpopulation of offspring
        offspring_vectorpopulation = offspring[:,:vector_length]
        
        # extract parameterpopulation of offspring
        offspring_parameterpopulation = offspring[:,vector_length:]

        ### output offspring

        # output vector offspring
        numpy.savez(self.out_vectoroffspring().path, data=offspring_vectorpopulation.data, indices=offspring_vectorpopulation.indices, indptr=offspring_vectorpopulation.indptr, shape=offspring_vectorpopulation.shape)

        # output parameter offspring
        numpy.savez(self.out_parameteroffspring().path, data=offspring_parameterpopulation.data, indices=offspring_parameterpopulation.indices, indptr=offspring_parameterpopulation.indptr, shape=offspring_parameterpopulation.shape)

################################################################################
###Fitness manager
################################################################################

class ScoreFitnessPopulation(Task):

    in_vectorpopulation = InputSlot()
    in_parameterpopulation = InputSlot()
    in_trainvectors = InputSlot()
    in_trainlabels = InputSlot()
    in_testvectors = InputSlot()
    in_testlabels = InputSlot()
    in_parameter_options = InputSlot()
    in_documents = InputSlot()

    population_size = IntParameter()
    classifier = Parameter()
    ordinal = BoolParameter()

    def out_fitness_exp(self):
        return self.outputfrominput(inputformat='vectorpopulation', stripextension='.npz', addextension='.fitness_exp')

    def run(self):

        # create output directory
        self.setup_output_dir(self.out_fitness_exp().path)

        # score fitness for each solution
        yield [ ScoreFitnessSolution(fitness_exp=self.out_fitness_exp().path, vectorpopulation=self.in_vectorpopulation().path, parameterpopulation=self.in_parameterpopulation().path, trainvectors=self.in_trainvectors().path, trainlabels=self.in_trainlabels().path, testvectors=self.in_testvectors().path, testlabels=self.in_testlabels().path, parameter_options=self.in_parameter_options().path, documents=self.in_documents().path, solution_index=i, classifier=self.classifier, ordinal=self.ordinal) for i in range(self.population_size) ]


################################################################################
###Fitness assessor
################################################################################

@registercomponent
class ScoreFitnessSolution(WorkflowComponent):

    fitness_exp = Parameter()
    vectorpopulation = Parameter()
    parameterpopulation = Parameter()
    trainvectors = Parameter()
    trainlabels = Parameter()
    testvectors = Parameter()
    testlabels = Parameter()
    parameter_options = Parameter()
    documents = Parameter()

    solution_index = IntParameter()
    classifier = Parameter(default='svm')
    ordinal = BoolParameter(default=False)

    def accepts(self):
        return [ ( InputFormat(self,format_id='fitness_exp',extension='.fitness_exp',inputparameter='fitness_exp'), InputFormat(self,format_id='vectorpopulation',extension='.npz',inputparameter='vectorpopulation'), InputFormat(self,format_id='parameterpopulation',extension='.npz',inputparameter='parameterpopulation'), InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='testvectors', extension='.vectors.npz',inputparameter='testvectors'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self, format_id='parameter_options', extension='.txt', inputparameter='parameter_options'), InputFormat(self, format_id='documents', extension='.txt', inputparameter='documents') ) ]

    def setup(self, workflow, input_feeds):
        solution_fitness_assessor = workflow.new_task('score_fitness_solution_task', ScoreFitnessSolutionTask, autopass=False, solution_index=self.solution_index, classifier=self.classifier, ordinal=self.ordinal)
        solution_fitness_assessor.in_fitness_exp = input_feeds['fitness_exp']
        solution_fitness_assessor.in_vectorpopulation = input_feeds['vectorpopulation']
        solution_fitness_assessor.in_parameterpopulation = input_feeds['parameterpopulation']
        solution_fitness_assessor.in_trainvectors = input_feeds['trainvectors']
        solution_fitness_assessor.in_trainlabels = input_feeds['trainlabels']
        solution_fitness_assessor.in_testvectors = input_feeds['testvectors']
        solution_fitness_assessor.in_testlabels = input_feeds['testlabels']
        solution_fitness_assessor.in_parameter_options = input_feeds['parameter_options']
        solution_fitness_assessor.in_documents = input_feeds['documents']

        return solution_fitness_assessor

class ScoreFitnessSolutionTask(Task):

    in_fitness_exp = InputSlot()
    in_vectorpopulation = InputSlot()
    in_parameterpopulation = InputSlot()
    in_trainvectors = InputSlot()
    in_trainlabels = InputSlot()
    in_testvectors = InputSlot()
    in_testlabels = InputSlot()
    in_parameter_options = InputSlot()
    in_documents = InputSlot()

    solution_index = IntParameter()
    classifier = Parameter()
    ordinal = BoolParameter()

    def out_solutiondir(self):
        return self.outputfrominput(inputformat='fitness_exp', stripextension='.fitness_exp', addextension='.fitness_exp/solution_fitness' + str(self.solution_index))

    def out_solution_trainvectors(self):
        return self.outputfrominput(inputformat='fitness_exp', stripextension='.fitness_exp', addextension='.fitness_exp/solution_fitness' + str(self.solution_index) + '/train.vectors.npz')

    def out_solution_testvectors(self):
        return self.outputfrominput(inputformat='fitness_exp', stripextension='.fitness_exp', addextension='.fitness_exp/solution_fitness' + str(self.solution_index) + '/test.vectors.npz')

    def out_solution_classifier_args(self):
        return self.outputfrominput(inputformat='fitness_exp', stripextension='.fitness_exp', addextension='.fitness_exp/solution_fitness' + str(self.solution_index) + '/classifier_args.txt')

    def run(self):

        # generate outputdir
        self.setup_output_dir(self.out_solutiondir().path)

        ### load solution vectors

        # load train instances
        loader = numpy.load(self.in_trainvectors().path)
        vectorized_traininstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load test instances
        loader = numpy.load(self.in_testvectors().path)
        vectorized_testinstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load vectorsolution
        loader = numpy.load(self.in_vectorpopulation().path)
        vectorpopulation = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        vectorsolution = vectorpopulation[self.solution_index,:].nonzero()[1]

        # transform train and test vectors according to solution
        transformed_traininstances = vectorizer.compress_vectors(vectorized_traininstances,vectorsolution)
        transformed_testinstances = vectorizer.compress_vectors(vectorized_testinstances,vectorsolution)

        # write vectors to solutiondir
        numpy.savez(self.out_solution_trainvectors().path, data=transformed_traininstances.data, indices=transformed_traininstances.indices, indptr=transformed_traininstances.indptr, shape=transformed_traininstances.shape)
        numpy.savez(self.out_solution_testvectors().path, data=transformed_testinstances.data, indices=transformed_testinstances.indices, indptr=transformed_testinstances.indptr, shape=transformed_testinstances.shape)

        ### load solution parameters

        # load parameter options
        with open(self.in_parameter_options().path) as infile:
            lines = infile.read().strip().split('\n')
            parameter_options = [line.split() for line in lines]

        # load parameter solution
        loader = numpy.load(self.in_parameterpopulation().path)
        parameterpopulation = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        parametersolution = parameterpopulation[self.solution_index,:].toarray().tolist()[0]

        # extract solution classifier arguments
        classifier_args = [parameter_options[i][paramindex] for i,paramindex in enumerate(parametersolution)]

        # write classifier arguments to solutiondir
        with open(self.out_solution_classifier_args().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(classifier_args))

        ### perform classification and report outcomes
        yield run_experiment.ExperimentComponentVector(train=self.out_solution_trainvectors().path, trainlabels=self.in_trainlabels().path, test=self.out_solution_testvectors().path, testlabels=self.in_testlabels().path, classifier_args=self.out_solution_classifier_args().path, documents=self.in_documents().path, classifier=self.classifier, ordinal=self.ordinal)


################################################################################
###Fitness reporter
################################################################################
class ReportFitnessPopulation(Task):

    in_fitness_exp = InputSlot()

    fitness_metric = Parameter()

    def out_fitnessreport(self):
        return self.outputfrominput(inputformat='fitness_exp', stripextension='.fitness_exp', addextension='.fitness.txt')

    def run(self):

        # gather fold reports
        print('gathering solution experiment reports')
        performance_files = sorted([ filename for filename in glob.glob(self.in_fitness_exp().path + '/solution_fitness*/test.performance.csv') ])

        # extract fitness score from reports
        metric_position = {'microPrecision':[-1,1],'microRecall':[-1,2],'microF1':[-1,3],'FPR':[-1,5],'AUC':[-1,6],'MAE':[-1,7],'RMSE':[-1,8],'ACC':[-1,9]}
        coordinates = metric_position[self.fitness_metric]
        dr = docreader.Docreader()
        performance_combined = [dr.parse_csv(performance_file) for performance_file in performance_files]
        all_fitness = [performance[coordinates[0]][coordinates[1]] for performance in performance_combined]
        with open(self.out_fitnessreport().path,'w') as outfile:
            outfile.write('\n'.join(all_fitness))

        # remove fitness experiment
        print('removing experiment directory')
        os.system('rm -r ' + self.in_fitness_exp().path)


################################################################################
###Iteration monitor
################################################################################
class MonitorIteration(Task):

    in_iteration_cursor = InputSlot()
    in_current_iterdir = InputSlot()
    in_fitnessreport = InputSlot()

    fitness_metric = Parameter()

    def out_iteration_cursor(self):
        return self.outputfrominput(inputformat='current_iterdir', stripextension='.iteration', addextension='.iteration/cursor.txt')

    def run(self):

        # load cursor of last iteration
        with open(self.in_iteration_cursor().path) as infile:
            last_best, last_best_since = [float(x) for x in infile.read().strip().split()]

        # load fitness report and extract highest score
        highest = True if self.fitness_metric in ['microPrecision','microRecall','microF1','FPR','AUC','AAC'] else False
        with open(self.in_fitnessreport().path) as infile:
            fitness_scores = [float(score) for score in infile.read().strip().split('\n')]

        # compare score to score of last cursor
        best_fitness = max(fitness_scores) if highest else min(fitness_scores)
        if (best_fitness < last_best and not highest) or (best_fitness > last_best and highest):
            last_best = best_fitness
            last_best_since = 1
        else:
            last_best_since += 1

        # write new stats
        with open(self.out_iteration_cursor().path,'w',encoding='utf-8') as outfile:
            outfile.write(' '.join([str(last_best),str(last_best_since)]))
