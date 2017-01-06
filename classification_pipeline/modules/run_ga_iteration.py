
import glob
import numpy
from scipy import sparse

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from modules import run_experiment
from functions import ga_functions, vectorizer, docreader

################################################################################
###Component to thread the tasks together
################################################################################

@registercomponent
class RunGAIteration(WorkflowComponent):
    
    dir_latest_iter = Parameter()
    trainvectors = Parameter()
    trainlabels = Parameter()
    testvectors = Parameter()
    testlabels = Parameter()
    documents = Parameter()

    iteration = IntParameter()
    population_size = IntParameter()
    crossover_probability = Parameter(default='0.9')
    mutation_rate = Parameter(default='0.3') 
    tournament_size = IntParameter(default=2)
    n_crossovers = IntParameter(default=1)
    classifier = Parameter(default='svm')
    classifier_args = Parameter(default=False)
    fitness_metric = Parameter(default='microF1')

    def accepts(self):
        return [ ( InputFormat(self,format_id='dir_latest_iter',extension='.iteration',inputparameter='dir_latest_iter'), InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.vectorlabels', inputparameter='trainlabels'), InputFormat(self, format_id='testvectors', extension='.vectors.npz',inputparameter='testvectors'), InputFormat(self, format_id='testlabels', extension='.vectorlabels', inputparameter='testlabels'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents') ) ]
                                
    def setup(self, workflow, input_feeds):

        offspring_generator = workflow.new_task('generate_offspring', GenerateOffspring, autopass=False, iteration=self.iteration, crossover_probability=self.crossover_probability, mutation_rate=self.mutation_rate, tournament_size=self.tournament_size, n_crossovers=self.n_crossovers)
        offspring_generator.in_dir_latest_iter = input_feeds['dir_latest_iter']

        fitness_manager = workflow.new_task('score_fitness_population', ScoreFitnessPopulation, autopass=False, population_size=self.population_size, classifier=self.classifier, classifier_args=self.classifier_args)
        fitness_manager.in_population = offspring_generator.out_offspring
        fitness_manager.in_trainvectors = input_feeds['trainvectors']
        fitness_manager.in_trainlabels = input_feeds['trainlabels']
        fitness_manager.in_testvectors = input_feeds['testvectors']
        fitness_manager.in_testlabels = input_feeds['testlabels']
        fitness_manager.in_documents = input_feeds['documents']

        fitness_reporter = workflow.new_task('report_fitness_population', ReportFitnessPopulation, autopass=False, fitness_metric=self.fitness_metric)
        fitness_reporter.in_fitness_exp = fitness_manager.out_fitness_exp

        return fitness_reporter


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

    def out_iterationdir(self):
        return self.outputfrominput(inputformat='in_dir_latest_iter', stripextension='.' + str(self.iteration-1) + '.iteration', addextension='.' + str(self.iteration) + '.iteration')

    def out_offspring(self):
        return self.outputfrominput(inputformat='in_dir_latest_iter', stripextension='.' + str(self.iteration-1) + '.iteration', addextension='.' + str(self.iteration) + '.iteration/population.npz')

    def run(self):

        # create output directory
        self.setup_output_dir(self.out_iterationdir().path)

        # load population
        populationfile = self.in_dir_latest_iter().path + '/population.npz'
        loader = numpy.load(populationfile)
        population = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load fitness 
        fitnessfile = self.in_dir_latest_iter().path + '/population.fitness.txt'
        with open(fitnessfile,'r',encoding='utf-8') as infile:
            fitness = [float(value) for value in infile.read().split('\n')]

        # generate offspring
        offspring = ga_functions.generate_offspring(population,fitness,tournament_size=self.tournament_size,crossover_prob=float(self.crossover_prob),n_crossovers=self.n_crossovers,mutation_rate=float(self.mutation_rate))

        # output offspring
        numpy.savez(self.out_offspring().path, data=offspring.data, indices=offspring.indices, indptr=offspring.indptr, shape=offspring.shape)


################################################################################
###Fitness manager
################################################################################

class ScoreFitnessPopulation(Task):

    in_population = InputSlot()
    in_trainvectors = InputSlot()
    in_trainlabels = InputSlot()
    in_testvectors = InputSlot()
    in_testlabels = InputSlot()
    in_documents = InputSlot()

    population_size = IntParameter()
    classifier = Parameter()
    classifier_args = Parameter()

    def out_fitness_exp(self):
        return self.outputfrominput(inputformat='population', stripextension='.npz', addextension='.fitness_exp')

    def run(self):

        # create output directory
        self.setup_output_dir(self.out_fitness_exp().path)

        # score fitness for each solution
        yield [ ScoreFitnessSolution(fitness_exp=self.out_fitness_exp().path, population=self.in_population().path, trainvectors=self.in_trainvectors().path, trainlabels=self.in_trainlabels().path, testvectors=self.in_testvectors().path, testlabels=self.in_testlabels().path, documents=self.in_documents().path, solution_index=i, classifier=self.classifier, classifier_args=self.classifier_args) for i in range(self.population_size) ]


################################################################################
###Fitness assessor
################################################################################

@registercomponent
class ScoreFitnessSolution(WorkflowComponent):

    fitness_exp = Parameter()
    population = Parameter()
    trainvectors = Parameter()
    trainlabels = Parameter()
    testvectors = Parameter()
    testlabels = Parameter()
    documents = Parameter()

    solution_index = IntParameter()
    classifier = Parameter(default='svm')
    classifier_args = Parameter(default=False)

    def accepts(self):
        return [ ( InputFormat(self,format_id='fitness_exp',extension='.fitness_exp',inputparameter='fitness_exp'), InputFormat(self,format_id='population',extension='.npz',inputparameter='population'), InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='testvectors', extension='.vectors.npz',inputparameter='testvectors'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels') ) ]
                                
    def setup(self, workflow, input_feeds):
        solution_fitness_assessor = workflow.new_task('score_fitness_solution_task', ScoreFitnessSolutionTask, autopass=False, solution_index=self.solution_index, classifier=self.classifier, classifier_args=self.classifier_args)
        solution_fitness_assessor.in_fitness_exp = input_feeds['fitness_exp']
        solution_fitness_assessor.in_population = input_feeds['population']
        solution_fitness_assessor.in_trainvectors = input_feeds['trainvectors']
        solution_fitness_assessor.in_trainlabels = input_feeds['trainlabels']
        solution_fitness_assessor.in_testvectors = input_feeds['testvectors']
        solution_fitness_assessor.in_testlabels = input_feeds['testlabels']

        return solution_fitness_assessor

class ScoreFitnessSolutionTask(Task):

    in_fitness_exp = InputSlot()
    in_population = InputSlot()
    in_trainvectors = InputSlot()
    in_trainlabels = InputSlot()
    in_testvectors = InputSlot()
    in_testlabels = InputSlot()
    in_documents = InputSlot()

    solution_index = IntParameter()
    classifier = Parameter()
    classifier_args = Parameter()

    def out_solutiondir(self):
        return self.outputfrominput(inputformat='fitness_exp', stripextension='.fitness_exp', addextension='.fitness_exp/solution_fitness' + str(self.solution_index))

    def out_solution_trainvectors(self):
        return self.outputfrominput(inputformat='fitness_exp', stripextension='.fitness_exp', addextension='.fitness_exp/solution_fitness' + str(self.solution_index) + '/train.vectors.npz')

    def out_solution_testvectors(self):
        return self.outputfrominput(inputformat='fitness_exp', stripextension='.fitness_exp', addextension='.fitness_exp/solution_fitness' + str(self.solution_index) + '/test.vectors.npz')

    def run(self):

        # generate outputdir
        self.setup_output_dir(self.out_solutiondir().path)

        # load train instances
        loader = numpy.load(self.in_trainvectors().path)
        vectorized_traininstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        
        # load test instances
        loader = numpy.load(self.in_testvectors().path)
        vectorized_testinstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        
        # load solution
        loader = numpy.load(self.in_population().path)
        population = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape']) 
        solution = population[self.solution_index,:].nonzero()[1]      
        
        # transform train and test vectors according to solution
        transformed_traininstances = vectorizer.compress_vectors(vectorized_traininstances,solution)
        transformed_testinstances = vectorizer.compress_vectors(vectorized_testinstances,solution)
        
        # write vectors to solutiondir
        numpy.savez(self.out_solution_trainvectors().path, data=transformed_traininstances.data, indices=transformed_traininstances.indices, indptr=transformed_traininstances.indptr, shape=transformed_traininstances.shape)
        numpy.savez(self.out_solution_testvectors().path, data=transformed_testinstances.data, indices=transformed_testinstances.indices, indptr=transformed_testinstances.indptr, shape=transformed_testinstances.shape)

        # perform classification and report outcomes
        yield run_experiment.ExperimentComponentVector(train=self.out_solution_trainvectors().path, trainlabels=self.in_trainlabels().path, test=self.out_solution_testvectors().path, testlabels=self.in_testlabels().path, documents=self.in_documents().path, classifier=self.classifier, classifier_args=self.classifier_args)


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
        metric_position = {'microF1':[-1,3]}
        coordinates = metric_position[self.fitness_metric]
        dr = docreader.Docreader()
        performance_combined = [dr.parse_csv(performance_file) for performance_file in performance_files]
        all_fitness = [performance[coordinates[0]][coordinates[1]] for performance in performance_combined]
        with open(self.out_fitnessreport().path,'w') as outfile:
            outfile.write('\n'.join(all_fitness))

