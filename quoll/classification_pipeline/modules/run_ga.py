
import numpy
from scipy import sparse
import glob

from luiginlp.engine import Task, WorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, IntParameter

from quoll.classification_pipeline.modules import run_ga_iteration
from quoll.classification_pipeline.functions import ga_functions

################################################################################
###Component to thread the tasks together
################################################################################

@registercomponent
class RunGA(WorkflowComponent):

    trainvectors = Parameter()
    trainlabels = Parameter()
    testvectors = Parameter()
    testlabels = Parameter()
    documents = Parameter()
    parameter_options = Parameter()

    num_iterations = IntParameter(default=300)
    population_size = IntParameter(default=100)
    crossover_probability = Parameter(default='0.9')
    mutation_rate = Parameter(default='0.3')
    tournament_size = IntParameter(default=2)
    n_crossovers = IntParameter(default=1)
    classifier = Parameter(default='svm')
    classifier_args = Parameter(default='')
    fitness_metric = Parameter(default='microF1')

    def accepts(self):
        return [ ( InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='testvectors', extension='.vectors.npz',inputparameter='testvectors'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self,format_id='documents',extension='.txt',inputparameter='documents'), InputFormat(self,format_id='parameter_options',extension='.txt',inputparameter='parameter_options') ) ]

    def setup(self, workflow, input_feeds):

        population_generator = workflow.new_task('generate_random_population', GenerateRandomPopulation, autopass=False, population_size=self.population_size)
        population_generator.in_vectors = input_feeds['trainvectors']
        population_generator.in_vectors = input_feeds['parameter_options']

        fitness_manager = workflow.new_task('score_fitness_population', run_ga_iteration.ScoreFitnessPopulation, autopass=False, population_size=self.population_size, classifier=self.classifier, classifier_args=self.classifier_args)
        fitness_manager.in_vectorpopulation = population_generator.out_vectorpopulation
        fitness_manager.in_parameterpopulation = population_generator.out_parameterpopulation
        fitness_manager.in_trainvectors = input_feeds['trainvectors']
        fitness_manager.in_trainlabels = input_feeds['trainlabels']
        fitness_manager.in_testvectors = input_feeds['testvectors']
        fitness_manager.in_testlabels = input_feeds['testlabels']
        fitness_manager.in_parameter_options = input_feeds['parameter_options']
        fitness_manager.in_documents = input_feeds['documents']

        fitness_reporter = workflow.new_task('report_fitness_population', run_ga_iteration.ReportFitnessPopulation, autopass=False, fitness_metric=self.fitness_metric)
        fitness_reporter.in_fitness_exp = fitness_manager.out_fitness_exp

        ga_iterator = workflow.new_task('manage_ga_iterations', ManageGAIterations, autopass=False, num_iterations=self.num_iterations, population_size=self.population_size, crossover_probability=self.crossover_probability, mutation_rate=self.mutation_rate, tournament_size=self.tournament_size, n_crossovers=self.n_crossovers, classifier=self.classifier, classifier_args=self.classifier_args, fitness_metric=self.fitness_metric)
        ga_iterator.in_random_population = population_generator.out_population
        ga_iterator.in_population_fitness = fitness_reporter.out_fitnessreport
        ga_iterator.in_trainvectors = input_feeds['trainvectors']
        ga_iterator.in_trainlabels = input_feeds['trainlabels']
        ga_iterator.in_testvectors = input_feeds['testvectors']
        ga_iterator.in_testlabels = input_feeds['testlabels']
        ga_iterator.in_documents = input_feeds['documents']

        ga_reporter = workflow.new_task('report_ga_iterations', ReportGAIterations, autopass=False)
        ga_reporter.in_iterations_dir = ga_iterator.out_iterations

        return ga_reporter


################################################################################
###Population generator
################################################################################

class GenerateRandomPopulation(Task):

    in_vectors = InputSlot()
    in_parameter_options = InputSlot()

    population_size = IntParameter()

    def out_vectorpopulation(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.ga.vectorpopulation.npz')

    def out_parameterpopulation(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.ga.parameterpopulation.npz')

    def run(self):

        # read in vectors
        loader = numpy.load(self.in_vectors().path)
        instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        num_dimensions = instances.shape[1]

        # generate vectorpopulation
        random_vectorpopulation = ga_functions.random_vectorpopulation(num_dimensions, self.population_size)
        numpy.savez(self.out_vectorpopulation().path, data=random_vectorpopulation.data, indices=random_vectorpopulation.indices, indptr=random_vectorpopulation.indptr, shape=random_vectorpopulation.shape)

        # read in parameter options
        with open(self.in_parameter_options().path) as infile:
            lines = infile.read().strip().split('\n')
            parameter_options = [range(len(line.split())) for line in lines]

        # generate parameterpopulation
        random_parameterpopulation = ga_functions.random_parameterpopulation(parameter_options, self.population_size)
        numpy.savez(self.out_parameterpopulation().path, data=random_parameterpopulation.data, indices=random_parameterpopulation.indices, indptr=random_parameterpopulation.indptr, shape=random_parameterpopulation.shape)



################################################################################
###GA Iterator
################################################################################

class ManageGAIterations(Task):

    in_random_population = InputSlot()
    in_population_fitness = InputSlot()
    in_trainvectors = InputSlot()
    in_trainlabels = InputSlot()
    in_testvectors = InputSlot()
    in_testlabels = InputSlot()
    in_documents = InputSlot()

    num_iterations = IntParameter()
    population_size = IntParameter()
    crossover_probability = Parameter(default='0.9')
    mutation_rate = Parameter(default='0.3')
    tournament_size = IntParameter(default=2)
    n_crossovers = IntParameter(default=1)
    classifier = Parameter(default='svm')
    classifier_args = Parameter(default='')
    fitness_metric = Parameter(default='microF1')

    def out_iterations(self):
        return self.outputfrominput(inputformat='random_population', stripextension='.population.npz', addextension='.iterations')

    def out_pre_iteration(self):
        return self.outputfrominput(inputformat='random_population', stripextension='.population.npz', addextension='.iterations/ga.0.iteration')

    def run(self):

        # create output directory
        self.setup_output_dir(self.out_iterations().path)

        # generate pre iteration directory
        self.setup_output_dir(self.out_pre_iteration().path)
        loader = numpy.load(self.in_random_population().path)
        random_population = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        numpy.savez(self.out_pre_iteration().path + '/population.npz', data=random_population.data, indices=random_population.indices, indptr=random_population.indptr, shape=random_population.shape)
        with open(self.in_population_fitness().path) as infile:
            fitness = infile.read().strip()
            with open(self.out_pre_iteration().path + '/population.fitness.txt','w') as outfile:
                outfile.write(fitness)

        # run iterations
        iterdir = self.out_pre_iteration().path
        for i in range(1,self.num_iterations+1):
            yield(run_ga_iteration.RunGAIteration(dir_latest_iter=iterdir, trainvectors=self.in_trainvectors().path, trainlabels=self.in_trainlabels().path, testvectors=self.in_testvectors().path, testlabels=self.in_testlabels().path, documents=self.in_documents().path, iteration=i, population_size=self.population_size, crossover_probability=self.crossover_probability, mutation_rate=self.mutation_rate, tournament_size=self.tournament_size, n_crossovers=self.n_crossovers, classifier=self.classifier, classifier_args=self.classifier_args, fitness_metric=self.fitness_metric))
            iterdir = iterdir[:-11] + str(i) + '.iteration'


################################################################################
###GA Reporter
################################################################################

class ReportGAIterations(Task):

    in_iterations_dir = InputSlot()

    def out_report(self):
        return self.outputfrominput(inputformat='iterations_dir', stripextension='.iterations', addextension='.report.txt')

    def out_best_solution(self):
        return self.outputfrominput(inputformat='iterations_dir', stripextension='.iterations', addextension='.best_solution.txt')

    def run(self):

        # gather reports by iteration
        print('gathering fitness reports by iteration')
        fitness_files = sorted([ filename for filename in glob.glob(self.in_iterations_dir().path + '/ga.*.iteration/population.fitness.txt') ])

        # summarize fitness files
        report = [['Average fitness','Median fitness','Best fitness','Best fitness index']]
        best_fitness_iterations = 0
        index_best_fitness_iterations = []
        for i,ff in enumerate(fitness_files):
            with open(ff) as infile:
                fitness_scores = [float(score) for score in infile.read().strip().split('\n')]
                avg_fitness = numpy.mean(fitness_scores)
                median_fitness = numpy.median(fitness_scores)
                best_fitness = max(fitness_scores)
                best_fitness_index = fitness_scores.index(best_fitness)
                if best_fitness > best_fitness_iterations:
                    best_fitness_iterations = best_fitness
                    index_best_fitness_iterations = [i,best_fitness_index]
                report.append([str(x) for x in [avg_fitness,median_fitness,best_fitness,best_fitness_index]])

        # write fitness report to file
        with open(self.out_report().path,'w') as outfile:
            outfile.write('\n'.join(['\t'.join(line) for line in report]))

        # extract best solution and write to output
        loader = numpy.load(self.in_iterations_dir().path + '/ga.' + str(index_best_fitness_iterations[0]) + '.iteration/population.npz')
        population_best_iteration = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        best_solution = list(population_best_iteration[index_best_fitness_iterations[1],:].nonzero()[1])
        with open(self.out_best_solution().path,'w') as outfile:
            outfile.write(' '.join([str(i) for i in best_solution]))
