
import random
from scipy import sparse

def tournament_selection(population_fitness,tournament_size=2,win_condition='highest'):
    """
    returns a list of integers - winners of tournament selection
    """
    winners = []
    candidate_keys = range(len(population_fitness))
    for contest in range(2):
        contestants = []
        for i in range(tournament_size):
            contestant=random.choice(candidate_keys)
            contestants.append((contestant,population_fitness[contestant]))
        sorted_contestants = sorted(contestants,key = lambda k : k[1],reverse=True)
        if win_condition=='highest':
            winner = sorted_contestants[0][0]
        else:
            winner = sorted_contestants[-1][0]
        winners.append(winner)
    return winners

def return_segment(vector,point1,point2):
    return vector[:,range(point1,point2)]

def offspring_crossover(parents,npoints=1):
    dimensions = parents.shape[1]
    crossover_points = []
    while len(set(crossover_points)) < npoints+1:
        crossover_points = sorted([random.choice(range(dimensions)) for point in range(npoints)] + [dimensions])
    parent_switch = 0
    point1 = 0
    segments = []
    for crossover in crossover_points:
        segments.append(return_segment(parents[parent_switch],point1,crossover))
        parent_switch = 1 if parent_switch == 0 else 0
        point1 = crossover
    offspring = sparse.hstack(segments).tocsr()
    return offspring

def mutate(child,mutation_rate):
    mutated_child = []
    for cell in range(child.shape[1]):
        cellval = child[0,cell]
        if random.random() < mutation_rate:
            newval = 1 if cellval == 0 else 0
            mutated_child.append(newval)
        else:
            mutated_child.append(cellval)
    mutated_child_sparse = sparse.csr_matrix(mutated_child)
    return mutated_child_sparse

def generate_offspring(population,fitness,tournament_size=2,crossover_prob=0.9,n_crossovers=1,mutation_rate=0.3,win_condition='highest'):
    new_population = []
    while len(new_population) < population.shape[0]:
        # select
        selections = tournament_selection(fitness,tournament_size,win_condition)
        parents = population[selections,:]
        # generate and mutate
        if random.random() < crossover_prob:
            offspring = []
            for generation in range(2):
                child = offspring_crossover(parents,n_crossovers)
                child_mutated = mutate(child,mutation_rate)
                offspring.append(child_mutated)
        else:
            offspring = parents
        # accept
        new_population.extend(offspring)
    return sparse.vstack(new_population)

def random_vectorpopulation(vector_size,population_size=100):
    vectorpopulation = sparse.csr_matrix([[random.choice([0,1]) for i in range(vector_size)] for j in range(population_size)])
    return vectorpopulation

def random_parameterpopulation(parameter_options,population_size=100):
    parameterpopulation = sparse.csr_matrix([[random.choice(parametervals) for parametervals in parameter_options] for i in range(population_size)])
    return parameterpopulation

def ga_test():
    firstpop=random_population(1000,100)
    print(firstpop.todense())
    fitness = [random.random() for i in range(100)]
    new_pop = generate_offspring(firstpop,fitness)
    return new_pop

