
import random
from scipy import sparse

def tournament_selection(population_fitness,tournament_size=2):
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
        winner = sorted(contestants,key = lambda k : k[1],reverse=True)[0][0]
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

def generate_offspring(population,fitness,tournament_size=2,crossover_prob=0.9,n_crossovers=1,mutation_rate=0.3):
    new_population = []
    while len(new_population) < population.shape[0]:
        # select
        selections = tournament_selection(fitness) 
        parents = population[selections,:]
        # generate and mutate
        offspring = []
        for generation in range(2):
            child = offspring_crossover(parents,n_crossovers)
            child_mutated = mutate(child,mutation_rate)
            offspring.append(child_mutated)
        # accept
        new_population.extend(offspring)
    return sparse.vstack(new_population)

def random_population(vector_size,population_size=100):
    population = sparse.csr_matrix([[random.choice([0,1]) for i in range(vector_size)] for j in range(population_size)])
    return population

def ga_test():
    firstpop=random_population(1000,100)
    print(firstpop.todense())
    fitness = [random.random() for i in range(100)]
    new_pop = generate_offspring(firstpop,fitness)
    return new_pop

