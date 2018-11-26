
import random
import numpy
from scipy import sparse

from quoll.classification_pipeline.functions.classifier import *
from quoll.classification_pipeline.functions import reporter, nfold_cv_functions

class GA:

    def __init__(self,vectors,labels,featurenames):
        self.vectors = vectors
        self.labels = labels
        self.featurenames = numpy.array(featurenames)

    def tournament_selection(self,population_fitness,tournament_size=2,win_condition='highest'):
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

    def return_segment(self,vector,point1,point2):
        return vector[:,range(point1,point2)]

    def draw_trainingsample(self,num_instances):
        samplesize = random.choice(range(num_instances))
        print('SAMPLESIZE:',samplesize)
        sample = random.sample(num_instances,samplesize)
        print('SAMPLE',sample)
        return sample

    def offspring_crossover(self,parents,npoints=1):
        dimensions = parents.shape[1]
        crossover_points = []
        while len(set(crossover_points)) < npoints+1:
            crossover_points = sorted([random.choice(range(dimensions)) for point in range(npoints)] + [dimensions])
        parent_switch = 0
        point1 = 0
        segments = []
        for crossover in crossover_points:
            segments.append(self.return_segment(parents[parent_switch],point1,crossover))
            parent_switch = 1 if parent_switch == 0 else 0
            point1 = crossover
        offspring = sparse.hstack(segments).tocsr()
        return offspring

    def mutate(self,child,mutation_rate):
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

    def generate_offspring(self,vectorpopulation,parameterpopulation,parameter_options,fitness,tournament_size=2,crossover_prob=0.9,n_crossovers=1,mutation_rate=0.3,win_condition='highest'):
        new_population = []
        new_parameterpopulation = []
        while len(new_population) < vectorpopulation.shape[0]:
            # select
            selections = self.tournament_selection(fitness,tournament_size,win_condition)
            parents = vectorpopulation[selections,:]
            parameterparents = parameterpopulation[selections,:]
            # generate and mutate
            if random.random() < crossover_prob:
                offspring = []
                paramoffspring = []
                for generation in range(2):
                    child = self.offspring_crossover(parents,n_crossovers)
                    child_mutated = self.mutate(child,mutation_rate)
                    offspring.append(child_mutated)
                    paramoffspring.append(self.random_parameterpopulation(parameter_options, 1)[0])
            else:
                offspring = parents
                paramoffspring = parameterparents
            # accept
            new_population.extend(offspring)
            new_parameterpopulation.extend(paramoffspring)
        return sparse.vstack(new_population), sparse.vstack(new_parameterpopulation)

    def random_vectorpopulation(self,vector_size,population_size=100):
        vectorpopulation = sparse.csr_matrix([[random.choice([0,1]) for i in range(vector_size)] for j in range(population_size)])
        return vectorpopulation

    def random_parameterpopulation(self,parameter_options,population_size=100):
        parameterpopulation = sparse.csr_matrix([[random.choice(parametervals) for parametervals in parameter_options] for i in range(population_size)])
        return parameterpopulation

    def score_fitness(self,trainvectors_solution,trainlabels,testvectors_solution,testlabels,clf,parameters_solution,jobs,ordinal,fitness_metric):

        # transform trainlabels
        clf.set_label_encoder(sorted(list(set(trainlabels))))

        # train classifier
        clf.train_classifier(trainvectors_solution, trainlabels, *parameters_solution)

        # apply classifier
        predictions, full_predictions = clf.apply_model(clf.model,testvectors_solution)

        # assess classifications
        rp = reporter.Reporter(predictions, full_predictions[1:], full_predictions[0], testlabels, ordinal)
        if fitness_metric == 'precision':
            fitness = float(rp.assess_micro_performance()[1])
        if fitness_metric == 'recall':
            fitness = float(rp.assess_micro_performance()[2])
        if fitness_metric == 'f1':
            fitness = float(rp.assess_micro_performance()[3])
        elif fitness_metric == 'roc_auc':
            fitness = float(rp.assess_micro_performance()[-4])
        elif fitness_metric == 'mae':
            fitness = rp.assess_overall_ordinal_performance()[7]
        elif fitness_metric == 'rmse':
            fitness = rp.assess_overall_ordinal_performance()[8]
        else:
            print('fitness metric \'',fitness_metric,'\' is not included in the options, exiting program...')
            quit()

        return fitness

    def score_population_fitness(self,population,vectorpopulation,parameterpopulation,trainvectors,trainlabels,testvectors,testlabels,parameters,c,jobs,ordinal,fitness_metric):
        population_fitness = []
        for i in population:
            vectorsolution = vectorpopulation[i,:].nonzero()[1]
            parametersolution = parameterpopulation[i,:]
            trainvectors_solution = trainvectors[:,vectorsolution]
            testvectors_solution = testvectors[:,vectorsolution]
            parameters_solution = [x.split()[parameterpopulation[i,j]] for j,x in enumerate(parameters)]
            clf = c()
            solution_fitness = self.score_fitness(trainvectors_solution,trainlabels,testvectors_solution,testlabels,clf,parameters_solution,jobs,ordinal,fitness_metric)
            population_fitness.append(solution_fitness)
        return population_fitness

    def make_folds(self,n=5,steps=1):
        self.folds = []
        fold_indices = nfold_cv_functions.return_fold_indices(self.vectors.shape[0], num_folds=n, steps=steps)
        for i,fi in enumerate(fold_indices):
            fold = {}
            fold['train'] = sparse.vstack([self.vectors[fi,:] for j,indices in enumerate(fold_indices) if j != i])
            fold['trainlabels'] = numpy.concatenate([self.labels[indices] for j,indices in enumerate(fold_indices) if j != i]).tolist()
            fold['test'] = self.vectors[fold_indices[i]]
            fold['testlabels'] = self.labels[fold_indices[i]].tolist()
            self.folds.append(fold)

    def write_report(self,population_fitness,vectorpopulation,parameterpopulation,parameters_split,direction='highest'):
        # summarize fitness files
        report = []
        best_fitness_iterations = 0 if direction=='highest' else 1000
        indices_best_fitness_iterations = []
        report.append('Avg. fitness: ' + str(numpy.mean(population_fitness)))
        report.append('Median_fitness: ' + str(numpy.median(population_fitness)))
        worst = min(population_fitness) if direction == 'highest' else max(population_fitness)
        report.append('Worst_fitness: ' + str(worst))
        best = max(population_fitness) if direction=='highest' else min(population_fitness)
        report.append('Best_fitness: ' + str(best))
        best_fitness_indices = [j for j, fitness_score in enumerate(population_fitness) if fitness_score == best]
        report.append('settings best fitness:')
        for i in best_fitness_indices:
            vector = vectorpopulation[i,:]
            indices = vector.toarray()[0].nonzero()[0].tolist()
            features = self.featurenames[indices].tolist()
            parametervector = parameterpopulation[i,:]
            parameters = [parameters_split[j][k] for j,k in enumerate(parametervector.toarray().tolist()[0])]
            report.append(','.join(features) + ' | ' + ','.join(parameters))
        report.append('\n')
        return '\n'.join(report)

    def return_overall_report(self):
        scores = [f[1] for f in self.foldreports]
        win_condition = self.foldreports[0][3]
        best = max(scores) if win_condition == 'highest' else min(scores)
        overall_report = ['\n','Best overall score: ' + str(best)]
        best_features_folds = sum([f[2] for f in self.foldreports if f[1] == best],[])
        overall_report.extend([','.join(self.featurenames[indices].tolist()) for indices in best_features_folds])
        overall_report.append('\n')
        for i,f in enumerate(self.foldreports):
            overall_report.append('\n' + 'REPORT FOLD ' + str(i+1) + ':\n\n')
            overall_report.append(f[0])
        overall_report_str = '\n'.join(overall_report)
        return best_features_folds, overall_report_str

    def run(self,sampling,num_iterations,population_size,crossover_probability,mutation_rate,tournament_size,n_crossovers,stop_condition,
        classifier,jobs,ordinal,fitness_metric,
        nb_alpha,nb_fit_prior,
        svm_c,svm_kernel,svm_gamma,svm_degree,svm_class_weight,
        lr_c,lr_solver,lr_dual,lr_penalty,lr_multiclass,lr_maxiter,
        xg_booster,xg_silent,xg_learning_rate,xg_min_child_weight,xg_max_depth,xg_gamma,xg_max_delta_step,xg_subsample,xg_colsample_bytree,xg_reg_lambda,xg_reg_alpha,xg_scale_pos_weight,xg_objective,xg_seed,xg_n_estimators,
        knn_n_neighbors,knn_weights,knn_algorithm,knn_leaf_size,knn_metric,knn_p
        ):

        classifierdict = {
                        'naive_bayes':[NaiveBayesClassifier,[nb_alpha,nb_fit_prior]],
                        'logistic_regression':[LogisticRegressionClassifier,[lr_c,lr_solver,lr_dual,lr_penalty,lr_multiclass,lr_maxiter]],
                        'svm':[SVMClassifier,[svm_c,svm_kernel,svm_gamma,svm_degree,svm_class_weight]], 
                        'xgboost':[XGBoostClassifier,[xg_booster,xg_silent,str(jobs),xg_learning_rate,xg_min_child_weight,xg_max_depth,xg_gamma,
                            xg_max_delta_step,xg_subsample,xg_colsample_bytree,xg_reg_lambda,xg_reg_alpha,xg_scale_pos_weight,xg_objective,xg_seed,xg_n_estimators]],
                        'knn':[KNNClassifier,[knn_n_neighbors,knn_weights,knn_algorithm,knn_leaf_size,knn_metric,knn_p]], 
                        'tree':[TreeClassifier,[]], 
                        'perceptron':[PerceptronLClassifier,[]], 
                        'linear_regression':[LinearRegressionClassifier,[]]
                        }

        self.foldreports = []
        # for each fold
        for f,fold in enumerate(self.folds):
            print('GA FOLD',f+1)
            trainvectors = fold['train']
            trainlabels = fold['trainlabels']
            testvectors = fold['test']
            testlabels = fold['testlabels']

            print('Starting with random population')
            # draw random population
            num_dimensions = self.vectors.shape[1]
            vectorpopulation = self.random_vectorpopulation(num_dimensions, population_size)

            # draw random parameter population
            parameters = classifierdict[classifier][1]
            parameters_split = [x.split() for x in parameters]
            parameter_options = [[i for i in range(len(x))] for x in parameters_split]
            parameterpopulation = self.random_parameterpopulation(parameter_options, population_size)

            # draw sample of train instances
            if sampling:
                print('SHAPE BEFORE',trainvectors.shape)
                vectorsample = self.draw_trainingsample(trainvectors.shape[0])
                trainvectors = trainvectors[vectorsample_indices,:]
                print('SHAPE AFTER',trainvectors.shape)

            # score population fitness
            population_fitness = self.score_population_fitness(range(population_size),vectorpopulation,parameterpopulation,trainvectors,trainlabels,testvectors,testlabels,parameters,classifierdict[classifier][0],jobs,ordinal,fitness_metric)

            # iterate
            win_condition = 'highest' if fitness_metric in ['precision','recall','f1','auc'] else 'lowest'
            report = ['INITIAL POPULATION','-----------------------',self.write_report(population_fitness,vectorpopulation,parameterpopulation,parameters_split,direction=win_condition)]
            highest_streak = 0
            last_best = max(population_fitness) if win_condition == 'highest' else min(population_fitness)
            best_features = []
            cursor = 1
            samplechance = [True,False,False,False]
            print('Starting iteration')
            while highest_streak < stop_condition and cursor <= num_iterations:
                print('Iteration',cursor)
                report.extend(['ITERATION #' + str(cursor),'-----------------------'])
                # generate offspring
                offspring, parameter_offspring = self.generate_offspring(vectorpopulation,parameterpopulation,parameter_options,population_fitness,tournament_size=tournament_size,crossover_prob=float(crossover_probability),n_crossovers=n_crossovers,mutation_rate=float(mutation_rate),win_condition=win_condition)
                if sampling:
                    if random.choice(samplechance):
                        print('TIME TO CHANGE SAMPLE')
                        print('SHAPE BEFORE',trainvectors.shape)
                        vectorsample = self.draw_trainingsample(trainvectors.shape[0])
                        trainvectors = trainvectors[vectorsample_indices,:]
                        print('SHAPE AFTER',trainvectors.shape)
                    else:
                        print('MAINTAINING CURRENT SAMPLE FOR NOW')
                # score population fitness
                population_fitness = self.score_population_fitness(range(population_size),offspring,parameter_offspring,trainvectors,trainlabels,testvectors,testlabels,parameters,classifierdict[classifier][0],jobs,ordinal,fitness_metric)
                # summarize results
                best_fitness = max(population_fitness) if win_condition == 'highest' else min(population_fitness)
                best_fitness_indices = [i for i, fitness_score in enumerate(population_fitness) if fitness_score == best_fitness]
                best_fitness_features = [vectorpopulation[i,:].toarray()[0].nonzero()[0].tolist() for i in best_fitness_indices]
                if (best_fitness > last_best and win_condition == 'highest') or (best_fitness < last_best and win_condition == 'lowest'):
                    last_best = best_fitness
                    best_features = best_fitness_features
                else:
                    highest_streak += 1
                    best_features.extend(best_fitness_features)
                report.append(self.write_report(population_fitness,vectorpopulation,parameterpopulation,parameters_split,win_condition))
                cursor+=1

            print('Breaking iteration; best fitness:',best_fitness)
            self.foldreports.append(['\n'.join(report),best_fitness,best_features,win_condition])

