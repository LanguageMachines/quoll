
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
        self.folds = False

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
                contestants.append((contestant,population_fitness[contestant][0],population_fitness[contestant][1]))
            sorted_contestants = sorted(contestants,key = lambda k : k[2],reverse=True)
            if win_condition=='highest':
                winner = sorted_contestants[0][1]
            else:
                winner = sorted_contestants[-1][1]
            winners.append(winner)
        return winners

    def return_segment(self,vector,point1,point2):
        return vector[:,range(point1,point2)]

    def draw_sample(self,steps):
        labels_array = numpy.array(self.labels)
        sample_labels = []
        while len(list(set(self.labels)-set(sample_labels))) > 0:
            trainsample_size = random.choice(range(10,int(self.vectors.shape[0]/steps)-10))
            trainsample = random.sample(range(int(self.vectors.shape[0]/steps)),trainsample_size)
            testsample = [i for i in range(int(self.vectors.shape[0]/steps)) if i not in trainsample]
            trainsample_full = sum([[x*2,(x*2)+1] for x in trainsample],[])
            testsample_full = sum([[x*2,(x*2)+1] for x in testsample],[])
            sample_labels = labels_array[testsample_full].tolist()
        return trainsample_full, testsample_full

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

    def generate_offspring(self,vectorpopulation,parameterpopulation,parameter_options,fitness,elite='0.1',tournament_size=2,crossover_prob=0.9,n_crossovers=1,mutation_rate=0.3,win_condition='highest'):
        fitness_numbered = [[i,x] for i,x in enumerate(fitness)]
        fitness_sorted = sorted(fitness_numbered,key = lambda k : k[1],reverse=True) if win_condition == 'highest' else sorted(fitness_numbered,key = lambda k : k[1])
        new_population = [vectorpopulation[x[0],:] for x in fitness_sorted[:int(elite*vectorpopulation.shape[0])]]
        new_parameterpopulation = [parameterpopulation[x[0]] for x in fitness_sorted[:int(elite*vectorpopulation.shape[0])]]
        fitness_candidates = fitness_sorted[int(elite*vectorpopulation.shape[0]):]
        while len(new_population) < vectorpopulation.shape[0]:
            # select
            selections = self.tournament_selection(fitness_candidates,tournament_size,win_condition)
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
        clf.train_classifier(trainvectors_solution, trainlabels, *parameters_solution, v=0)

        # apply classifier
        predictions, full_predictions = clf.apply_model(clf.model,testvectors_solution)

        # assess classifications
        rp = reporter.Reporter(predictions, full_predictions[1:], full_predictions[0], testlabels, ordinal)
        if fitness_metric == 'precision_micro':
            fitness = float(rp.assess_micro_performance()[1])
        if fitness_metric == 'recall_micro':
            fitness = float(rp.assess_micro_performance()[2])
        if fitness_metric == 'f1_micro':
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

    def score_population_fitness(self,population,vectorpopulation,parameterpopulation,trainvectors,trainlabels,testvectors,testlabels,parameters,c,jobs,ordinal,fitness_metric,weight_feature_size, win_condition):
        population_fitness = []
        for i in population:
            vectorsolution = vectorpopulation[i,:].nonzero()[1]
            parametersolution = parameterpopulation[i,:]
            trainvectors_solution = trainvectors[:,vectorsolution]
            testvectors_solution = testvectors[:,vectorsolution]
            parameters_solution = [x.split()[parameterpopulation[i,j]] for j,x in enumerate(parameters)]
            clf = c()
            solution_fitness_clf = self.score_fitness(trainvectors_solution,trainlabels,testvectors_solution,testlabels,clf,parameters_solution,jobs,ordinal,fitness_metric)
            feature_size = len(vectorpopulation[i,:].nonzero()[0])
            feature_size_fraction = feature_size / vectorpopulation.shape[1]
            weighted_fitness = round(((1-weight_feature_size) * solution_fitness_clf) - (weight_feature_size*feature_size_fraction),2) if win_condition == 'highest' else round(solution_fitness_clf + (weight_feature_size*feature_size_fraction),2)
            population_fitness.append([solution_fitness_clf,feature_size,weighted_fitness])
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

    def write_report(self,c,ntrain,ntest,population_fitness_clf,population_fitness_featurecount,population_fitness_weighted,vectorpopulation,parameterpopulation,parameters_split,direction='highest'):
        # summarize fitness files
        report = ['\n','ITERATION ' + str(c),'----------']
        best_fitness_iterations = 0 if direction=='highest' else 1000
        indices_best_fitness_iterations = []
        report.append('# Training instances: ' + str(ntrain))
        report.append('# Test instances: ' + str(ntest))        
        report.append('Avg. fitness classifier: ' + str(numpy.mean(population_fitness_clf)))
        report.append('Median_fitness classifier: ' + str(numpy.median(population_fitness_clf)))
        worst = min(population_fitness_clf) if direction == 'highest' else max(population_fitness_clf)
        report.append('Worst_fitness: ' + str(worst))
        best = max(population_fitness_clf) if direction=='highest' else min(population_fitness_clf)
        report.append('Best_fitness: ' + str(best))
        report.append('Avg. fitness featurecount: ' + str(numpy.mean(population_fitness_featurecount)))
        report.append('Median_fitness_featurecount: ' + str(numpy.median(population_fitness_featurecount)))
        worst = min(population_fitness_featurecount) if direction == 'highest' else max(population_fitness_featurecount)
        report.append('Worst_fitness_featurecount: ' + str(worst))
        best = max(population_fitness_featurecount) if direction=='highest' else min(population_fitness_featurecount)
        report.append('Best_fitness_featurecount: ' + str(best))
        report.append('Avg. fitness weighted: ' + str(numpy.mean(population_fitness_weighted)))
        report.append('Median_fitness_weighted: ' + str(numpy.median(population_fitness_weighted)))
        worst = min(population_fitness_weighted) if direction == 'highest' else max(population_fitness_weighted)
        report.append('Worst_fitness_weighted: ' + str(worst))
        best = max(population_fitness_weighted) if direction=='highest' else min(population_fitness_weighted)
        report.append('Best_fitness_weighted: ' + str(best))
        best_fitness_indices = [j for j, fitness_score in enumerate(population_fitness_weighted) if fitness_score == best]
        report.append('settings best fitness weighted:')
        for i in best_fitness_indices:
            vector = vectorpopulation[i,:]
            indices = vector.toarray()[0].nonzero()[0].tolist()
            features = self.featurenames[indices].tolist()
            parametervector = parameterpopulation[i,:]
            parameters = [parameters_split[j][k] for j,k in enumerate(parametervector.toarray().tolist()[0])]
            report.append(','.join(features) + ' | ' + ','.join(parameters))
        report.append('\n')
        return '\n'.join(report)

    def return_overall_report(self,report,best_features,best_parameters):
        output_clf = '\n'.join(['#train-'+str(row[0]) + '|#test-'+str(row[1]) + ' ' + ','.join([str(x[0]) for x in row[2]]) for row in report])
        output_features = '\n'.join(['#train-'+str(row[0]) + '|#test-'+str(row[1]) + ' ' + ','.join([str(x[1]) for x in row[2]]) for row in report])
        output_weighted = '\n'.join(['#train-'+str(row[0]) + '|#test-'+str(row[1]) + ' ' + ','.join([str(x[2]) for x in row[2]]) for row in report])
        output_overview = '\n'.join([row[3] for row in report])
        best_features_overview = '\n'.join([','.join(self.featurenames[indices].tolist()) for indices in best_features])
        best_parameters_overview = '\n'.join([','.join(row) for row in best_parameters])
        return best_features,best_parameters,best_features_overview, best_parameters_overview, output_clf, output_features, output_weighted, output_overview

    def run(self,num_iterations,population_size,elite,crossover_probability,mutation_rate,tournament_size,n_crossovers,stop_condition,classifier,jobs,ordinal,fitness_metric,weight_feature_size,steps,
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

        # draw sample of train instances
        trainsample, testsample  = self.draw_sample(steps=steps)
        trainvectors = self.vectors[trainsample,:]
        testvectors = self.vectors[testsample,:]
        trainlabels = numpy.array(self.labels)[trainsample].tolist()
        testlabels = numpy.array(self.labels)[testsample].tolist()
        print('TRAINSAMPLE',trainvectors.shape,'TESTSAMPLE',testvectors.shape)
        
        # self.foldreports = []
        # # for each fold
        # for f,fold in enumerate(self.folds):
        #     print('GA FOLD',f+1)
        #     trainvectors = fold['train']
        #     trainlabels = fold['trainlabels']
        #     testvectors = fold['test']
        #     testlabels = fold['testlabels']

        print('Starting with random population')
        # draw random population
        num_dimensions = self.vectors.shape[1]
        offspring = self.random_vectorpopulation(num_dimensions, population_size)

        # draw random parameter population
        parameters = classifierdict[classifier][1]
        parameters_split = [x.split() for x in parameters]
        parameter_options = [[i for i in range(len(x))] for x in parameters_split]
        parameter_offspring = self.random_parameterpopulation(parameter_options, population_size)

        # score population fitness
        win_condition = 'highest' if fitness_metric in ['precision_micro','recall_micro','f1_micro','roc_auc'] else 'lowest'
        population_fitness = self.score_population_fitness(range(population_size),offspring,parameter_offspring,trainvectors,trainlabels,testvectors,testlabels,parameters,classifierdict[classifier][0],jobs,ordinal,fitness_metric,weight_feature_size,win_condition)
        population_weighted_fitness = [x[2] for x in population_fitness]

        # iterate
        report = [[len(trainlabels),len(testlabels),population_fitness,self.write_report('Initialization',len(trainlabels),len(testlabels),[x[0] for x in population_fitness],[x[1] for x in population_fitness],[x[2] for x in population_fitness],offspring,parameter_offspring,parameters_split,win_condition)]]
        highest_streak = 1
        last_best = max([x[2] for x in population_fitness]) if win_condition == 'highest' else min([x[2] for x in population_fitness])
        best_indices = [i for i, fitness_score in enumerate(population_fitness) if fitness_score[2] == last_best]
        best_features = [offspring[i,:].toarray()[0].nonzero()[0].tolist() for i in best_indices]
        best_parameters = [[parameters_split[j][k] for j,k in enumerate(parameter_offspring[i,:].toarray().tolist()[0])] for i in best_indices]
        cursor = 1
        print('BEST FITNESS',last_best,population_fitness[population_weighted_fitness.index(last_best)][0],population_fitness[population_weighted_fitness.index(last_best)][1],'NUM BEST FEATURES:',len(best_features),'NUM BEST PARAMETERS:',len(best_parameters),'HIGHEST STREAK',highest_streak)
        samplechance = [True,False,False,False]
        print('Starting iteration')
        while highest_streak < stop_condition and cursor <= num_iterations:
            print('Iteration',cursor)
            # generate offspring
            offspring, parameter_offspring = self.generate_offspring(offspring,parameter_offspring,parameter_options,population_weighted_fitness,elite=elite,tournament_size=tournament_size,crossover_prob=float(crossover_probability),n_crossovers=n_crossovers,mutation_rate=float(mutation_rate),win_condition=win_condition)
            if random.choice(samplechance):
                trainsample, testsample  = self.draw_sample(steps=steps)
                trainvectors = self.vectors[trainsample,:]
                testvectors = self.vectors[testsample,:]
                trainlabels = numpy.array(self.labels)[trainsample].tolist()
                testlabels = numpy.array(self.labels)[testsample].tolist()
                print('NEW TRAINSAMPLE',trainvectors.shape,'NEW TESTSAMPLE',testvectors.shape)
                    
            # score population fitness
            population_fitness = self.score_population_fitness(range(population_size),offspring,parameter_offspring,trainvectors,trainlabels,testvectors,testlabels,parameters,classifierdict[classifier][0],jobs,ordinal,fitness_metric,weight_feature_size,win_condition)
            # summarize results
            population_weighted_fitness = [x[2] for x in population_fitness]
            best_fitness = max(population_weighted_fitness) if win_condition == 'highest' else min(population_weighted_fitness)
            if (best_fitness > last_best and win_condition == 'highest') or (best_fitness < last_best and win_condition == 'lowest'):
                last_best = best_fitness
                best_features = []
                best_parameters = []
                highest_streak = 1
            else:
                highest_streak += 1
            best_fitness_indices = [i for i, fitness_score in enumerate(population_weighted_fitness) if fitness_score == last_best]
            best_fitness_features = [offspring[i,:].toarray()[0].nonzero()[0].tolist() for i in best_fitness_indices]
            best_fitness_parameters = [[parameters_split[j][k] for j,k in enumerate(parameter_offspring[i,:].toarray().tolist()[0])] for i in best_fitness_indices]
            best_features.extend(best_fitness_features)
            best_parameters.extend(best_fitness_parameters)
            report.append([len(trainlabels),len(testlabels),population_fitness,self.write_report(cursor,len(trainlabels),len(testlabels),[x[0] for x in population_fitness],[x[1] for x in population_fitness],[x[2] for x in population_fitness],offspring,parameter_offspring,parameters_split,win_condition)])
            print('LAST BEST',last_best,'BEST FITNESS',best_fitness,population_fitness[population_weighted_fitness.index(best_fitness)][0],population_fitness[population_weighted_fitness.index(best_fitness)][1],'NUM BEST FEATURES:',len(best_features),'NUM BEST PARAMETERS:',len(best_parameters),'HIGHEST STREAK',highest_streak)
            cursor+=1

        print('Breaking iteration; best fitness:',last_best)
        return self.return_overall_report(report,best_features,best_parameters)
