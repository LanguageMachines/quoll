
import os
import numpy
from scipy import sparse
import pickle
import math
import random
from collections import defaultdict

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.modules.vectorize import Vectorize, FeaturizeTask, Combine

from quoll.classification_pipeline.functions.classifier import *
from quoll.classification_pipeline.functions import ga, vectorizer

#################################################################
### Tasks #######################################################
#################################################################

class Train(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()

    scale = BoolParameter()
    min_scale = Parameter()
    max_scale = Parameter()

    ga = BoolParameter()
    ga_parameters = Parameter()

    classifier_parameters = Parameter()
   
    def in_featureselection(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.featureselection.txt')   

    def out_model(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.model.pkl')
    
    def out_model_insights(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.model_insights')

    def run(self):

        # initiate directory with model insights
        self.setup_output_dir(self.out_model_insights().path)
        
        # extract_classifier parameters
        clf_params = self.classifier_parameters.split('--')

        # initiate classifier
        classifierdict = {
                        'random':[RandomClassifier(),[clf_params[6]]],
                        'naive_bayes':[NaiveBayesClassifier(),clf_params[7:9] + [clf_params[2]]],
                        'svm':[SVMClassifier(),clf_params[9:14] + clf_params[2:5]],
                        'svorim':[SvorimClassifier(),clf_params[9:13]],
                        'logistic_regression':[LogisticRegressionClassifier(),clf_params[14:20]],
                        'linreg':[LinearRegressionClassifier(),clf_params[20:23]],
                        'xgboost':[XGBoostClassifier(),clf_params[23:38] + clf_params[2:5]], 
                        'knn':[KNNClassifier(),clf_params[38:]], 
                        }
        clf = classifierdict[clf_params[0]][0]

        # load vectorized instances
        loader = numpy.load(self.in_train().path)
        vectorized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load trainlabels
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')

        # load featureselection
        with open(self.in_featureselection().path,'r',encoding='utf-8') as infile:
            vocab = infile.read().strip().split('\n')
            featureselection = [line.split('\t') for line in vocab]
            featureselection_names = [x[0] for x in featureselection]

        # scale vectors
        if self.scale:
            min_scale = float(self.min_scale)
            max_scale = float(self.max_scale)            
            scaler = vectorizer.fit_scale(vectorized_instances,min_scale,max_scale)
            vectorized_instances = vectorizer.scale_vectors(vectorized_instances,scaler)
            # write scaler
            with open(self.out_model_insights().path + '/scaler.pkl', 'wb') as fid:
                pickle.dump(scaler, fid)
            # write scaled vectors
            numpy.savez(self.out_model_insights().path + '/scaled.vectors.npz', data=vectorized_instances.data, indices=vectorized_instances.indices, indptr=vectorized_instances.indptr, shape=vectorized_instances.shape)

        # select features
        if self.ga:
            ga_params = self.ga_parameters.split('--')
            # load GA class
            ga_instance = ga.GA(vectorized_instances,trainlabels,featureselection_names)
            best_features, best_parameters, best_features_output, best_parameters_output, clf_output, evolution_output, parameter_evolution_output, features_output, weighted_output, ga_report = ga_instance.run(*ga_params,*clf_params)

            # collect output
            selection = random.choice(range(len(best_features)))
            feature_selection_indices = best_features[selection]
            new_parameters = best_parameters[selection]
            vectorized_instances = vectorized_instances[:,feature_selection_indices]
            new_featureselection = [vocab[i] for i in feature_selection_indices]

            # write ga insights
            #ga_insights = ga_instance.return_insights()
            #for gi in ga_insights:
            #    with open(self.out_model_insights().path + '/' + gi[0],'w',encoding='utf-8') as outfile:
            #        outfile.write(gi[1])

            # # save featureselection
            # with open(self.out_model_insights().path + '/ga.featureselection.txt','w',encoding='utf-8') as f_out:
            #     f_out.write('\n'.join(new_featureselection))

            # write vectors
            numpy.savez(self.out_model_insights().path + '/ga.vectors.npz', data=vectorized_instances.data, indices=vectorized_instances.indices, indptr=vectorized_instances.indptr, shape=vectorized_instances.shape)

            # # save reports
            # with open(self.out_model_insights().path + '/ga.report.txt','w',encoding='utf-8') as r_out:
            #     r_out.write(ga_report)

            # with open(self.out_model_insights().path + '/ga.clf_fitness.txt','w',encoding='utf-8') as r_out:
            #     r_out.write(clf_output)

            # with open(self.out_model_insights().path + '/ga.features_output.txt','w',encoding='utf-8') as r_out:
            #     r_out.write(features_output)

            # with open(self.out_model_insights().path + '/ga.weighted_fitness.txt','w',encoding='utf-8') as r_out:
            #     r_out.write(weighted_output)

            # with open(self.out_model_insights().path + '/ga.best_features.txt','w',encoding='utf-8') as r_out:
            #     r_out.write(best_features_output)

            # with open(self.out_model_insights().path + '/ga.best_parameters.txt','w',encoding='utf-8') as r_out:
            #     r_out.write(best_parameters_output)

            # with open(self.out_model_insights().path + '/ga.evolution.txt','w',encoding='utf-8') as r_out:
            #     r_out.write(evolution_output)

            # with open(self.out_model_insights().path + '/ga.parameter_evolution.txt','w',encoding='utf-8') as r_out:
            #     r_out.write(parameter_evolution_output)

        # train classifier
        if clf_params[1] == 'True' or clf_params[5] == 'True':
            clflabels = [float(x) for x in trainlabels]
        else:
            clf.set_label_encoder(sorted(list(set(trainlabels))))
            clflabels = clf.label_encoder.transform(trainlabels)
        clf.train_classifier(vectorized_instances, clflabels, *classifierdict[clf_params[0]][1])

        # save classifier
        model = clf.return_classifier()
        with open(self.out_model().path, 'wb') as fid:
            pickle.dump(model, fid)

        # save model insights
        model_insights = clf.return_model_insights(featureselection_names)
        for mi in model_insights:
            with open(self.out_model_insights().path + '/' + mi[0],'w',encoding='utf-8') as outfile:
                outfile.write(mi[1])

class Predict(Task):

    in_model = InputSlot()
    in_test = InputSlot()
    in_trainlabels = InputSlot()

    classifier = Parameter()
    ordinal = IntParameter()
    linear_raw = IntParameter()
    scale = BoolParameter()
    ga = BoolParameter()
        
    def in_model_insights(self):
        return self.outputfrominput(inputformat='model', stripextension='.model.pkl', addextension='.model_insights')

    def out_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.predictions.txt')

    def out_full_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.full_predictions.txt')

    def run(self):

        # load vectorized instances
        loader = numpy.load(self.in_test().path)
        vectorized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load classifier
        with open(self.in_model().path, 'rb') as fid:
            model = pickle.load(fid)

        # needed to prevent Knn classifier bug
        if self.classifier == 'knn':
            model.n_neighbors = int(model.n_neighbors)

        # inititate classifier
        clf = SvorimClassifier() if self.classifier == 'svorim' else RandomClassifier() if self.classifier == 'random' else AbstractSKLearnClassifier()

        # load labels (for the label encoder)
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')
        
        if self.scale:
            assert os.path.exists(self.in_model_insights().path + '/scaler.pkl'), 'Train scaler file not found, make sure the file exists and/or change scaler path name to ' + self.in_model_insights().path + '/scaler.pkl'
            # read scaler
            with open(self.in_model_insights().path + '/scaler.pkl', 'rb') as fid:
                scaler = pickle.load(fid)
            # scale vectors
            vectorized_instances = vectorizer.scale_vectors(vectorized_instances,scaler)

        if self.ga:
            assert os.path.exists(self.in_model_insights().path + '/ga.best_features.txt'), 'Ga featureselection file not found, make sure the file exists and/or change path name to ' + self.in_model_insights().path + '/ga.best_features.txt'
            # load featureselection
            with open(self.in_model_insights().path + '/ga.best_features.txt','r',encoding='utf-8') as infile:
                lines = infile.read().split('\n')
                featureselection = [line.split('\t') for line in lines]
            featureselection_vocabulary = [x[0] for x in featureselection]
            try:
                featureweights = dict([(i, float(feature[1])) for i, feature in enumerate(featureselection)])
            except:
                featureweights = False

            # align the test instances to the top features to get the right indices
            with open(self.in_featureselection().path,'r',encoding='utf-8') as infile:
                lines = infile.read().strip().split('\n')
                testfeatureselection = [line.split('\t') for line in lines]
            testfeatureselection_vocabulary = [x[0] for x in testfeatureselection]
            vectorized_instances = vectorizer.align_vectors(vectorized_instances, featureselection_vocabulary, testfeatureselection_vocabulary)

        # apply classifier
        if self.ordinal:
            predictions, full_predictions = clf.apply_model(model,vectorized_instances,sorted(list(set([int(x) for x in trainlabels]))))
            predictions = [str(int(x)) for x in predictions]
        elif self.linear_raw:
            predictions, full_predictions = clf.apply_model(model,vectorized_instances,['raw'])
            predictions = [str(x) for x in predictions]
        else:
            clf.set_label_encoder(trainlabels)
            predictions, full_predictions = clf.apply_model(model,vectorized_instances)
            predictions = clf.label_encoder.inverse_transform(predictions)

        # write predictions to file
        with open(self.out_predictions().path,'w',encoding='utf-8') as pr_out:
            pr_out.write('\n'.join(predictions))

        # write full predictions to file
        with open(self.out_full_predictions().path,'w',encoding='utf-8') as fpr_out:
            fpr_out.write('\n'.join(['\t'.join([str(prob) for prob in full_prediction]) for full_prediction in full_predictions]))


class TransformVectors(Task):

    in_train = InputSlot()
    in_test = InputSlot()

    vectors = BoolParameter()

    def in_train_featureselection(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.featureselection.txt')

    def in_test_featureselection(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.featureselection.txt' if self.vectors else '.vocabulary.txt')

    def out_vectors(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.'.join(self.in_train().path.split('/')[-1].split('.')[-6:-2]) + '.transformed.vectors.npz')

    def out_test_featureselection(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.'.join(self.in_train().path.split('/')[-1].split('.')[-6:-2]) + '.transformed.featureselection.txt')        

    def run(self):

        # assert that train and featureselection files exists (not checked in component)
        assert os.path.exists(self.in_train_featureselection().path), 'Train featureselection file not found, make sure the file exists and/or change vocabulary path name to ' + self.in_train_featureselection().path 
        assert os.path.exists(self.in_test_featureselection().path), 'Test featureselection file not found, make sure the file exists and/or change vocabulary path name to ' + self.in_test_featureselection().path 

        # load train
        loader = numpy.load(self.in_train().path)
        traininstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load test
        loader = numpy.load(self.in_test().path)
        testinstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load train featureselection
        with open(self.in_train_featureselection().path,'r',encoding='utf-8') as infile:
            trainlines = infile.read().split('\n')
            train_featureselection = [line.split('\t') for line in trainlines]
        train_featureselection_vocabulary = [x[0] for x in train_featureselection]
        print('Loaded',len(train_featureselection_vocabulary),'selected trainfeatures, now loading test featureselection...')

        # load test featureselection
        with open(self.in_test_featureselection().path,'r',encoding='utf-8') as infile:
            testlines = infile.read().split('\n')
            test_featureselection = [line.split('\t') for line in testlines]
        test_featureselection_vocabulary = [x[0] for x in test_featureselection]
        print('Loaded',len(test_featureselection_vocabulary),'selected testfeatures, now aligning testvectors...')

        # transform vectors
        testvectors = vectorizer.align_vectors(testinstances, train_featureselection_vocabulary, test_featureselection_vocabulary)
        
        # write instances to file
        numpy.savez(self.out_vectors().path, data=testvectors.data, indices=testvectors.indices, indptr=testvectors.indptr, shape=testvectors.shape)

        # write featureselection to file
        print('WRITING',len(trainlines),'TRAINLINES')
        with open(self.out_test_featureselection().path, 'w', encoding = 'utf-8') as t_out:
            t_out.write('\n'.join(trainlines))
 
class TranslatePredictions(Task):

    in_linear_labels = InputSlot()
    in_predictions = InputSlot()

    def in_nominal_labels(self):
        return self.outputfrominput(inputformat='linear_labels', stripextension='.raw.labels', addextension='.labels')

    def out_predictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.translated.predictions.txt')

    def out_full_predictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.translated.full_predictions.txt')

    def out_translator(self):
        return self.outputfrominput(inputformat='linear_labels', stripextension='.labels', addextension='.labeltranslator.txt')

    def run(self):

        # open linear labels
        with open(self.in_linear_labels().path,'r',encoding='utf-8') as linear_labels_in:
            linear_labels = [float(x) for x in linear_labels_in.read().strip().split('\n')]
            
        # open nominal labels
        with open(self.in_nominal_labels().path,'r',encoding='utf-8') as nominal_labels_in:
            nominal_labels = nominal_labels_in.read().strip().split('\n')

        # open predictions
        with open(self.in_predictions().path,'r',encoding='utf-8') as predictions_in:
            predictions = [float(x) for x in predictions_in.read().strip().split('\n')]

        # check if both lists have the same length
        if len(linear_labels) != len(nominal_labels):
            print('Linear labels (',len(linear_labels),') and nominal labels (',len(nominal_labels),') do not have the same length; exiting program...')
            quit()

        # generate dictionary (1 0 14.4\n2 14.4 15.6)
        translated_labels = []
        for label in sorted(list(set(nominal_labels))):
            fitting_linear_labels = [x for i,x in enumerate(linear_labels) if nominal_labels[i] == label]
            translated_labels.append([label,min(fitting_linear_labels),max(fitting_linear_labels)])
        translated_labels_sorted = sorted(translated_labels,key = lambda k : k[1])
        translator = []
        for i,tl in enumerate(translated_labels_sorted):
            if i == 0:
                new_min = translated_labels_sorted[i][1] - 500000
            else:
                new_min = translated_labels_sorted[i-1][2]
            if i == len(translated_labels_sorted)-1:
                new_max = translated_labels_sorted[i][2] * 50
            else:
                new_max = translated_labels_sorted[i][2]
            new_tl = [str(translated_labels_sorted[i][0]),new_min,new_max]
            translator.append(new_tl)

        # translate predictions
        translated_predictions = []
        for prediction in predictions:
            for candidate in translator:
                if prediction > candidate[1] and prediction < candidate[2]:
                    translated_predictions.append(candidate[0])
                    break

        # write translated predictions to files
        with open(self.out_predictions().path,'w',encoding='utf-8') as out:
            out.write('\n'.join(translated_predictions))

        with open(self.out_full_predictions().path,'w',encoding='utf-8') as out:
            classes = sorted([x[0] for x in translator])
            full_predictions = [classes]
            for i,x in enumerate(translated_predictions):
                full_predictions.append(['-'] * len(classes)) # dummy output
            out.write('\n'.join(['\t'.join([prob for prob in full_prediction]) for full_prediction in full_predictions]))

        # write translator to file
        with open(self.out_translator().path,'w',encoding='utf-8') as out:
            out.write('\n'.join([' '.join([str(x) for x in line]) for line in translator]))

class VectorizeTrain(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()

    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()
    select = BoolParameter()
    selector = Parameter()
    select_threshold = Parameter()
    delimiter=Parameter()

    ngrams = Parameter()
    blackfeats = Parameter()
    lowercase = BoolParameter()    
    minimum_token_frequency = IntParameter()
    featuretypes = Parameter()

    tokconfig = Parameter()
    frogconfig = Parameter()
    strip_punctuation = BoolParameter()
    
    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.'.join(self.in_train().path.split('.')[-2:]) if self.in_train().path[-3:] == 'npz' or self.in_train().path[-7:-4] == 'tok' else '.' + self.in_train().path.split('.')[-1], addextension='.vectors.npz')

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.vectors.labels')       

    def run(self):
        
        if self.complete(): # necessary as it will not complete otherwise
            return True
        else:
            yield Vectorize(train=self.in_train().path,trainlabels=self.in_trainlabels().path,weight=self.weight,prune=self.prune,select=self.select,selector=self.selector,select_threshold=self.select_threshold,balance=self.balance,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)

class VectorizeTrainTest(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_test = InputSlot()

    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()
    select = BoolParameter()
    selector = Parameter()
    select_threshold = Parameter()
    delimiter=Parameter()

    ngrams = Parameter()
    blackfeats = Parameter()
    lowercase = BoolParameter()    
    minimum_token_frequency = IntParameter()
    featuretypes = Parameter()

    tokconfig = Parameter()
    frogconfig = Parameter()
    strip_punctuation = BoolParameter()
    
    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.'.join(self.in_train().path.split('.')[-2:]) if self.in_train().path[-3:] == 'npz' or self.in_train().path[-7:-4] == 'tok' else '.' + self.in_train().path.split('.')[-1], addextension='.vectors.npz')
    
    def out_trainlabels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.vectors.labels')       

    def out_test(self):
        return self.outputfrominput(inputformat='test', stripextension='.'.join(self.in_test().path.split('.')[-2:]) if self.in_test().path[-3:] == 'npz' or self.in_test().path[-7:-4] == 'tok' else '.' + self.in_test().path.split('.')[-1], addextension='.vectors.npz')

    def run(self):

        if self.complete(): # necessary as it will not complete otherwise
            return True
        else:
            yield Vectorize(train=self.in_train().path,trainlabels=self.in_trainlabels().path,test=self.in_test().path,weight=self.weight,prune=self.prune,select=self.select,selector=self.selector,select_threshold=self.select_threshold,balance=self.balance,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)

class VectorizeTrainCombinedTask(Task):

    in_trainfeatures = InputSlot()
    in_trainvectors_append = InputSlot()
    in_trainlabels = InputSlot()

    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()

    def out_train_combined(self):
        return self.outputfrominput(inputformat='trainfeatures', stripextension='.features.npz', addextension='.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.' + self.in_trainvectors_append().path.split('.')[-3] + '.vectors.npz' if self.balance and self.in_trainvectors_append().path[-3:] == 'npz' else '.balance.weight_' + self.weight + '.prune_' + str(self.prune) + '.' + self.in_trainvectors_append().path.split('.')[-2] + '.vectors.npz' if self.balance and self.in_trainvectors_append().path[-3:] == 'csv' else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.' + self.in_trainvectors_append().path.split('.')[-3] + '.vectors.npz' if self.in_trainvectors_append().path[-3:] == 'npz' else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.' + self.in_trainvectors_append().path.split('.')[-2] + '.vectors.npz')

    def out_train(self):
        return self.outputfrominput(inputformat='trainfeatures', stripextension='.features.npz', addextension='.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.balance else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz')

    def out_trainlabels(self):
           return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.balanced.labels' if self.balance else '.labels')       
    
    def run(self):
        
        if self.complete(): # necessary check as it will not complete otherwise
            return True
        else:
            yield Vectorize(traininstances=self.in_trainfeatures().path,traininstances_append=self.in_trainvectors_append().path,trainlabels=self.in_trainlabels().path,weight=self.weight,prune=self.prune,balance=self.balance)
    
class VectorizeTestCombinedTask(Task):

    in_trainvectors = InputSlot()
    in_testfeatures = InputSlot()
    in_testvectors_append = InputSlot()
    in_trainlabels = InputSlot()

    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()
    
    def out_vectors(self):
        return self.outputfrominput(inputformat='testfeatures', stripextension='.features.npz', addextension='.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.' + self.in_testvectors_append().path.split('.')[-3] + '.vectors.npz' if self.balance and self.in_testvectors_append().path[-3:] == 'npz' else '.balanced.weight_' + self.weight + '.prune_' + str(self.prune) + '.' + self.in_testvectors_append().path.split('.')[-2] + '.vectors.npz' if self.balance and self.in_testvectors_append().path[-3:] == 'csv' else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.' + self.in_testvectors_append().path.split('.')[-3] + '.vectors.npz' if self.in_testvectors_append().path[-3:] == 'npz' else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.' + self.in_testvectors_append().path.split('.')[-2] + '.vectors.npz')

    def run(self):

        if self.complete(): # necessary as it will not complete otherwise
            return True
        else:
            yield Vectorize(traininstances=self.in_trainvectors().path,trainlabels=self.in_trainlabels().path,testinstances=self.in_testfeatures().path,testinstances_append=self.in_testvectors_append().path,weight=self.weight,prune=self.prune,balance=self.balance)


#################################################################
### Component ###################################################
#################################################################

@registercomponent
class Classify(WorkflowComponent):
    
    train = Parameter()
    trainlabels = Parameter()
    test = Parameter(default = 'xxx.xxx') # not obligatory, dummy extension to enable a pass

    # featureselection parameters
    ga = BoolParameter()
    num_iterations = IntParameter(default=300)
    population_size = IntParameter(default=100)
    elite = Parameter(default='0.1')
    crossover_probability = Parameter(default='0.9')
    mutation_rate = Parameter(default='0.3')
    tournament_size = IntParameter(default=2)
    n_crossovers = IntParameter(default=1)
    stop_condition = IntParameter(default=5)
    weight_feature_size = Parameter(default='0.0')
    instance_steps = IntParameter(default=1)
    sampling = IntParameter(default = 0) # repeated resampling to prevent overfitting
    samplesize = Parameter(default='0.8') # size of trainsample
    
    # classifier parameters
    classifier = Parameter(default='naive_bayes')
    ordinal = IntParameter(default=0)
    jobs = IntParameter(default=1)
    iterations = IntParameter(default=10)
    scoring = Parameter(default='roc_auc') # optimization metric for grid search
    linear_raw = IntParameter(default=0)
    scale = BoolParameter()
    min_scale = Parameter(default='0')
    max_scale = Parameter(default='1')

    random_clf = Parameter(default='equal')
    
    nb_alpha = Parameter(default='1.0')
    nb_fit_prior = BoolParameter()
    
    svm_c = Parameter(default='1.0')
    svm_kernel = Parameter(default='linear')
    svm_gamma = Parameter(default='0.1')
    svm_degree = Parameter(default='1')
    svm_class_weight = Parameter(default='balanced')

    lr_c = Parameter(default='1.0')
    lr_solver = Parameter(default='liblinear')
    lr_dual = IntParameter(default=0)
    lr_penalty = Parameter(default='l2')
    lr_multiclass = Parameter(default='ovr')
    lr_maxiter = Parameter(default='1000')

    linreg_fit_intercept = Parameter(default='1')
    linreg_normalize = Parameter(default='0')
    linreg_copy_X = Parameter(default='1')

    xg_booster = Parameter(default='gbtree') # choices: ['gbtree', 'gblinear']
    xg_silent = Parameter(default='1') # set to '1' to mute printed info on progress
    xg_learning_rate = Parameter(default='0.1') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_min_child_weight = Parameter(default='1') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_max_depth = Parameter(default='6') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_gamma = Parameter(default='0') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_max_delta_step = Parameter(default='0')
    xg_subsample = Parameter(default='1') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_colsample_bytree = Parameter(default='1.0') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_reg_lambda = Parameter(default='1')
    xg_reg_alpha = Parameter(default='0') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_scale_pos_weight = Parameter('1')
    xg_objective = Parameter(default='binary:logistic') # choices: ['binary:logistic', 'multi:softmax', 'multi:softprob']
    xg_seed = Parameter(default='7')
    xg_n_estimators = Parameter(default='100') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 

    knn_n_neighbors = Parameter(default='3')
    knn_weights = Parameter(default='uniform')
    knn_algorithm = Parameter(default='auto')
    knn_leaf_size = Parameter(default='30')
    knn_metric = Parameter(default='euclidean')
    knn_p = IntParameter(default=2)

    # vectorizer parameters
    weight = Parameter(default = 'frequency') # options: frequency, binary, tfidf
    prune = IntParameter(default = 5000) # after ranking the topfeatures in the training set, based on frequency or idf weighting
    delimiter = Parameter(default=',')
    select = BoolParameter()
    selector = Parameter(default=False)
    select_threshold = Parameter(default=False)
    balance = BoolParameter()

    # featurizer parameters
    ngrams = Parameter(default='1 2 3')
    blackfeats = Parameter(default=False)
    lowercase = BoolParameter()    
    minimum_token_frequency = IntParameter(default=1)
    featuretypes = Parameter(default='tokens')

    # ucto / frog parameters
    tokconfig = Parameter(default=False)
    frogconfig = Parameter(default=False)
    strip_punctuation = BoolParameter(default=True)

    def accepts(self):
        return [tuple(x) for x in numpy.array(numpy.meshgrid(*
            [
                (
                InputFormat(self, format_id='modeled_train',extension ='.model.pkl',inputparameter='train'),
                InputFormat(self, format_id='vectorized_train',extension='.vectors.npz',inputparameter='train'),
                InputFormat(self, format_id='featurized_train_csv',extension='.csv',inputparameter='train'),
                InputFormat(self, format_id='featurized_train',extension='.features.npz',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txt',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txtdir',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.json',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.jsondir',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.txt',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.txtdir',inputparameter='train')
                ),
                (
                InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels')
                ),
                (
                InputFormat(self, format_id='vectorized_test',extension='.vectors.npz',inputparameter='test'),
                InputFormat(self, format_id='featurized_test_csv',extension='.csv',inputparameter='test'),
                InputFormat(self, format_id='featurized_test',extension='.features.npz',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.tok.txt',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.tok.txtdir',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.frog.json',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.frog.jsondir',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.txt',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.txtdir',inputparameter='test')
                ),
            ]
            )).T.reshape(-1,3)]

    def setup(self, workflow, input_feeds):
        
        ######################
        ### vectorize ########
        ######################
        
        trainlabels = input_feeds['labels_train']

        traininstances = False
        model = False
        if 'modeled_train' in input_feeds.keys():
            model = input_feeds['modeled_train']
        elif 'vectorized_train' in input_feeds.keys():
            traininstances = input_feeds['vectorized_train']
        elif 'featurized_train_csv' in input_feeds.keys():
            traininstances = input_feeds['featurized_train_csv']
        elif 'featurized_train' in input_feeds.keys():
            traininstances = input_feeds['featurized_train']
        elif 'pre_featurized_train' in input_feeds.keys():
            traininstances = input_feeds['pre_featurized_train']

        if set(['vectorized_test','vectorized_test_csv','featurized_test','pre_featurized_test']) & set(list(input_feeds.keys())):
 
            if 'vectorized_test' in input_feeds.keys():
                testvectors = input_feeds['vectorized_test']
            
            else:
                if 'featurized_test_csv' in input_feeds.keys():
                    testinstances = input_feeds['featurized_test_csv']
                elif 'featurized_test' in input_feeds.keys():
                    testinstances = input_feeds['featurized_test']
                elif 'pre_featurized_test' in input_feeds.keys():
                    testinstances = input_feeds['pre_featurized_test']
                    
                # assert that trainvectors (instead of classifier model) are inputted, in order to vectorize testinstances
                if not traininstances:
                    traininstances = '.'.join(model.split('.')[:2]) + '.vectors.npz'
                
                vectorizer = workflow.new_task('vectorize_traintest',VectorizeTrainTest,autopass=True,weight=self.weight,prune=self.prune,select=self.select,selector=self.selector,select_threshold=self.select_threshold,balance=self.balance,delimiter=self.delimiter,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                vectorizer.in_train = traininstances
                vectorizer.in_trainlabels = trainlabels
                vectorizer.in_test = testinstances

                trainvectors = vectorizer.out_train
                trainlabels = vectorizer.out_trainlabels
                testvectors = vectorizer.out_test                

        else: # only train

            # assert that traininstances rather than a model is inputted; otherwise there is no need for this module
            assert traininstances, 'Only model given as inputfile, no need for this module...' 

            # vectorize traininstances
            vectorizer = workflow.new_task('vectorize_train',VectorizeTrain,autopass=True,
                weight=self.weight,prune=self.prune,select=self.select,selector=self.selector,select_threshold=self.select_threshold,balance=self.balance,delimiter=self.delimiter)
            vectorizer.in_train = traininstances
            vectorizer.in_trainlabels = trainlabels

            trainvectors = vectorizer.out_train
            trainlabels = vectorizer.out_trainlabels

        ######################
        ### Training phase ###
        ######################

        classifier_parameters = '--'.join([str(x) for x in [self.classifier,self.ordinal,self.jobs,self.iterations,self.scoring,self.linear_raw,
            self.random_clf,
            self.nb_alpha,self.nb_fit_prior,
            self.svm_c,self.svm_kernel,self.svm_gamma,self.svm_degree,self.svm_class_weight,
            self.lr_c,self.lr_solver,self.lr_dual,self.lr_penalty,self.lr_multiclass,self.lr_maxiter,
            self.linreg_normalize,self.linreg_fit_intercept,self.linreg_copy_X,
            self.xg_booster,self.xg_silent,self.xg_learning_rate,self.xg_min_child_weight,self.xg_max_depth,self.xg_gamma,self.xg_max_delta_step,self.xg_subsample, 
            self.xg_colsample_bytree,self.xg_reg_lambda,self.xg_reg_alpha,self.xg_scale_pos_weight,self.xg_objective,self.xg_seed,self.xg_n_estimators, 
            self.knn_n_neighbors,self.knn_weights,self.knn_algorithm,self.knn_leaf_size,self.knn_metric,self.knn_p]])

        ga_parameters = '--'.join([str(x) for x in [self.num_iterations, self.population_size, self.elite, self.crossover_probability,self.mutation_rate,self.tournament_size,
            self.n_crossovers,self.stop_condition,self.weight_feature_size,self.instance_steps,self.sampling,self.samplesize]])

        trainer = workflow.new_task('train',Train,autopass=True,scale=self.scale,min_scale=self.min_scale,max_scale=self.max_scale,ga=self.ga,ga_parameters=ga_parameters,classifier_parameters=classifier_parameters)
        trainer.in_train = trainvectors
        trainer.in_trainlabels = trainlabels            

        ######################
        ### Testing phase ####
        ######################

        if len(list(set(['vectorized_test','featurized_test_csv','featurized_test','pre_featurized_test']) & set(list(input_feeds.keys())))) > 0:
                
            if not model:
                model = trainer.out_model

            predictor = workflow.new_task('predictor',Predict,autopass=True,
                classifier=self.classifier,ordinal=self.ordinal,linear_raw=self.linear_raw,scale=self.scale,ga=self.ga)
            predictor.in_test = testvectors
            predictor.in_trainlabels = trainlabels
            predictor.in_model = model

            if self.linear_raw:
                translator = workflow.new_task('predictor',TranslatePredictions,autopass=True)
                translator.in_linear_labels = trainlabels
                translator.in_predictions = predictor.out_predictions

                return translator

            else:

                return predictor

        else:

            return trainer
