
import os
import numpy
from scipy import sparse
import pickle
import math
import random
from collections import defaultdict

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.modules.validate import Validate
from quoll.classification_pipeline.modules.vectorize import Vectorize, FeaturizeTask, Combine, PredictionsToVectors

from quoll.classification_pipeline.functions.classifier import *
from quoll.classification_pipeline.functions import ga, quoll_helpers, vectorizer

#################################################################
### Tasks #######################################################
#################################################################

class Train(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()

    classifier = Parameter()
    ga_parameters = Parameter()
    classify_parameters = Parameter()
   
    def in_featureselection(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.featureselection.txt')   

    def out_model(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.model.pkl')
    
    def out_model_insights(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.model_insights')

    def run(self):

        kwargs = quoll_helpers.decode_task_input(['ga','classify'],[self.ga_parameters,self.classify_parameters])

        # initiate directory with model insights
        self.setup_output_dir(self.out_model_insights().path)

        # initiate classifier
        classifierdict = {
            'random':[RandomClassifier(),[kwargs['random_clf']]],
            'naive_bayes':[NaiveBayesClassifier(),[kwargs['nb_alpha'],kwargs['nb_fit_prior'],kwargs['jobs']]],
            'svm':[SVMClassifier(),[kwargs['svm_c'],kwargs['svm_kernel'],kwargs['svm_gamma'],kwargs['svm_degree'],kwargs['svm_class_weight'],kwargs['jobs'],kwargs['iterations'],kwargs['scoring']]],
            'svorim':[SvorimClassifier(),[kwargs['svm_c'],kwargs['svm_kernel'],kwargs['svm_gamma'],kwargs['svm_degree']]],
            'logistic_regression':[LogisticRegressionClassifier(),[kwargs['lr_c'],kwargs['lr_solver'],kwargs['lr_dual'],kwargs['lr_penalty'],kwargs['lr_multiclass'],kwargs['lr_maxiter'],kwargs['iterations'],kwargs['jobs']]],
            'linreg':[LinearRegressionClassifier(),[kwargs['linreg_normalize'],kwargs['linreg_fit_intercept'],kwargs['linreg_copy_X'],kwargs['jobs']]],
            'xgboost':[XGBoostClassifier(),[kwargs['xg_booster'],kwargs['xg_silent'],kwargs['xg_learning_rate'],kwargs['xg_min_child_weight'],kwargs['xg_max_depth'],kwargs['xg_gamma'],kwargs['xg_max_delta_step'],kwargs['xg_subsample'],
                kwargs['xg_colsample_bytree'],kwargs['xg_reg_lambda'],kwargs['xg_reg_alpha'],kwargs['xg_scale_pos_weight'],kwargs['xg_objective'],kwargs['xg_seed'],kwargs['xg_n_estimators'],kwargs['jobs'],kwargs['iterations'],kwargs['scoring']]],
            'knn':[KNNClassifier(),[kwargs['knn_n_neighbors'],kwargs['knn_weights'],kwargs['knn_algorithm'],kwargs['knn_leaf_size'],kwargs['knn_metric'],kwargs['knn_p']]],
            'perceptron':[PerceptronClassifier(),[kwargs['perceptron_alpha'],kwargs['iterations'],kwargs['jobs']]],
            'tree':[TreeClassifier(),[kwargs['tree_class_weight']]]
        }
        clf = classifierdict[self.classifier][0]

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
        if kwargs['scale']:
            min_scale = float(kwargs['min_scale'])
            max_scale = float(kwargs['max_scale'])            
            scaler = vectorizer.fit_scale(vectorized_instances,min_scale,max_scale)
            vectorized_instances = vectorizer.scale_vectors(vectorized_instances,scaler)
            # write scaler
            with open(self.out_model_insights().path + '/scaler.pkl', 'wb') as fid:
                pickle.dump(scaler, fid)
            # write scaled vectors
            numpy.savez(self.out_model_insights().path + '/scaled.vectors.npz', data=vectorized_instances.data, indices=vectorized_instances.indices, indptr=vectorized_instances.indptr, shape=vectorized_instances.shape)

        # select features
        if kwargs['ga']:
            # load GA class
            ga_instance = ga.GA(vectorized_instances,trainlabels,featureselection_names)
            best_features, best_parameters = ga_instance.run(
                kwargs['num_iterations'],kwargs['population_size'],kwargs['elite'],kwargs['crossover_probability'],kwargs['mutation_rate'],kwargs['tournament_size'],kwargs['n_crossovers'],kwargs['stop_condition'],kwargs['weight_feature_size'],
                kwargs['steps'],kwargs['sampling'],kwargs['samplesize'],kwargs['classifier'],kwargs['ordinal'],kwargs['jobs'],kwargs['iterations'],kwargs['fitness_metric'],kwargs['linear_raw'],kwargs['random_clf'],kwargs['nb_alpha'],
                kwargs['nb_fit_prior'],kwargs['svm_c'],kwargs['svm_kernel'],kwargs['svm_gamma'],kwargs['svm_degree'],kwargs['svm_class_weight'],kwargs['lr_c'],kwargs['lr_solver'],kwargs['lr_dual'],kwargs['lr_penalty'],kwargs['lr_multiclass'],
                kwargs['lr_maxiter'],kwargs['xg_booster'],kwargs['xg_silent'],kwargs['xg_learning_rate'],kwargs['xg_min_child_weight'],kwargs['xg_max_depth'],kwargs['xg_gamma'],kwargs['xg_max_delta_step'],kwargs['xg_subsample'],
                kwargs['xg_colsample_bytree'],kwargs['xg_reg_lambda'],kwargs['xg_reg_alpha'],kwargs['xg_scale_pos_weight'],kwargs['xg_objective'],kwargs['xg_seed'],kwargs['xg_n_estimators'],kwargs['knn_n_neighbors'],kwargs['knn_weights'],
                kwargs['knn_algorithm'],kwargs['knn_leaf_size'],kwargs['knn_metric'],kwargs['knn_p'],kwargs['linreg_normalize'],kwargs['linreg_fit_intercept'],kwargs['linreg_copy_X']
            )

            # collect output
            selection = random.choice(range(len(best_features)))
            feature_selection_indices = best_features[selection]
            new_parameters = best_parameters[selection]
            vectorized_instances = vectorized_instances[:,feature_selection_indices]
            new_featureselection = [vocab[i] for i in feature_selection_indices]

            # write ga insights
            ga_insights = ga_instance.return_insights()
            for gi in ga_insights:
               with open(self.out_model_insights().path + '/' + gi[0],'w',encoding='utf-8') as outfile:
                   outfile.write(gi[1])

            # save featureselection
            with open(self.out_model_insights().path + '/ga.featureselection.txt','w',encoding='utf-8') as f_out:
                f_out.write('\n'.join(new_featureselection))

            # write vectors
            numpy.savez(self.out_model_insights().path + '/ga.vectors.npz', data=vectorized_instances.data, indices=vectorized_instances.indices, indptr=vectorized_instances.indptr, shape=vectorized_instances.shape)

        # train classifier
        if kwargs['ordinal'] or kwargs['linear_raw']:
            clflabels = [float(x) for x in trainlabels]
        else:
            clf.set_label_encoder(sorted(list(set(trainlabels))))
            clflabels = clf.label_encoder.transform(trainlabels)
        clf.train_classifier(vectorized_instances, clflabels, *classifierdict[self.classifier][1])

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
    ordinal = BoolParameter()
    linear_raw = BoolParameter()
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


class EnsembleTrainTask(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()

    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()
    
    def in_docs(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.txt')

    def in_vocabulary(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.vocabulary.txt')

    def out_ensembledir(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.ensemble')

    def out_vectors(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.ensemble.vectors.npz')

    def out_labels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.ensemble.labels')

    def out_featurenames(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.ensemble.featurenames.txt')

    def run(self):
        
        if self.complete(): # necessary as it will not complete otherwise
            return True

        # make ensemble directory
        self.setup_output_dir(self.out_ensembledir().path)

        # extract ensemble classifiers
        kwargs = quoll_helpers.decode_task_input(['ga','classify'],[self.ga_parameters,self.classify_parameters])
        ensemble_clfs = kwargs['ensemble'].split()
        kwargs['ensemble'] = False
        kwargs['n'] = 5

        vectors = []
        featurenames = []
        # for each ensemble clf
        for ensemble_clf in ensemble_clfs:
            # prepare files
            kwargs['classifier'] = ensemble_clf
            clfdir = self.out_ensembledir().path + '/' + ensemble_clf
            os.mkdir(clfdir)
            instances = clfdir + '/instances.npz'
            labels = clfdir + '/instances.labels'
            docs = clfdir + '/docs.txt'
            vocabulary = clfdir + '/instances.vocabulary.txt'
            os.system('cp ' + self.in_train().path + ' ' + instances)
            os.system('cp ' + self.in_trainlabels().path + ' ' + labels)
            os.system('cp ' + self.in_docs().path + ' ' + docs)
            os.system('cp ' + self.in_vocabulary().path + ' ' + vocabulary)
            yield Validate(instances=instances,labels=labels,docs=docs,**kwargs)
            yield PredictionsToVectors(bins=bins,predictions=clfdir + '/instances.validated.predictions.txt',featurename=ensemble_clf)
            featurenames.append(ensemble_clf)
            loader = numpy.load(clfdir + '/instances.validated.predictions.vectors.npz')
            vectors.append(sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape']))

        # combine and write vectors
        ensemblevectors = sparse.hstack(vectors)
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            ensemblelabels = infile.read().strip().split('\n')
        if kwargs['balance']:
            ensemblevectors, ensemblelabels = vectorizer.balance_data(ensemblevectors, ensemblelabels)
        numpy.savez(self.out_vectors().path, data=ensemblevectors.data, indices=ensemblevectors.indices, indptr=ensemblevectors.indptr, shape=ensemblevectors.shape)
        with open(self.out_labels().path, 'w', encoding='utf-8') as l_out:
            l_out.write('\n'.join(ensemblelabels))

        # combine and write featurenames
        with open(self.out_featurenames().path,'w',encoding='utf-8') as out:
            out.write('\n'.join(featurenames))


class EnsemblePredictTask(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_test = InputSlot()

    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()

    def in_train_vocabulary(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.vocabulary.txt')

    def in_test_vocabulary(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.vocabulary.txt') 

    def out_ensembledir(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.ensemble')

    def out_vectors(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.ensemble.vectors.npz')

    def out_featurenames(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.ensemble.featurenames.txt')

    def run(self):
        
        if self.complete(): # necessary as it will not complete otherwise
            return True

        # make ensemble directory
        self.setup_output_dir(self.out_ensembledir().path)

        # extract ensemble classifiers
        kwargs = quoll_helpers.decode_task_input(['ga','classify','vectorize'],[self.ga_parameters,self.classify_parameters,self.vectorize_parameters])
        ensemble_clfs = kwargs['ensemble'].split()
        kwargs['ensemble'] = False

        vectors = []
        featurenames = []
        # for each ensemble clf
        for ensemble_clf in ensemble_clfs:
            # prepare files
            kwargs['classifier'] = ensemble_clf
            clfdir = self.out_ensembledir().path + '/' + ensemble_clf
            os.mkdir(clfdir)
            train = clfdir + '/train.features.npz'
            trainlabels = clfdir + '/train.labels'
            test = clfdir + '/test.features.npz'
            os.system('cp ' + self.in_train().path + ' ' + train)
            os.system('cp ' + self.in_trainlabels().path + ' ' + labels)
            os.system('cp ' + self.in_test().path + ' ' + test)
            yield Classify(train=self.in_train().path,trainlabels=self.in_trainlabels().path,test=self.in_test().path,**kwargs)
            yield PredictionsToVectors(predictions=clfdir+'/test.predictions.txt')
            featurenames.append(ensemble_clf)
            loader = numpy.load(clfdir + '/test.predictions.vectors.npz')
            vectors.append(sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape']))

        # combine and write vectors
        ensemblevectors = sparse.hstack(vectors)
        numpy.savez(self.out_vectors().path, data=ensemblevectors.data, indices=ensemblevectors.indices, indptr=ensemblevectors.indptr, shape=ensemblevectors.shape)

        # combine and write featurenames
        with open(self.out_featurenames().path,'w',encoding='utf-8') as out:
            out.write('\n'.join(featurenames))                


class EnsembleTrain(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()

    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()

    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.ensemble.vectors.npz')

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.ensemble.labels')

    def run(self):
        
        if self.complete(): # necessary as it will not complete otherwise
            return True

        yield Ensemble(train=self.in_train().path,trainlabels=self.in_trainlabels().path,ga_parameters=self.ga_parameters,classify_parameters=self.classify_parameters,vectorize_parameters=self.vectorize_parameters)


class EnsembleTrainTest(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_test = InputSlot()

    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()

    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.features.npz', addextension='.ensemble.vectors.npz')

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.ensemble.labels')
    
    def out_test(self):
        return self.outputfrominput(inputformat='test', stripextension='.features.npz', addextension='.ensemble.vectors.npz')

    def run(self):

        if self.complete(): # necessary as it will not complete otherwise
            return True
        
        yield Ensemble(train=self.in_train().path,trainlabels=self.in_trainlabels().path,test=self.in_test().path,ga_parameters=self.ga_parameters,classify_parameters=self.classify_parameters,vectorize_parameters=self.vectorize_parameters)


class VectorizeTrain(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()

    vectorize_parameters = Parameter()
    featurize_parameters = Parameter()
    preprocess_parameters = Parameter()
    
    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.'.join(self.in_train().path.split('.')[-2:]) if (self.in_train().path[-3:] == 'npz' or self.in_train().path[-7:-4] == 'tok') else '.' + self.in_train().path.split('.')[-1], addextension='.vectors.npz')

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.vectors.labels')       

    def run(self):
        
        if self.complete(): # necessary as it will not complete otherwise
            return True
        kwargs = quoll_helpers.decode_task_input(['vectorize','featurize','preprocess'],[self.vectorize_parameters,self.featurize_parameters,self.preprocess_parameters])
        yield Vectorize(train=self.in_train().path,trainlabels=self.in_trainlabels().path,**kwargs)

class VectorizeTrainTest(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_test = InputSlot()

    vectorize_parameters = Parameter()
    featurize_parameters = Parameter()
    preprocess_parameters = Parameter()
    
    def out_train(self):
        return self.outputfrominput(inputformat='train', stripextension='.'.join(self.in_train().path.split('.')[-2:]) if (self.in_train().path[-3:] == 'npz' or self.in_train().path[-3:] == 'pkl' or self.in_train().path[-7:-4] == 'tok') else '.' + self.in_train().path.split('.')[-1], addextension='.vectors.npz')
    
    def out_trainlabels(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='.vectors.labels')       

    def out_test(self):
        return self.outputfrominput(inputformat='test', stripextension='.'.join(self.in_test().path.split('.')[-2:]) if (self.in_test().path[-3:] == 'npz' or self.in_test().path[-7:-4] == 'tok') else '.' + self.in_test().path.split('.')[-1], addextension='.vectors.npz')

    def run(self):

        if self.complete(): # necessary as it will not complete otherwise
            return True
        
        kwargs = quoll_helpers.decode_task_input(['vectorize','featurize','preprocess'],[self.vectorize_parameters,self.featurize_parameters,self.preprocess_parameters])
        if '.'.join(self.in_train().path.split('.')[-2:]) == 'model.pkl':
            train = '.'.join(self.in_train().path.split('.')[:-2]) + '.vectors.npz'
        else:
            train = self.in_train().path
        yield Vectorize(train=train,trainlabels=self.in_trainlabels().path,test=self.in_test().path,**kwargs)


##################################################################
### Components ###################################################
##################################################################

@registercomponent
class Ensemble(WorkflowComponent):

    train = Parameter()
    trainlabels = Parameter()
    test = Parameter(default = 'xxx.xxx')

    ga_parameters = Parameter()
    classify_parameters = Parameter()
    vectorize_parameters = Parameter()
    
    def accepts(self):
        return [ ( 
            InputFormat(self,format_id='train',extension='.features.npz',inputparameter='train'), 
            InputFormat(self,format_id='trainlabels',extension='.labels',inputparameter='trainlabels'),
            InputFormat(self,format_id='test',extension='.features.npz',inputparameter='test'), 
        ) ]
 
    def setup(self, workflow, input_feeds):

        kwargs = quoll_helpers.decode_task_input(['ga','classify','vectorize'],[self.ga_parameters,self.classify_parameters,self.vectorize_parameters])

        ensemble_trainer = workflow.new_task('train_ensemble',EnsembleTrainTask,ga_parameters=self.ga_parameters,classify_parameters=self.classify_parameters,vectorize_parameters=self.vectorize_parameters)
        ensemble_trainer.in_train = input_feeds['train']
        ensemble_trainer.in_trainlabels = input_feeds['trainlabels']

        if 'test' in input_feeds.keys():

            ensemble_predictor = workflow.new_task('predict_ensemble',EnsemblePredictTask,ga_parameters=self.ga_parameters,classify_parameters=self.classify_parameters,vectorize_parameters=self.vectorize_parameters)
            ensemble_predictor.in_train = input_feeds['train']
            ensemble_predictor.in_trainlabels = input_feeds['trainlabels']
            ensemble_predictor.in_test = input_feeds['test']

            return ensemble_trainer, ensemble_predictor

        else:

            return ensemble_trainer


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
    steps = IntParameter(default=1)
    sampling = BoolParameter() # repeated resampling to prevent overfitting
    samplesize = Parameter(default='0.8') # size of trainsample
    
    # classifier parameters
    classifier = Parameter(default='naive_bayes')
    ensemble = Parameter(default=False)
    ordinal = BoolParameter()
    jobs = IntParameter(default=1)
    iterations = IntParameter(default=10)
    scoring = Parameter(default='roc_auc') # optimization metric for grid search
    linear_raw = BoolParameter()
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
    lr_dual = BoolParameter()
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

    perceptron_alpha = Parameter(default='1.0')

    tree_class_weight = Parameter(default=False)

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
                InputFormat(self, format_id='vectors_train',extension='.vectors.npz',inputparameter='train'),
                InputFormat(self, format_id='train_csv',extension='.csv',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.features.npz',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.tok.txt',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.tok.txtdir',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.frog.json',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.frog.jsondir',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.txt',inputparameter='train'),
                InputFormat(self, format_id='train',extension='.txtdir',inputparameter='train')
                ),
                (
                InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels')
                ),
                (
                InputFormat(self, format_id='vectors_test',extension='.vectors.npz',inputparameter='test'),
                InputFormat(self, format_id='test_csv',extension='.csv',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.features.npz',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.tok.txt',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.tok.txtdir',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.frog.json',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.frog.jsondir',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.txt',inputparameter='test'),
                InputFormat(self, format_id='test',extension='.txtdir',inputparameter='test')
                ),
            ]
            )).T.reshape(-1,3)]

    def setup(self, workflow, input_feeds):

        task_args = quoll_helpers.prepare_task_input(['preprocess','featurize','vectorize','classify','ga'],workflow.param_kwargs)

        ######################
        ### vectorize ########
        ######################
        
        trainlabels = input_feeds['labels_train']

        traininstances = False
        model = False
        if 'modeled_train' in input_feeds.keys():
            model = input_feeds['modeled_train']
        else:

            if self.ensemble:
                if 'vectors_train' in input_feeds.keys():
                    print('Ensemble classification can not be run with .vectors.npz file, change input...')
                    quit()
                elif 'train_csv' in input_feeds.keys():
                    traincsvtransformer = workflow.new_task('train_transformer_csv',TransformCsv,autopass=True,delimiter=self.delimiter)
                    traincsvtransformer.in_csv = input_feeds['featurized_train_csv']
                    trainfeatures = traincsvtransformer.out_features
                else:
                    trainfeaturizer = workflow.new_task('featurize_train',FeaturizeTask,autopass=False,preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'])
                    trainfeaturizer.in_pre_featurized = input_feeds['train']
                    trainfeatures = trainfeaturizer.out_featurized

            else:
                if 'vectors_train' in input_feeds.keys(): 
                    traininstances = input_feeds['vectors_train']
                elif 'train_csv' in input_feeds.keys():
                    traininstances = input_feeds['train_csv']
                else:
                    traininstances = input_feeds['train']

        if set(['test','test_csv','vectors_test']) & set(list(input_feeds.keys())):
            
            if self.ensemble:
                if 'vectors_test' in input_feeds.keys():
                    print('Ensemble classification can not be run with .vectors.npz file, change input...')
                    quit()
                elif 'test_csv' in input_feeds.keys():
                    testcsvtransformer = workflow.new_task('test_transformer_csv',TransformCsv,autopass=True,delimiter=self.delimiter)
                    testcsvtransformer.in_csv = input_feeds['featurized_test_csv']
                    testfeatures = testcsvtransformer.out_features
                else:
                    testfeaturizer = workflow.new_task('featurize_test',FeaturizeTask,autopass=False,preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'])
                    testfeaturizer.in_pre_featurized = input_feeds['test']
                    testfeatures = testfeaturizer.out_featurized

            else:
                # assert that trainvectors (instead of classifier model) are inputted, in order to vectorize testinstances
                if not traininstances:
                    traininstances = model

                if 'vectors_test' in input_feeds.keys(): 
                    testinstances = input_feeds['vectors_test']
                elif 'test_csv' in input_feeds.keys():
                    testinstances = input_feeds['test_csv']
                else:
                    testinstances = input_feeds['test']
                
                vectorizer = workflow.new_task('vectorize_traintest',VectorizeTrainTest,autopass=True,
                    preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize']
                )
                vectorizer.in_train = traininstances
                vectorizer.in_trainlabels = trainlabels
                vectorizer.in_test = testinstances

                trainvectors = vectorizer.out_train
                trainlabels = vectorizer.out_trainlabels
                testvectors = vectorizer.out_test                

        else: # only train

            # assert that traininstances rather than a model is inputted; otherwise there is no need for this module
            assert traininstances, 'Only model given as inputfile, no need for this module...' 

            if not self.ensemble:
                # vectorize traininstances
                vectorizer = workflow.new_task('vectorize_train',VectorizeTrain,autopass=True,
                    preprocess_parameters=task_args['preprocess'],featurize_parameters=task_args['featurize'],vectorize_parameters=task_args['vectorize']
                )
                vectorizer.in_train = traininstances
                vectorizer.in_trainlabels = trainlabels

                trainvectors = vectorizer.out_train
                trainlabels = vectorizer.out_trainlabels

        ######################
        ### Training phase ###
        ######################

        if self.ensemble:
            if 'vectors_test' in input_feeds.keys():
                ensembler = workflow.new_task('ensemble_traintest',EnsembleTrainTest,autopass=True,
                    classify_parameters=task_args['classify'],ga_parameters=task_args['ga'],vectorize_parameters=task_args['vectorize'])
                ensembler.in_test = testfeatures
                ensembler.in_train = trainfeatures
                ensembler.in_trainlabels = trainlabels
            else: # only train
                ensembler = workflow.new_task('ensemble_train',EnsembleTrain,autopass=True,
                    classify_parameters=task_args['classify'],ga_parameters=task_args['ga'],vectorize_parameters=task_args['vectorize'])
                ensembler.in_train = trainfeatures
                ensembler.in_trainlabels = trainlabels

            trainvectors = ensembler.out_train
            trainlabels = ensembler.out_trainlabels

        trainer = workflow.new_task('train',Train,autopass=True,classifier=self.classifier,classify_parameters=task_args['classify'],ga_parameters=task_args['ga'])
        trainer.in_train = trainvectors
        trainer.in_trainlabels = trainlabels            

        ######################
        ### Testing phase ####
        ######################

        if 'vectors_test' in input_feeds.keys():

            if self.ensemble:
                testvectors = ensembler.out_test

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
