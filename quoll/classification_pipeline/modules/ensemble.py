
import os
import numpy
from scipy import sparse
import pickle
import math
import random
from collections import defaultdict

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.modules.validate import Validate
from quoll.classification_pipeline.modules.vectorize import PredictionsToVectors

from quoll.classification_pipeline.functions import quoll_helpers, vectorizer

#################################################################
### Tasks #######################################################
#################################################################

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

