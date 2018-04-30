
import os
import numpy
from scipy import sparse
import pickle
import math
from collections import defaultdict

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.functions.classifier import *
from quoll.classification_pipeline.modules.vectorize import Vectorize, FeaturizeTask

#################################################################
### Tasks #######################################################
#################################################################

class Train(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_classifier_args = InputSlot()

    classifier = Parameter()
    ordinal = BoolParameter()

    def in_vocabulary(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.featureselection.txt')   

    def out_model(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension=self.classifier + '.model.pkl')

    def out_label_encoding(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension=self.classifier + '.le')

    def out_model_insights(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension=self.classifier + '.model_insights')

    def run(self):

        print('Run Trainer')

        # initiate directory with model insights
        self.setup_output_dir(self.out_model_insights().path)

        # initiate classifier
        classifierdict = {
                        'naive_bayes':NaiveBayesClassifier(), 
                        'knn':KNNClassifier(), 
                        'svm':SVMClassifier(), 
                        'tree':TreeClassifier(), 
                        'perceptron':PerceptronLClassifier(), 
                        'logistic_regression':LogisticRegressionClassifier(), 
                        'linear_regression':LinearRegressionClassifier()
                        }
        clf = classifierdict[self.classifier]

        # load vectorized instances
        loader = numpy.load(self.in_train().path)
        vectorized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load trainlabels
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')
        if self.ordinal:
            trainlabels = [float(x) for x in trainlabels]

        # load vocabulary
        with open(self.in_vocabulary().path,'r',encoding='utf-8') as infile:
            vocab = infile.read().strip().split('\n')
            featureselection = [line.split('\t') for line in vocab]
            vocabulary = [x[0] for x in featureselection]

        # transform trainlabels
        clf.set_label_encoder(sorted(list(set(trainlabels))))
            
        # load classifier arguments        
        with open(self.in_classifier_args().path,'r',encoding='utf-8') as infile:
            classifier_args = infile.read().rstrip().split('\n')

        # train classifier
        clf.train_classifier(vectorized_instances, trainlabels, *classifier_args)

        # save classifier
        model = clf.return_classifier()
        with open(self.out_model().path, 'wb') as fid:
            pickle.dump(model, fid)

        # save model insights
        model_insights = clf.return_model_insights(vocabulary)
        for mi in model_insights:
            with open(self.out_model_insights().path + '/' + mi[0],'w',encoding='utf-8') as outfile:
                outfile.write(mi[1])

        # save label encoding
        label_encoding = clf.return_label_encoding(sorted(list(set(trainlabels))))
        with open(self.out_label_encoding().path,'w',encoding='utf-8') as le_out:
            le_out.write('\n'.join([' '.join(le) for le in label_encoding]))

class Predict(Task):

    in_test = InputSlot()
    in_trainlabels = InputSlot()
    in_model = InputSlot()

    classifier = Parameter()
    ordinal = BoolParameter()

    def out_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.' + self.classifier + '.predictions.txt')

    def out_full_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.' + self.classifier + '.full_predictions.txt')

    def run(self):

        # load vectorized instances
        loader = numpy.load(self.in_test().path)
        vectorized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load classifier
        with open(self.in_model().path, 'rb') as fid:
            model = pickle.load(fid)

        # load labels (for the label encoder)
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')
        if self.ordinal:
            trainlabels = [float(x) for x in labels]

        # inititate classifier
        clf = AbstractSKLearnClassifier()

        # transform labels
        clf.set_label_encoder(trainlabels)

        # apply classifier
        predictions, full_predictions = clf.apply_model(model,vectorized_instances)

        # write predictions to file
        with open(self.out_predictions().path,'w',encoding='utf-8') as pr_out:
            pr_out.write('\n'.join(predictions))

        # write full predictions to file
        with open(self.out_full_predictions().path,'w',encoding='utf-8') as fpr_out:
            fpr_out.write('\n'.join(['\t'.join([str(prob) for prob in full_prediction]) for full_prediction in full_predictions]))


class VectorizeTrainTask(Task):

    in_trainfeatures = InputSlot()
    in_trainlabels = InputSlot()

    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()

    def out_train(self):
        return self.outputfrominput(inputformat='trainfeatures', stripextension='.features.npz', addextension='.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz')
        # return self.outputfrominput(inputformat='trainfeatures', stripextension='.features.npz', addextension='balanced.' + '.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.balance else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz')

    #def out_trainlabels(self):
    #    return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='balanced.labels' if self.balance else '.labels')

#    def out_featureweights(self):
#        return self.outputfrominput(inputformat='trainfeatures', stripextension='.features.npz', addextension='.weight_' + self.weight + '.prune_' + str(self.prune) + '.featureweights.txt')
#        return self.outputfrominput(inputformat='trainfeatures', stripextension='.features.npz', addextension='balanced.' + '.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.balance else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.featureweights.txt')

#    def out_featureselection(self):
#        return self.outputfrominput(inputformat='trainfeatures', stripextension='.features.npz', addextension='.weight_' + self.weight + '.prune_' + str(self.prune) + '.featureselection.txt')
#        return self.outputfrominput(inputformat='trainfeatures', stripextension='.features.npz', addextension='balanced.' + '.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz' if self.balance else '.weight_' + self.weight + '.prune_' + str(self.prune) + '.featureselection.txt')
    
    def run(self):

        print(dir(self))
        #print(dir(self.workflow_task))
        #print(self._event_callbacks())
        print('DEPS',self.deps())
        print('ID',self.task_id)
        print('FAMILY',self.task_family)
        print('NAMESPACE',self.task_namespace)
        print('MODULE',self.task_module)
        print('OUTPUT TARGETS',self._output_targets())
        print('INIT',self.initialized())
        print('COMPLETE',self.complete())
        if self.complete():
            return True
        else:
            yield Vectorize(traininstances=self.in_trainfeatures().path,trainlabels=self.in_trainlabels().path,weight=self.weight,prune=self.prune)
            print('AFTER STATUS',self.status)
        #print('DONE',os.listdir('/'.join(self.out_train().path.split('/')[:-1])))
        
# class VectorizeTestTask(Task):

#     in_trainvectors = InputSlot()
#     in_testfeatures = InputSlot()
#     in_trainlabels = InputSlot()

#     weight = Parameter()
#     prune = IntParameter()

#     def out_vectors(self):
#         return self.outputfrominput(inputformat='testfeatures', stripextension='.features.npz', addextension='.weight_' + self.weight + '.prune_' + str(self.prune) + '.vectors.npz')

#     def run(self):
       
#         yield Vectorize(traininstances=self.in_trainvectors().path,trainlabels=self.in_trainlabels().path,testinstances=self.in_testfeatures().path,weight=self.weight,prune=self.prune)


#################################################################
### Component ###################################################
#################################################################

@registercomponent
class Classify(WorkflowComponent):
    
    traininstances = Parameter(default = 'xxx.xxx') # not obligatory, dummy extension to enable a pass
    trainlabels = Parameter()
    testinstances = Parameter(default = 'xxx.xxx') # not obligatory, dummy extension to enable a pass
    classifier_args = Parameter(default = 'xxx.xxx') # not obligatory, dummy extension to enable a pass
    classifier_model = Parameter(default = 'xxx.xxx') # not obligatory, dummy extension to enable a pass

    # classifier parameters
    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter()

    # vectorizer parameters
    weight = Parameter(default = 'frequency') # options: frequency, binary, tfidf
    prune = IntParameter(default = 5000) # after ranking the topfeatures in the training set, based on frequency or idf weighting
    balance = BoolParameter()
    delimiter = Parameter(default=' ')

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
                InputFormat(self, format_id='vectorized_train',extension='.vectors.npz',inputparameter='traininstances'),
                InputFormat(self, format_id='feat_train',extension='.features.npz',inputparameter='traininstances'),
                InputFormat(self, format_id='feat_train_csv',extension='.features.csv',inputparameter='traininstances'),
                InputFormat(self, format_id='feat_train_txt',extension='.features.txt',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_feat_train',extension='.tok.txt',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_feat_train',extension='.tok.txtdir',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_feat_train',extension='.frog.json',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_feat_train',extension='.frog.jsondir',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_feat_train',extension='.txt',inputparameter='traininstances'),
                InputFormat(self, format_id='pre_feat_train',extension='.txtdir',inputparameter='traininstances')
                ),
                (
                InputFormat(self, format_id='lab_train',extension='.labels',inputparameter='trainlabels')
                ),
                (
                InputFormat(self, format_id='classifier_args',extension='.txt',inputparameter='classifier_args')
                ),
                (
                InputFormat(self, format_id='classifier_model',extension='.model.pkl',inputparameter='classifier_model')
                ),
                (
                InputFormat(self, format_id='vectorized_test',extension='.vectors.npz',inputparameter='testinstances'),
                InputFormat(self, format_id='feat_test',extension='.features.npz',inputparameter='testinstances'),
                InputFormat(self, format_id='feat_test_csv',extension='.features.csv',inputparameter='testinstances'),
                InputFormat(self, format_id='feat_test_txt',extension='.features.txt',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_feat_test',extension='.tok.txt',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_feat_test',extension='.tok.txtdir',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_feat_test',extension='.frog.json',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_feat_test',extension='.frog.jsondir',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_feat_test',extension='.txt',inputparameter='testinstances'),
                InputFormat(self, format_id='pre_feat_test',extension='.txtdir',inputparameter='testinstances')
                ),
            ]
            )).T.reshape(-1,5)]

    def setup(self, workflow, input_feeds):

        ######################
        ### Training phase ###
        ######################

        print('CLASSIFY INPUT FEEDS',input_feeds.keys())

        trainlabels = input_feeds['lab_train']

        if 'vectorized_train' in input_feeds.keys():
            trainvectors = input_feeds['vectorized_train']

        else: # pre_vectorized

            # if 'featurized_train_csv' in input_feeds.keys():
            #     trainvectorizer = workflow.new_task('tr_vectorizer_csv',VectorizeCsv,autopass=True,delimiter=self.delimiter)
            #     trainvectorizer.in_csv = input_feeds['featurized_train_csv']
                
            #     trainvectors = trainvectorizer.out_vectors

            # elif 'featurized_train_txt' in input_feeds.keys():
            #     trainvectorizer = workflow.new_task('tr_vectorizer_txt',VectorizeTxt,autopass=True,delimiter=self.delimiter)
            #     trainvectorizer.in_txt = input_feeds['featurized_train_txt']

            #     trainvectors = trainvectorizer.out_vectors

            # else:

            if 'pre_feat_train' in input_feeds.keys():
                trainfeaturizer = workflow.new_task('featurize_tr',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                trainfeaturizer.in_pre_featurized = input_feeds['pre_feat_train']

                traininstances = trainfeaturizer.out_featurized

            else:
                traininstances = input_feeds['feat_train']
                
            trainvectorizer = workflow.new_task('vectorize_tr',VectorizeTrainTask,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance)
            trainvectorizer.in_trainfeatures = traininstances
            trainvectorizer.in_trainlabels = trainlabels
            
            trainvectors = trainvectorizer.out_train
                
        if 'classifier_args' in input_feeds.keys():
            trainer = workflow.new_task('train',Train,autopass=True,classifier=self.classifier,ordinal=self.ordinal)
            trainer.in_train = trainvectors
            trainer.in_trainlabels = trainlabels
            trainer.in_classifier_args = input_feeds['classifier_args']
            

        ######################
        ### Testing phase ####
        ######################

        if len(list(set(['vectorized_test','feat_test_csv','feat_test_txt','feat_test','pre_feat_test']) & set(list(input_feeds.keys())))) > 0:

            if 'vectorized_test' in input_feeds.keys():
                testvectors = input_feeds['vectorized_test']

            else: # pre_vectorized
       
                # if 'featurized_test_csv' in input_feeds.keys():
                #     testvectorizer = workflow.new_task('vectorizer_csv_te',VectorizeCsv,autopass=True,delimiter=self.delimiter)
                #     testvectorizer.in_csv = input_feeds['featurized_test_csv']
            
                # elif 'featurized_test_txt' in input_feeds.keys():
                #     testvectorizer = workflow.new_task('vectorizer_txt_te',VectorizeTxt,autopass=True,delimiter=self.delimiter)
                #     testvectorizer.in_txt = input_feeds['featurized_test_txt']

                # else:
                
                    # if 'pre_featurized_test' in input_feeds.keys():
                    #     testfeaturizer = workflow.new_task('featurize_te',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                    #     testfeaturizer.in_pre_featurized = input_feeds['pre_featurized_test']

                    #     testinstances = testfeaturizer.out_featurized

                    # else:
                    #     testinstances = input_feeds['featurized_test']

                testvectorizer = workflow.new_task('vectorize_te',VectorizeTestTask,autopass=True,weight=self.weight,prune=self.prune)
                testvectorizer.in_trainvectors = trainvectors
                testvectorizer.in_trainlabels = trainlabels
                testvectorizer.in_testfeatures = testinstances

                testvectors = testvectorizer.out_vectors

            if 'classifier_model' in input_feeds.keys():
                model = input_feeds['classifier_model']
            else:
                model = trainer.out_model

            predictor = workflow.new_task('predictor',Predict,autopass=True,classifier=self.classifier,ordinal=self.ordinal)
            predictor.in_test = testvectors
            predictor.in_trainlabels = trainlabels
            predictor.in_model = model

            return predictor

        else:

            print('RETURN TRAINER')
            return trainer
