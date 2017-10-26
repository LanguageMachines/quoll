
import os
import numpy
from scipy import sparse
import pickle
import math

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.functions.classifier import *
from quoll.classification_pipeline.functions.lcs_classifier import LCS_classifier
from quoll.classification_pipeline.functions.decisiontree_continuous import DecisionTreeContinuous

class TrainClassifier(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_classifier_args = InputSlot()

    classifier = Parameter()

    def out_model(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension=self.classifier + '.model.pkl')

    def out_label_encoding(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension=self.classifier + '.le')

    def out_model_insights(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension=self.classifier + '.model_insights')

    def run(self):

        # initiate directory with model insights
        self.setup_output_dir(self.out_model_insights().path)

        # initiate classifier
        classifierdict = {'naive_bayes':NaiveBayesClassifier(), 'svm':SVMClassifier(), 'tree':TreeClassifier(), 'continuous_tree':DecisionTreeContinuous(), 'perceptron':PerceptronLClassifier(), 'logistic_regression':LogisticRegressionClassifier(), 'ordinal_ridge':OrdinalRidge(), 'ordinal_at':OrdinalLogisticAT(), 'ordinal_se':OrdinalLogisticSE(), 'ordinal_it':OrdinalLogisticIT()}
        clf = classifierdict[self.classifier]

        # load vectorized instances
        loader = numpy.load(self.in_train().path)
        vectorized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load trainlabels
        with open(self.in_trainlabels().path,'r',encoding='utf-8') as infile:
            trainlabels = infile.read().strip().split('\n')

        # transform trainlabels
        clf.set_label_encoder(trainlabels)

        # load classifier arguments        
        with open(self.in_classifier_args().path,'r',encoding='utf-8') as infile:
            classifier_args_str = infile.read().rstrip()
            classifier_args = classifier_args_str.split('\n')

        # train classifier
        clf.train_classifier(vectorized_instances, trainlabels, *classifier_args)

        # save classifier and insights
        model = clf.return_classifier()
        with open(self.out_model().path, 'wb') as fid:
            pickle.dump(model, fid)
        try:
            model_insights = clf.return_model_insights()
            for mi in model_insights:
                with open(self.out_model_insights().path + '/' + mi[0],'w',encoding='utf-8') as outfile:
                    outfile.write(mi[1])
        except:
            print('No model insights returned.')

        # save label encoding
        label_encoding = clf.return_label_encoding(trainlabels)
        with open(self.out_label_encoding().path,'w',encoding='utf-8') as le_out:
            le_out.write('\n'.join([' '.join(le) for le in label_encoding]))

class ApplyClassifier(Task):

    in_test = InputSlot()
    in_labels = InputSlot()
    in_model = InputSlot()

    classifier = Parameter()

    def out_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension=self.classifier + '.predictions.txt')

    def out_full_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension=self.classifier + '.full_predictions.txt')

    def run(self):

        # load vectorized instances
        loader = numpy.load(self.in_test().path)
        vectorized_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load classifier
        with open(self.in_model().path, 'rb') as fid:
            model = pickle.load(fid)

        # load labels (for the label encoder)
        with open(self.in_labels().path,'r',encoding='utf-8') as infile:
            labels = infile.read().strip().split('\n')

        # inititate classifier
        clf = AbstractSKLearnClassifier()

        # transform labels
        clf.set_label_encoder(labels)

        # apply classifier
        predictions, full_predictions = clf.apply_model(model,vectorized_instances)

        # write predictions to file
        with open(self.out_predictions().path,'w',encoding='utf-8') as pr_out:
            pr_out.write('\n'.join(predictions))

        # write full predictions to file
        with open(self.out_full_predictions().path,'w',encoding='utf-8') as fpr_out:
            fpr_out.write('\n'.join(['\t'.join([str(prob) for prob in full_prediction]) for full_prediction in full_predictions]))

@registercomponent
class Train(WorkflowComponent):

    trainvectors = Parameter()
    trainlabels = Parameter()
    classifier_args = Parameter()

    classifier = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.vectorlabels', inputparameter='trainlabels'), InputFormat(self, format_id='classifier_args', extension='.txt', inputparameter='classifier_args') ) ]

    def setup(self, workflow, input_feeds):

        trainer = workflow.new_task('train_classifier', TrainClassifier, autopass=True, classifier=self.classifier, classifier_args=self.classifier_args)
        trainer.in_train = input_feeds['trainvectors']
        trainer.in_trainlabels = input_feeds['trainlabels']
        trainer.in_classifier_args = input_feeds['classifier_args']

        return trainer

@registercomponent
class TrainApply(WorkflowComponent):

    trainvectors = Parameter()
    trainlabels = Parameter()
    classifier_args = Parameter()
    testvectors = Parameter()

    classifier = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.vectorlabels', inputparameter='trainlabels'), InputFormat(self, format_id='classifier_args', extension='.txt', inputparameter='classifier_args'), InputFormat(self, format_id='testvectors', extension='.vectors.npz',inputparameter='testvectors') ) ]

    def setup(self, workflow, input_feeds):

        trainer = workflow.new_task('train_classifier', TrainClassifier, autopass=True, classifier=self.classifier)
        trainer.in_train = input_feeds['trainvectors']
        trainer.in_trainlabels = input_feeds['trainlabels']
        trainer.in_classifier_args = input_feeds['classifier_args']

        predictor = workflow.new_task('apply_classifier', ApplyClassifier, autopass=True)
        predictor.in_test = input_feeds['testvectors']
        predictor.in_labels = input_feeds['trainlabels']
        predictor.in_model = trainer.out_model

        return predictor

@registercomponent
class Apply(WorkflowComponent):

    testvectors = Parameter()
    model = Parameter()
    trainlabels = Parameter()

    classifier = Parameter()

    def accepts(self):
        return [ ( InputFormat(self, format_id='labels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='testvectors', extension='.vectors.npz',inputparameter='testvectors'), InputFormat(self, format_id='model', extension='.model.pkl',inputparameter='model') ) ]

    def setup(self, workflow, input_feeds):

        predictor = workflow.new_task('apply_classifier', ApplyClassifier, autopass=True, classifier=self.classifier)
        predictor.in_test = input_feeds['testvectors']
        predictor.in_labels = input_feeds['labels']
        predictor.in_model = input_feeds['model']

        return predictor

#################################################
######## Svorim classification (command line tool)
#################################################

@registercomponent
class TrainApplySvorim(WorkflowComponent):

    trainvectors = Parameter()
    trainlabels = Parameter()
    testvectors = Parameter()

    svorim_path = Parameter()
    ga_catch = BoolParameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='testvectors', extension='.vectors.npz',inputparameter='testvectors') ) ]

    def setup(self, workflow, input_feeds):

        classifier = workflow.new_task('svorim_classifier', SvorimClassifier, autopass=True, svorim_path=self.svorim_path, ga_catch=self.ga_catch)
        classifier.in_train = input_feeds['trainvectors']
        classifier.in_labels = input_feeds['trainlabels']
        classifier.in_test = input_feeds['testvectors']

        return classifier

class SvorimClassifier(Task):

    in_train = InputSlot()
    in_labels = InputSlot()
    in_test = InputSlot()

    svorim_path = Parameter()
    ga_catch = BoolParameter()

    def out_classifications(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.classifications.txt')

    def run(self):

        # open train
        loader = numpy.load(self.in_train().path)
        sparse_train_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        array_train_instances = sparse_train_instances.toarray()
        list_train_instances = array_train_instances.tolist()

        # open labels
        with open(self.in_labels().path) as infile:
            labels = infile.read().strip().split('\n')

        # open test
        loader = numpy.load(self.in_test().path)
        sparse_test_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        array_test_instances = sparse_test_instances.toarray()
        list_test_instances = array_test_instances.tolist()

        # set files
        train_instances_labels = []
        for i,instance in enumerate(list_train_instances):
            train_instances_labels.append(instance + [labels[i]])
        expdir = '/'.join(self.in_train().path.split('/')[:-1]) + '/'
        with open(expdir+'svorim_train.0','w',encoding='utf-8') as out:
            out.write('\n'.join([' '.join([str(x) for x in instance]) for instance in train_instances_labels]))
        with open(expdir+'svorim_test.0','w',encoding='utf-8') as out:
            out.write('\n'.join([' '.join([str(x) for x in instance]) + ' ' for instance in list_test_instances]))
        #with open(self.ou,'w',encoding='utf-8') as out:
        #    out.write('\n'.join(['1' for instance in list_test_instances]))

        # perform classification
        os.system(self.svorim_path + ' -i ' + expdir+'svorim_train.0')

        # read in predictions and probabilities
        try:
            predictionfile = expdir + 'svorim_cguess.0'
            probfile = expdir + 'svorim_cguess.0.svm.conf'
            with open(predictionfile) as infile:
                predictions = infile.read().strip().split('\n')
            with open(probfile) as infile:
                probs = infile.read().strip().split('\n')
        except:
            # if self.ga_catch:
            predictions = ['0'] * len(list_test_instances)
            probs = ['0'] * len(list_test_instances)
            # else:
            #     predictions = ['0']
            #     probs = ['0']

        # write classifications to file
        with open(self.out_classifications().path,'w',encoding='utf-8') as cl_out:
            cl_out.write('\n'.join(['\t'.join([prediction,probs[i]]) for i,prediction in enumerate(predictions)])) 


#####################################################################
######## Decision tree classification (continuous segmentation)
#####################################################################

@registercomponent
class TrainApplyDTC(WorkflowComponent):

    trainvectors = Parameter()
    trainlabels = Parameter()
    testvectors = Parameter()
    featurenames = Parameter()

    minimum_per_class = IntParameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='testvectors', extension='.vectors.npz',inputparameter='testvectors'), InputFormat(self, format_id='featurenames',extension='.txt',inputparameter='featurenames') ) ]

    def setup(self, workflow, input_feeds):

        classifier = workflow.new_task('dtc_classifier', DTCClassifier, autopass=True, minimum_per_class=self.minimum_per_class)
        classifier.in_train = input_feeds['trainvectors']
        classifier.in_labels = input_feeds['trainlabels']
        classifier.in_test = input_feeds['testvectors']
        classifier.in_featurenames = input_feeds['featurenames']

        return classifier

class DTCClassifier(Task):

    in_train = InputSlot()
    in_labels = InputSlot()
    in_test = InputSlot()
    in_featurenames = InputSlot()

    minimum_per_class = IntParameter()
    minimum_IG = Parameter()

    def out_classifications(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.dtc.classifications.txt')

    def out_tree(self):
        return self.outputfrominput(inputformat='train', stripextension='.vectors.npz', addextension='.dtc.tree.txt')

    def out_best_features(self):
        return self.outputfrominput(inputformat='train',stripextension='.vectors.npz', addextension='.best_features.txt')

    def run(self):

        # open train
        loader = numpy.load(self.in_train().path)
        sparse_train_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        array_train_instances = sparse_train_instances.toarray()
        # list_train_instances = array_train_instances.tolist()

        # open labels
        with open(self.in_labels().path) as infile:
            labels = infile.read().strip().split('\n')

        # open test
        loader = numpy.load(self.in_test().path)
        sparse_test_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])
        array_test_instances = sparse_test_instances.toarray()
        # list_test_instances = array_test_instances.tolist()

        # open features
        with open(self.in_featurenames().path) as infile:
            featurenames = infile.read().strip().split('\n')

        # train classifier
        dtc = DecisionTreeContinuous()
        dtc.fit(array_train_instances,labels,self.minimum_per_class,float(self.minimum_IG))

        # apply classifier
        predictions = dtc.transform(array_test_instances)
        # print('PREDICTIONS',predictions)
        predictions = [x[1] for x in sorted(predictions.items(),key = lambda k : k[0])]

        # write tree to file
        best_features = []
        with open(self.out_tree().path,'w',encoding='utf-8') as tree_out:
            tree = dtc.Tree
            for item in sorted(tree.items(),key=lambda k : k[0]):
                node = str(item[0])
                feature_index = item[1][0]
                fi = str(feature_index) if item[0] else 'LAST'
                if feature_index:
                    best_features.append(feature_index)
                featurename = featurenames[feature_index] if item[0] else 'LAST'
                segmentation = item[1][1]
                # st = '\t'.join([' - '.join([str(round(x[0],10)),str(round(x[1],10))]) for x in segmentation]) if segmentation else 'LAST'
                st = str(segmentation) if segmentation else 'LAST'
                label = item[1][2]
                tree_out.write('\n'.join([node,featurename,st,label]) + '\n\n')

        # write classifications to file
        with open(self.out_classifications().path,'w',encoding='utf-8') as cl_out:
            cl_out.write('\n'.join(['\t'.join([str(prediction),'-']) for prediction in predictions])) 

        # write best features to file
        best_featurenames = list(numpy.array(featurenames)[best_features])
        with open(self.out_best_features().path,'w') as outfile:
            outfile.write('\n'.join(best_featurenames))

#####################################################################
######## Balanced Winnow classification (using LCS)
#####################################################################

@registercomponent
class TrainApplyBalancedWinnow(WorkflowComponent):

    trainvectors = Parameter()
    trainlabels = Parameter()
    testvectors = Parameter()
    testlabels = Parameter()
    vocabulary = Parameter()

    lcs_path = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='testvectors', extension='.vectors.npz',inputparameter='testvectors'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self, format_id='vocabulary',extension='.txt',inputparameter='vocabulary') ) ]

    def setup(self, workflow, input_feeds):

        bw_classifier = workflow.new_task('bw_classifier', BalancedWinnowClassifier, lcs_path=self.lcs_path, autopass=True)
        bw_classifier.in_train = input_feeds['trainvectors']
        bw_classifier.in_trainlabels = input_feeds['trainlabels']
        bw_classifier.in_test = input_feeds['testvectors']
        bw_classifier.in_testlabels = input_feeds['testlabels']
        bw_classifier.in_vocabulary = input_feeds['vocabulary']

        return bw_classifier

class BalancedWinnowClassifier(Task):

    in_train = InputSlot()
    in_trainlabels = InputSlot()
    in_test = InputSlot()
    in_testlabels = InputSlot()
    in_testdocs = InputSlot()
    in_vocabulary = InputSlot()

    lcs_path = Parameter()

    def out_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.lcs.predictions.txt')

    def out_full_predictions(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.lcs.full_predictions.txt')
    
    def out_test_indices(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.lcs.doc_indices.txt')
    
    def out_docs(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.lcs.docs.txt')

    def out_labels(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.lcs.labels')

    def out_model_insights(self):
        return self.outputfrominput(inputformat='trainlabels', stripextension='.labels', addextension='lcs.model_insights')

    def out_lcsdir(self):
        return self.outputfrominput(inputformat='test', stripextension='.vectors.npz', addextension='.lcsdir')

    def run(self):

        #Set up the output directories, will create it and tear it down on failure automatically
        self.setup_output_dir(self.out_lcsdir().path)
        self.setup_output_dir(self.out_model_insights().path)

        # open train
        loader = numpy.load(self.in_train().path)
        sparse_train_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # open trainlabels
        with open(self.in_trainlabels().path) as infile:
            trainlabels = infile.read().strip().split('\n')

        # open test
        loader = numpy.load(self.in_test().path)
        sparse_test_instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # open testlabels
        with open(self.in_testlabels().path) as infile:
            testlabels = infile.read().strip().split('\n')

        # open testdocs
        with open(self.in_testdocs().path) as infile:
            testdocs = infile.read().strip().split('\n')

        # open vocabulary
        with open(self.in_vocabulary().path,'r',encoding='utf-8') as infile:
            vocabulary = [feature.strip().split('\t')[0] for feature in infile.read().strip().split('\n')]

        # train classifier
        lcsc = LCS_classifier(self.out_lcsdir().path)
        lcsc.experiment(sparse_train_instances,trainlabels,vocabulary,sparse_test_instances,testlabels)
        predictions = lcsc.predictions
        full_predictions = lcsc.full_predictions
        docs = lcsc.docs
        model_insights = lcsc.model
        with open(self.out_lcsdir().path + '/testfile_index.txt','r',encoding='utf-8') as tfi_in:
            testfile_index = {}
            for line in tfi_in.read().strip().split('\n'):
                tf_i = line.strip().split()
                testfile_index[tf_i[0]] = tf_i[1]
        testindices = [testfile_index[d] for d in docs]

        testdocs_lcs = [testdocs[int(ti)] for ti in testindices]
        testlabels_lcs = [testlabels[int(ti)] for ti in testindices]

        # write predictions to file
        with open(self.out_predictions().path,'w',encoding='utf-8') as pr_out:
            pr_out.write('\n'.join(predictions))
    
        # write full predictions to file
        with open(self.out_full_predictions().path,'w',encoding='utf-8') as fpr_out:
            fpr_out.write('\n'.join(['\t'.join(full_prediction) for full_prediction in full_predictions]))

        # write model insights to files
        for category in model_insights.keys():
            outfile = self.out_model_insights().path + '/' + category + '_model.txt'
            with open(outfile,'w',encoding='utf-8') as out:
                mi = model_insights[category]
                out.write('\n'.join(['\t'.join([str(x) for x in line]) for line in mi]))

        # write testindices to file
        with open(self.out_test_indices().path,'w',encoding='utf-8') as ti_out:
            ti_out.write('\n'.join(testindices))

        # write succesfully classified test documents to file
        with open(self.out_docs().path,'w',encoding='utf-8') as td_out:
            td_out.write('\n'.join(testdocs_lcs))

        # write succesfully classified test labels to file
        with open(self.out_labels().path,'w',encoding='utf-8') as tl_out:
            tl_out.write('\n'.join(testlabels_lcs))

