
import numpy
from scipy import sparse
from collections import defaultdict
import glob

from luiginlp.engine import Task, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, IntParameter, BoolParameter

import quoll.classification_pipeline.functions.hierarchical_classification_functions as hierarchical_classification_functions
import quoll.classification_pipeline.functions.linewriter as linewriter
import quoll.classification_pipeline.functions.docreader as docreader

from quoll.classification_pipeline.modules.run_experiment import ExperimentComponent


class SetupSecondLayerBins(Task):

    in_trainlabels_layer1 = InputSlot()
    in_testpredictions_layer1 = InputSlot()

    def out_layer2_bins(self):
        return self.outputfrominput(inputformat='in_testpredictions_layer1', stripextension='.txt', addextension='.bins.csv')

    def run(self):

        # make second layer experiment directory
        self.setup_output_dir(self.out_exp_layer2().path)

        # read trainlabels layer 1
        with open(in_trainlabels_layer1().path,'r',encoding='utf-8') as file_in:
        	trainlabels_layer1 = file_in.read().strip().split('\n')

        # read testpredictions layer 1
        with open(in_testpredictions_layer1().path,'r',encoding='utf-8') as file_in:
        	testpredictions_layer1 = file_in.read().strip().split('\n')

        # generate data for second layer experiment directory
        second_layer_dict = hierarchical_classification_functions.setup_second_layer(trainlabels_layer1,testpredictions_layer1)

        # setup lines for each train and test bin
        bins = []
        for label in second_layer_dict.keys():
        	trainlabel_indices = second_layer_dict[label]['testindices']
        	testlabel_indices = second_layer_dict[label]['testindices']
        	bins.append([label + '_train'] + trainlabel_indices)
        	bins.append([label + '_test'] + testlabel_indices)

	    # write bins to file
        lw = linewriter.Linewriter(bins)
        lw.write_csv(self.out_layer2_bins().path)

class RunSecondLayer(Task):

    in_bins = InputSlot()
    in_trainfeatures = InputSlot()
    in_trainlabels_first_layer = InputSlot()
    in_trainlabels_second_layer = InputSlot()
    in_testfeatures = InputSlot()
    in_testlabels_second_layer = InputSlot()
    in_trainvocabulary = InputSlot()
    in_testvocabulary = InputSlot()
    in_testdocuments = InputSlot()
    in_classifier_args = InputSlot()

    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()
    classifier = Parameter()
    ordinal = BoolParameter()
    pca = BoolParameter()

    def out_exp_layer2(self):
        return self.outputfrominput(inputformat='in_bins', stripextension='.bins.csv', addextension='.exp_second_layer')

    def run(self):

        # make experiment directory
        self.setup_output_dir(self.out_exp_layer2().path)

        # read first layer labels to setup classification by label bin
        with open(in_trainlabels_first_layer().path,'r',encoding='utf-8') as file_in:
        	trainlabels_first_layer = file_in.read().strip().split('\n')
        	unique_labels = list(set(trainlabels_first_layer))

        # for each fold
        for label in unique_labels:
            yield SecondLayerBin(second_layer_directory=self.out_exp_layer2().path, trainfeatures=self.in_trainfeatures().path, testfeatures=self.in_testfeatures().path, trainlabels_second_layer=self.in_trainlabels_second_layer().path, testlabels_second_layer=self.in_testlabels_second_layer().path, trainvocabulary=self.in_trainvocabulary().path, testvocabulary=self.in_testvocabulary().path, bins=self.in_bins().path, testdocuments=self.in_testdocuments().path, classifier_args=self.in_classifier_args().path, first_layer_label=label, weight=self.weight, prune=self.prune, balance=self.balance, classifier=self.classifier, ordinal=self.ordinal, pca=self.pca)


@registercomponent
class SecondLayerBin(WorkflowComponent):

    second_layer_directory = Parameter()
    trainfeatures = Parameter()
    trainlabels_second_layer = Parameter()
    testfeatures = Parameter()
    testlabels_second_layer = Parameter()
    trainvocabulary = Parameter()
    testvocabulary = Parameter()
    testdocuments = Parameter()
    classifier_args = Parameter()
    bins = Parameter()

    first_layer_label = Parameter()
    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()
    classifier = Parameter()
    ordinal = BoolParameter()
    pca = BoolParameter()
    no_label_encoding = BoolParameter()
    
    def accepts(self):
        return [ ( InputFormat(self,format_id='second_layer_directory',extension='.exp_second_layer',inputparameter='second_layer_directory'), InputFormat(self,format_id='trainfeatures',extension='.features.npz',inputparameter='trainfeatures'), InputFormat(self,format_id='testfeatures',extension='.features.npz',inputparameter='testfeatures'), InputFormat(self, format_id='trainlabels_second_layer', extension='.labels', inputparameter='trainlabels_second_layer'), InputFormat(self, format_id='testlabels_second_layer', extension='.labels', inputparameter='testlabels_second_layer'), InputFormat(self,format_id='testdocuments',extension='.txt',inputparameter='testdocuments'), InputFormat(self, format_id='trainvocabulary', extension='.vocabulary.txt', inputparameter='trainvocabulary'), InputFormat(self, format_id='testvocabulary', extension='.vocabbulary.txt', inputparameter='testvocabulary'), InputFormat(self,format_id='classifier_args',extension='.txt',inputparameter='classifier_args'), InputFormat(self,format_id='bins',extension='.bins.csv',inputparameter='bins') ) ]
 
    def setup(self, workflow, input_feeds):

        layer2_bin = workflow.new_task('run_second_layer_bin', SecondLayerBinTask, autopass=False, first_layer_label=self.first_layer_label, classifier=self.classifier, weight=self.weight, prune=self.prune, balance=self.balance, ordinal=self.ordinal, pca=self.pca, no_label_encoding=self.no_label_encoding)
        layer2_bin.in_second_layer_directory = input_feeds['second_layer_directory']
        layer2_bin.in_trainfeatures = input_feeds['trainfeatures']
        layer2_bin.in_testfeatures = input_feeds['testfeatures']
        layer2_bin.in_trainvocabulary = input_feeds['trainvocabulary']
        layer2_bin.in_testvocabulary = input_feeds['testvocabulary']
        layer2_bin.in_trainlabels_second_layer = input_feeds['trainlabels_second_layer']
        layer2_bin.in_testlabels_second_layer = input_feeds['testlabels_second_layer']
        layer2_bin.in_testdocuments = input_feeds['testdocuments']
        layer2_bin.in_classifier_args = input_feeds['classifier_args']  
        layer2_bin.in_bins = input_feeds['bins']   

        return layer2


class SecondLayerBinTask(Task):

    in_second_layer_directory = InputSlot()
    in_bins = InputSlot()
    in_trainfeatures = InputSlot()
    in_testfeatures = InputSlot()
    in_trainlabels_second_layer = InputSlot()
    in_testlabels_second_layer = InputSlot()
    in_trainvocabulary = InputSlot()
    in_testvocabulary = InputSlot()
    in_testdocuments = InputSlot()
    in_classifier_args = InputSlot()
    
    first_layer_label = Parameter()
    weight = Parameter()
    prune = IntParameter()
    balance = BoolParameter()
    classifier = Parameter()
    ordinal = BoolParameter()
    pca = BoolParameter()
    no_label_encoding = BoolParameter()

    def out_second_layer_bin(self):
        return self.outputfrominput(inputformat='second_layer_directory', stripextension='.exp_second_layer', addextension='.exp_second_layer/' + self.first_layer_label)    

    def out_trainfeatures(self):
        return self.outputfrominput(inputformat='second_layer_directory', stripextension='.exp_second_layer', addextension='.exp_second_layer/' + self.first_layer_label + '/train.features.npz')        

    def out_testfeatures(self):
        return self.outputfrominput(inputformat='second_layer_directory', stripextension='.exp_second_layer', addextension='.exp_second_layer/' + self.first_layer_label + '/test.features.npz')

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='second_layer_directory', stripextension='.exp_second_layer', addextension='.exp_second_layer/' + self.first_layer_label + '/train.labels')

    def out_testlabels(self):
        return self.outputfrominput(inputformat='second_layer_directory', stripextension='.exp_second_layer', addextension='.exp_second_layer/' + self.first_layer_label + '/test.labels')

    def out_testdocuments(self):
        return self.outputfrominput(inputformat='second_layer_directory', stripextension='.exp_second_layer', addextension='.exp_second_layer/' + self.first_layer_label + '/test.docs.txt')

    def out_classifier_args(self):
        return self.outputfrominput(inputformat='second_layer_directory', stripextension='.exp_second_layer', addextension='.exp_second_layer/' + self.first_layer_label + '/classifier_args.txt')

    def run(self):

        # make fold directory
        self.setup_output_dir(self.out_second_layer_bin().path)

        # open bin indices
        dr = docreader.Docreader()
        bins = dr.parse_csv(self.in_bins().path)
        for line in bins:
        	if line[0] == self.first_layer_label + '_train':
        		trainindices = [int(col) for col in line[1:]]
        	elif line[0] == self.first_layer_label + '_test':
        		testindices = [int(col) for col in line[1:]]

        # open trainfeatures
        loader = numpy.load(self.in_train_features().path)
        featurized_traininstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # open testfeatures
        loader = numpy.load(self.in_test_features().path)
        featurized_testinstances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # open trainlabels
        with open(self.in_trainlabels_second_layer().path,'r',encoding='utf-8') as infile:
            trainlabels = numpy.array(infile.read().strip().split('\n'))

        # open testlabels
        with open(self.in_testlabels_second_layer().path,'r',encoding='utf-8') as infile:
            testlabels = numpy.array(infile.read().strip().split('\n'))

        # open testdocuments
        with open(self.in_testdocuments().path,'r',encoding='utf-8') as infile:
            testdocuments = numpy.array(infile.read().strip().split('\n'))

        # open classifier args
        with open(self.in_classifier_args().path) as infile:
            classifier_args = infile.read().rstrip().split('\n')

        # set training and test data
        train_features_second_layer_bin = featurized_traininstances[trainindices]
        train_labels_second_layer_bin = trainlabels[trainindices]
        test_features_second_layer_bin = featurized_testinstances[testindices]
        test_labels_second_layer_bin = testlabels[testindices]
        test_documents_second_layer_bin = testdocuments[testindices]

        # write experiment data to files in fold directory
        numpy.savez(self.out_trainfeatures().path, data=train_features.data, indices=train_features.indices, indptr=train_features.indptr, shape=train_features.shape)
        numpy.savez(self.out_testfeatures().path, data=test_features.data, indices=test_features.indices, indptr=test_features.indptr, shape=test_features.shape)
        with open(self.out_trainlabels().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(train_labels))
        with open(self.out_testlabels().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_labels))
        with open(self.out_testdocuments().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(test_documents))
        with open(self.out_classifier_args().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(classifier_args))

        print('Running experiment for label',self.first_layer_label)

        yield ExperimentComponent(trainfeatures=self.out_trainfeatures().path, trainlabels=self.out_trainlabels().path, testfeatures=self.out_testfeatures().path, testlabels=self.out_testlabels().path, trainvocabulary=self.in_trainvocabulary().path, testvocabulary=self.in_testvocabulary().path, classifier_args=self.out_classifier_args().path, documents=self.out_testdocuments().path, weight=self.weight, prune=self.prune, balance=self.balance, classifier=self.classifier, ordinal=self.ordinal, pca=self.pca, no_label_encoding=self.no_label_encoding)


class SecondLayerClassifications2Predictions(Task):

	in_exp_layer2 = InputSlot()
	in_bins = InputSlot()
	in_testlabels_layer2 = InputSlot()
	
	def out_predictions(self):
	    return self.outputfrominput(inputformat='exp_layer2', stripextension='.exp_second_layer', addextension='.exp_second_layer.predictions.txt')
	    
	def out_full_predictions(self):
	    return self.outputfrominput(inputformat='exp_layer2', stripextension='.exp_second_layer', addextension='.exp_second_layer.full_predictions.txt')
	    
	def run(self):
	
	    # read bins
	    dr = docreader.Docreader()
	    bins = dr.parse_csv(self.in_bins().path)
	    bins_dict = defaultdict(lambda : defaultdict(list))
	    for line in bins:
        	label_cat = line[0]
        	label = line[0].split('_')[0]
        	cat = line[0].split('_')[1]
        	indices = [int(x) for x in line[1:]]
        	bins_dict[label][cat] = indices
        unique_labels_sorted = sorted(bins_dict.keys())
        
        # read testlabels
        with open(in_testlabels_layer2().path,'r',encoding='utf-8') as file_in:
    		testlabels_layer2 = file_in.read().strip().split('\n')
    		
    	# initialize testpredictions and testpredictions_full based on the length of the testlabels
    	testpredictions = ['-'] * len(testlabels_layer2)
    	testpredictions_full = [unique_labels_sorted] + [['0.0'] * len(unique_labels_sorted)] * len(testlabels_layer2)
    	print('testpredictions_full_template:',testpredictions_full)

        # gather predictions - instances pairs
        prediction_files = sorted([ filename for filename in glob.glob(self.in_exp_layer2().path + '/*/*.predictions.txt') ])
        full_prediction_files = sorted([ filename for filename in glob.glob(self.in_exp_layer2().path + '/*/*.full_predictions.txt') ])
        
        # combine predictions with the right order
        for i,prediction_file in enumerate(prediction_files):
        	full_prediction_file = full_prediction_files[i]
        	# extract label
        	predictions_label = prediction_file.split('/')[-2]
        	full_predictions_label = full_prediction_file.split('/')[-2]
        	# assert that labels are the same
        	print('Predictions_label:',predictions_label,'Full_predictions_label:',full_predictions_label)
        	# read in predictions for label
        	with open(prediction_file,'r',encoding='utf-8') as file_in:
        		predictions = file_in.read().strip().split('\n')
        	# read in full predictions for label, and set in right format
        	with open(full_prediction_file,'r',encoding='utf-8') as file_in:
            	lines = [line.split('\t') for line in file_in.read().strip().split('\n')]
        	full_prediction_label_order = lines[0]
        	full_prediction_line_template = ['0.0'] * len(unique_labels_sorted)
        	full_prediction_indices = [unique_labels_sorted.index(label) for label in full_prediction_label_order]
        	full_predictions = []
        	for fp in lines[1:]:
        		full_predictions_line = full_prediction_line_template
        		for j,value in enumerate(fp):
        			full_predictions_line[full_prediction_indices[i]] = str(value)
        		full_predictions.append(full_predictions_line)
        	# fill in predictions and full predictions in appropriate position give the label indices
        	for j,prediction in enumerate(predictions):
        		testpredictions[bins_dict[predictions_label]['test'][j]] = prediction
        	for j,full_prediction in enumerate(full_predictions):
        		testpredictions_full[bins_dict[full_predictions_label]['test'][j]+1] = full_prediction

        # write predictions to file
        with open(self.out_predictions().path,'w',encoding='utf-8') as pr_out:
            pr_out.write('\n'.join(testpredictions))

        # write full predictions to file
        with open(self.out_full_predictions().path,'w',encoding='utf-8') as fpr_out:
            fpr_out.write('\n'.join(['\t'.join([prob for prob in full_prediction]) for full_prediction in testpredictions_full]))
