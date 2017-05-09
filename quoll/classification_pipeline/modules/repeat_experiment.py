
from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, IntParameter, BoolParameter
import numpy
from scipy import sparse
from collections import defaultdict
import glob
import os

import quoll.classification_pipeline.functions.nfold_cv_functions as nfold_cv_functions
import quoll.classification_pipeline.functions.linewriter as linewriter
import quoll.classification_pipeline.functions.docreader as docreader
import quoll.classification_pipeline.functions.reporter as reporter

from quoll.classification_pipeline.modules.select_features import SelectFeatures
from quoll.classification_pipeline.modules.run_experiment import ExperimentComponentVector, ExperimentComponentSvorimVector, ExperimentComponentDTCVector 
from quoll.classification_pipeline.modules.make_bins import MakeBins 

################################################################################
###Component to thread the tasks together
################################################################################

@registercomponent
class RepeatExperiment(WorkflowComponent):
    
    trainvectors = Parameter()
    trainlabels = Parameter()
    testvectors = Parameter()
    testlabels = Parameter()
    traindocuments = Parameter()
    testdocuments = Parameter()
    classifier_args = Parameter()

    repeats = IntParameter(default=1000)
    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter(default=False)

    def accepts(self):
        return [ ( InputFormat(self,format_id='train',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='test', extension='.vectors.npz',inputparameter='testvectors'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self,format_id='traindocuments',extension='.txt',inputparameter='traindocuments'), InputFormat(self,format_id='testdocuments',extension='.txt',inputparameter='testdocuments'), InputFormat(self,format_id='classifier_args',extension='.txt',inputparameter='classifier_args') ) ]
    
    def setup(self, workflow, input_feeds):

        repeater = workflow.new_task('repeat_experiment', RepeatExperimentTask, autopass=True, repeats=self.repeats, classifier=self.classifier, ordinal=self.ordinal)
        repeater.in_trainvectors = input_feeds['train']
        repeater.in_trainlabels = input_feeds['trainlabels']
        repeater.in_testvectors = input_feeds['train']
        repeater.in_testlabels = input_feeds['trainlabels']        
        repeater.in_traindocuments = input_feeds['traindocuments']      
        repeater.in_testdocuments = input_feeds['testdocuments'] 
        repeater.in_classifier_args = input_feeds['classifier_args']

        reporter = workflow.new_task('report_repeats', ReportRepeats, ordinal=self.ordinal, autopass = False)
        reporter.in_repeatdirectory = repeater.out_repeat_directory

        return reporter


class RepeatExperimentTask(Task):

    in_trainvectors = InputSlot()
    in_trainlabels = InputSlot()
    in_testvectors = InputSlot()
    in_testlabels = InputSlot()
    in_traindocuments = InputSlot()
    in_testdocuments = InputSlot()
    in_classifier_args = InputSlot()

    repeats = IntParameter()
    classifier = Parameter()
    ordinal = BoolParameter()

    def out_repeat_directory(self):
        return self.outputfrominput(inputformat='trainvectors', stripextension='.vectors.npz', addextension='.repeats')
        
    def run(self):

        # make experiment directory
        self.setup_output_dir(self.out_repeat_directory().path)

        # for each fold
        for repeat in range(self.repeats):
            yield FoldVectors(directory=self.out_exp().path, trainvectors=self.in_vectors().path, testlabels=self.in_labels().path, testvectors=self.in_vectors().path, trainlabels=self.in_labels().path, traindocuments=self.in_traindocuments().path, testdocuments=self.in_testdocuments().path, classifier_args=self.in_classifier_args().path, i=repeat, classifier=self.classifier, ordinal=self.ordinal)



@registercomponent
class RepeatInstance(WorkflowComponent):

    directory = Parameter()
    trainvectors = Parameter()
    trainlabels = Parameter()
    testvectors = Parameter()
    testlabels = Parameter()
    traindocuments = Parameter()
    testdocuments = Parameter()
    classifier_args = Parameter()

    i = IntParameter()
    classifier = Parameter()
    ordinal = BoolParameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='directory',extension='.exp',inputparameter='directory'), InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self,format_id='testvectors',extension='.vectors.npz',inputparameter='testvectors'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self,format_id='traindocuments',extension='.txt',inputparameter='traindocuments'), InputFormat(self,format_id='testdocuments',extension='.txt',inputparameter='testdocuments'), InputFormat(self,format_id='classifier_args',extension='.txt',inputparameter='classifier_args'), InputFormat(self,format_id='bins',extension='.bins.csv',inputparameter='bins'), InputFormat(self,format_id='featurenames',extension='.txt',inputparameter='featurenames') ) ]
    
    def setup(self, workflow, input_feeds):

        repeat = workflow.new_task('repeat', RepeatInstanceTask, autopass=False, i=self.i, classifier=self.classifier, ordinal=self.ordinal)
        repeat.in_directory = input_feeds['directory']
        repeat.in_trainvectors = input_feeds['trainvectors']
        repeat.in_trainlabels = input_feeds['trainlabels']
        repeat.in_testvectors = input_feeds['testvectors']
        repeat.in_testlabels = input_feeds['testlabels']
        repeat.in_traindocuments = input_feeds['traindocuments']
        repeat.in_testdocuments = input_feeds['testdocuments']
        repeat.in_classifier_args = input_feeds['classifier_args']  

        return repeat

class RepatInstanceTask(Task):

    in_directory = InputSlot()
    in_trainvectors = InputSlot()
    in_trainlabels = InputSlot()
    in_testvectors = InputSlot()
    in_testlabels = InputSlot()
    in_traindocuments = InputSlot()
    in_testdocuments = InputSlot()
    in_classifier_args = InputSlot()
    
    i = IntParameter()
    classifier = Parameter()
    ordinal = BoolParameter()

    def out_repeatdir(self):
        return self.outputfrominput(inputformat='directory', stripextension='.repeats', addextension='.repeats/repeat' + str(self.i))    

    def out_trainvectors(self):
        return self.outputfrominput(inputformat='directory', stripextension='.repeats', addextension='.repeats/repeat' + str(self.i) + '/train.vectors.npz')

    def out_testvectors(self):
        return self.outputfrominput(inputformat='directory', stripextension='.repeats', addextension='.repeats/repeat' + str(self.i) + '/test.vectors.npz')

    def out_trainlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.repeats', addextension='.repeats/repeat' + str(self.i) + '/train.labels')

    def out_testlabels(self):
        return self.outputfrominput(inputformat='directory', stripextension='.repeats', addextension='.repeats/repeat' + str(self.i) + '/test.labels')

    def out_traindocuments(self):
        return self.outputfrominput(inputformat='directory', stripextension='.repeats', addextension='.repeats/repeat' + str(self.i) + '/train.docs.txt')

    def out_testdocuments(self):
        return self.outputfrominput(inputformat='directory', stripextension='.repeats', addextension='.repeats/repeat' + str(self.i) + '/test.docs.txt')

    def out_classifier_args(self):
        return self.outputfrominput(inputformat='directory', stripextension='.repeats', addextension='.repeats/repeat' + str(self.i) + '/classifier_args.txt')

    def run(self):

        # make fold directory
        self.setup_output_dir(self.out_repeatdir().path)

        # copy files
        os.system('cp ' + self.in_trainvectors().path + ' ' + self.out_trainvectors().path)
        os.system('cp ' + self.in_trainlabels().path + ' ' + self.out_trainlabels().path)
        os.system('cp ' + self.in_testvectors().path + ' ' + self.out_testvectors().path)
        os.system('cp ' + self.in_testlabels().path + ' ' + self.out_testlabels().path)
        os.system('cp ' + self.in_traindocuments().path + ' ' + self.out_traindocuments().path)
        os.system('cp ' + self.in_testdocuments().path + ' ' + self.out_testdocuments().path)

        # open classifier args
        with open(self.in_classifier_args().path) as infile:
            classifier_args = infile.read().rstrip().split('\n')

        print('Running ' + str(i) + 'th experiment')
        if self.classifier == 'svorim':
            svorim_path = classifier_args[0]
            yield ExperimentComponentSvorimVector(train=self.out_trainvectors().path, trainlabels=self.out_trainlabels().path, test=self.out_testvectors().path, testlabels=self.out_testlabels().path, documents=self.out_testdocuments().path, svorim_path=svorim_path)
        elif self.classifier == 'dtc':
            minimum_per_class = int(classifier_args[0])
            minimum_IG = classifier_args[1]
            yield ExperimentComponentDTCVector(train=self.out_trainvectors().path, trainlabels=self.out_trainlabels().path, test=self.out_testvectors().path, testlabels=self.out_testlabels().path, documents=self.out_testdocuments().path, featurenames=self.in_featurenames().path, ordinal=self.ordinal, minimum_per_class=minimum_per_class, minimum_IG=minimum_IG)
        else:
            yield ExperimentComponentVector(train=self.out_trainvectors().path, trainlabels=self.out_trainlabels().path, test=self.out_testvectors().path, testlabels=self.out_testlabels().path, classifier_args=self.out_classifier_args().path, documents=self.out_testdocuments().path, classifier=self.classifier, ordinal=self.ordinal) 


################################################################################
###Reporter
################################################################################

@registercomponent
class ReportRepeatsComponent(StandardWorkflowComponent):

    def accepts(self):
        return InputFormat(self, format_id='repeatdirectory', extension='.repeats')
                    
    def autosetup(self):
        return ReportRepeats

class ReportRepeats(Task):

    in_repeatdirectory = InputSlot()

    ordinal = BoolParameter()

    # def out_avg_performance(self):
    #     return self.outputfrominput(inputformat='expdirectory', stripextension='.exp', addextension='.avg_performance.csv')    

    def out_docpredictions(self):
        return self.outputfrominput(inputformat='repeatdirectory', stripextension='.repeats', addextension='.all_docpredictions.csv')
 
    def run(self):

        # gather fold reports
        # performance_files = [ filename for filename in glob.glob(self.in_repeatdirectory().path + '/repeat*/*.performance.csv') ]
        docprediction_files = [ filename for filename in glob.glob(self.in_repeatdirectory().path + '/repeat*/*.docpredictions.csv') ]

        # write predictions per document
        doc_all_predictions = defaultdict(list)
        docpredictions = sum([dr.parse_csv(docprediction_file)[1:] for docprediction_file in docprediction_files], [])
        for dp in docpredictions:
            if len(doc_all_predictions[dp[0]]) == 0:
                doc_all_predictions[dp[0]].append(dp[1])
            doc_all_predictions[dp[0]].append(dp[2])

        for doc in doc_all_predictions.keys():
            line = [doc] + doc_all_predictions[doc]
            lines.append(line)
        lw = linewriter.Linewriter(lines)
        lw.write_csv(self.out_docpredictions().path)            
