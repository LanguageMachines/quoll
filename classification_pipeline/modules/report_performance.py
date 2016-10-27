
import numpy
from scipy import sparse
from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

import functions.reporter as reporter
import functions.linewriter as linewriter

class ReportPerformance(Task):

    in_predictions = InputSlot()
    in_labels = InputSlot()
    
    documents = Parameter()

    def out_performance(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.classifications.txt', addextension='.performance.csv')

    def out_docpredictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.classifications.txt', addextension='.docpredictions.csv')

    def out_confusionmatrix(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.classifications.txt', addextension='.confusion_matrix.csv')

    def out_fps_dir(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.classifications.txt', addextension='.ranked_fps')

    def out_tps_dir(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.classifications.txt', addextension='.ranked_tps')

    def run(self):

        # load predictions and probabilities
        with open(self.in_predictions().path) as infile:
            predictions_probabilities = [line.split('\t') for line in infile.read().split('\n')]
        predictions = [x[0] for x in predictions_probabilities]
        probabilities = [x[1] for x in predictions_probabilities]

        # load labels
        with open(self.in_labels().path) as infile:
            labels = infile.read().split('\n')

        # load documents
        if self.documents:
            with open(self.documents,'r',encoding='utf-8') as infile:
                documents = infile.read().split('\n')
        else:
            documents = False

        # initiate reporter
        rp = reporter.Reporter(predictions, labels, probabilities, documents)

        # report performance
        performance = rp.assess_performance()
        lw = linewriter.Linewriter(performance)
        lw.write_csv(self.out_performance().path)

        # report predictions by document
        predictions_by_document = rp.predictions_by_document()
        lw = linewriter.Linewriter(predictions_by_document)
        lw.write_csv(self.out_docpredictions().path)

        # report confusion matrix
        confusion_matrix = rp.return_confusion_matrix()
        with open(self.out_confusionmatrix().path,'w') as cm_out:
            cm_out.write(confusion_matrix)

        # report fps per label
        self.setup_output_dir(self.out_fps_dir().path)
        for label in list(set(labels)):
            ranked_fps = rp.return_ranked_fps(label)
            outfile = self.out_fps_dir().path + '/' + label + '.csv'
            lw = linewriter.Linewriter(ranked_fps)
            lw.write_csv(outfile)

        # report fps per label
        self.setup_output_dir(self.out_tps_dir().path)
        for label in list(set(labels)):
            ranked_tps = rp.return_ranked_tps(label)
            outfile = self.out_tps_dir().path + '/' + label + '.csv'
            lw = linewriter.Linewriter(ranked_tps)
            lw.write_csv(outfile)

@registercomponent
class ReporterComponent(WorkflowComponent):    
   
    predictions = Parameter()
    labels = Parameter()
    documents = Parameter(default=False)

    def accepts(self):
        return [ ( InputFormat(self, format_id='predictions', extension='.classifications.txt',inputparameter='predictions'), InputFormat(self, format_id='labels', extension='.vectorlabels', inputparameter='labels'), InputFormat(self, format_id='documents', extension='.txt',inputparameter='documents') ) ]
                                
    def setup(self, workflow, input_feeds):

        reporter = workflow.new_task('report_performance', ReportPerformance, autopass=True, documents=self.documents)
        reporter.in_predictions = input_feeds['predictions']
        reporter.in_labels = input_feeds['labels']

        return reporter
