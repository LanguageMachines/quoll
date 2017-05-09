
import numpy
from scipy import sparse
from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

import quoll.classification_pipeline.functions.reporter as reporter
import quoll.classification_pipeline.functions.linewriter as linewriter

class ReportPerformance(Task):

    in_predictions = InputSlot()
    in_labels = InputSlot()
    in_trainlabels = InputSlot()
    in_documents = InputSlot()

    ordinal = BoolParameter()

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
            predictions_probabilities = [line.split('\t') for line in infile.read().strip().split('\n')]
        predictions = [x[0] for x in predictions_probabilities]
        probabilities = [x[1] for x in predictions_probabilities]

        # load labels
        with open(self.in_labels().path) as infile:
            labels = infile.read().strip().split('\n')

        # load trainlabels
        with open(self.in_trainlabels().path) as infile:
            unique_labels = list(set(infile.read().strip().split('\n')))

        # load documents
        with open(self.in_documents().path,'r',encoding='utf-8') as infile:
            documents = infile.read().strip().split('\n')

        # initiate reporter
        rp = reporter.Reporter(predictions, probabilities, labels, unique_labels, self.ordinal, documents)

        # report performance
        if self.ordinal:
            performance = rp.assess_ordinal_performance()
        else:
            performance = rp.assess_performance()
        lw = linewriter.Linewriter(performance)
        lw.write_csv(self.out_performance().path)

        # report predictions by document
        predictions_by_document = rp.predictions_by_document()
        lw = linewriter.Linewriter(predictions_by_document)
        lw.write_csv(self.out_docpredictions().path)

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

        # report confusion matrix
        if self.ordinal: # to make a confusion matrix, the labels should be formatted as string
            rp = reporter.Reporter([str(x) for x in predictions], probabilities, [str(x) for x in labels], [str(x) for x in unique_labels], False, documents)
        confusion_matrix = rp.return_confusion_matrix()
        with open(self.out_confusionmatrix().path,'w') as cm_out:
            cm_out.write(confusion_matrix)

class ReportDocpredictions(Task):

    in_predictions = InputSlot()
    in_documents = InputSlot()

    def out_docpredictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.classifications.txt', addextension='.docpredictions.csv')

    def run(self):

        # load predictions and probabilities
        with open(self.in_predictions().path) as infile:
            predictions_probabilities = [line.split('\t') for line in infile.read().strip().split('\n')]
        predictions = [x[0] for x in predictions_probabilities]
        probabilities = [x[1] for x in predictions_probabilities]

        # load documents
        with open(self.in_documents().path,'r',encoding='utf-8') as infile:
            documents = infile.read().strip().split('\n')

        # initiate reporter
        rp = reporter.Reporter(predictions, probabilities, documents=documents)

        # report predictions by document
        predictions_by_document = rp.predictions_by_document()
        lw = linewriter.Linewriter(predictions_by_document)
        lw.write_csv(self.out_docpredictions().path)

@registercomponent
class ReporterComponent(WorkflowComponent):

    predictions = Parameter()
    labels = Parameter()
    documents = Parameter()
    ordinal = BoolParameter()

    def accepts(self):
        return [ ( InputFormat(self, format_id='predictions', extension='.classifications.txt',inputparameter='predictions'), InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), InputFormat(self, format_id='documents', extension='.txt',inputparameter='documents') ) ]

    def setup(self, workflow, input_feeds):

        reporter = workflow.new_task('report_performance', ReportPerformance, autopass=True, documents=self.documents, ordinal=self.ordinal)
        reporter.in_predictions = input_feeds['predictions']
        reporter.in_labels = input_feeds['labels']
        reporter.in_documents = input_feeds['documents']

        return reporter

class ReportDocpredictionsComponent(WorkflowComponent):

    predictions = Parameter()
    documents = Parameter()

    def accepts(self):
        return [ ( InputFormat(self, format_id='predictions', extension='.classifications.txt',inputparameter='predictions'), InputFormat(self, format_id='documents', extension='.txt',inputparameter='documents') ) ]

    def setup(self, workflow, input_feeds):

        reportdocs = workflow.new_task('report_performance', ReportDocpredictions, autopass=True)
        reportdocs.in_predictions = input_feeds['predictions']
        reportdocs.in_documents = input_feeds['documents']

        return reportdocs
