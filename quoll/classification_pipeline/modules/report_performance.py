
import numpy
from scipy import sparse
from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

import quoll.classification_pipeline.functions.reporter as reporter
import quoll.classification_pipeline.functions.linewriter as linewriter
import quoll.classification_pipeline.functions.regression_reporter as regression_reporter

class ReportPerformance(Task):

    in_predictions = InputSlot()
    in_full_predictions = InputSlot()
    in_labels = InputSlot()
    in_trainlabels = InputSlot()
    in_documents = InputSlot()

    ordinal = BoolParameter()

    def out_performance(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.performance.csv')

    def out_docpredictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.docpredictions.csv')

    def out_confusionmatrix(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.confusion_matrix.csv')

    def out_fps_dir(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.ranked_fps')

    def out_tps_dir(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.ranked_tps')

    def out_fns_dir(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.ranked_fns')

    def out_tns_dir(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.ranked_tns')

    def run(self):

        # load predictions and full_predictions
        with open(self.in_predictions().path) as infile:
            predictions = infile.read().strip().split('\n')

        with open(self.in_full_predictions().path) as infile:
            lines = [line.split('\t') for line in infile.read().strip().split('\n')]
        label_order = lines[0]
        full_predictions = lines[1:]

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
        rp = reporter.Reporter(predictions, full_predictions, label_order, labels, unique_labels, self.ordinal, documents)

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
        for label in list(set(unique_labels)):
            ranked_fps = rp.return_ranked_fps(label)
            outfile = self.out_fps_dir().path + '/' + label + '.csv'
            lw = linewriter.Linewriter(ranked_fps)
            lw.write_csv(outfile)

        # report fps per label
        self.setup_output_dir(self.out_tps_dir().path)
        for label in list(set(unique_labels)):
            ranked_tps = rp.return_ranked_tps(label)
            outfile = self.out_tps_dir().path + '/' + label + '.csv'
            lw = linewriter.Linewriter(ranked_tps)
            lw.write_csv(outfile)

        # report fns per label
        self.setup_output_dir(self.out_fns_dir().path)
        for label in list(set(unique_labels)):
            ranked_fns = rp.return_ranked_fns(label)
            outfile = self.out_fns_dir().path + '/' + label + '.csv'
            lw = linewriter.Linewriter(ranked_fns)
            lw.write_csv(outfile)

        # report tns per label
        self.setup_output_dir(self.out_tns_dir().path)
        for label in list(set(unique_labels)):
            ranked_tns = rp.return_ranked_tns(label)
            outfile = self.out_tns_dir().path + '/' + label + '.csv'
            lw = linewriter.Linewriter(ranked_tns)
            lw.write_csv(outfile)

        # report confusion matrix
        if self.ordinal: # to make a confusion matrix, the labels should be formatted as string
            rp = reporter.Reporter(predictions, full_predictions, label_order, labels, unique_labels, False, documents)
        confusion_matrix = rp.return_confusion_matrix()
        with open(self.out_confusionmatrix().path,'w') as cm_out:
            cm_out.write(confusion_matrix)


class ReportRegressionPerformance(Task):

    in_predictions = InputSlot()
    in_labels = InputSlot()

    def out_performance(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.regression_performance.csv')

    def run(self):

        # load predictions and full_predictions
        with open(self.in_predictions().path) as infile:
            predictions = [float(x) for x in infile.read().strip().split('\n')]
            
        # load labels
        with open(self.in_labels().path) as infile:
            labels = [float(x) for x in infile.read().strip().split('\n')]

        # report performance
        performance = regression_reporter.assess_performance(labels,predictions)
        with open(self.out_performance().path,'w',encoding='utf-8') as out:
            out.write(str(performance))
            
class ReportDocpredictions(Task):

    in_predictions = InputSlot()
    in_full_predictions = InputSlot()
    in_documents = InputSlot()

    def out_docpredictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.full_predictions.txt', addextension='.docpredictions.csv')

    def run(self):

        # load predictions and full_predictions
        with open(self.in_predictions().path) as infile:
            predictions = infile.read().strip().split('\n')

        with open(self.in_full_predictions().path) as infile:
            lines = [line.split('\t') for line in infile.read().strip().split('\n')]
        label_order = lines[0]
        full_predictions = lines[1:]

        # load documents
        with open(self.in_documents().path,'r',encoding='utf-8') as infile:
            documents = infile.read().strip().split('\n')

        # initiate reporter
        rp = reporter.Reporter(predictions, full_predictions, label_order, documents=documents)

        # report predictions by document
        predictions_by_document = rp.predictions_by_document()
        lw = linewriter.Linewriter(predictions_by_document)
        lw.write_csv(self.out_docpredictions().path)


#################################################################
### Components ##################################################
#################################################################

@registercomponent
class ReporterComponent(WorkflowComponent):

    predictions = Parameter()
    full_predictions = Parameter()
    labels = Parameter()
    trainlabels = Parameter()
    documents = Parameter()

    ordinal = BoolParameter()

    def accepts(self):
        return [ ( InputFormat(self, format_id='predictions', extension='.predictions.txt',inputparameter='predictions'), InputFormat(self, format_id='full_predictions', extension='.full_predictions.txt',inputparameter='full_predictions'), InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='documents', extension='.txt',inputparameter='documents') ) ]

    def setup(self, workflow, input_feeds):

        reporter = workflow.new_task('report_performance', ReportPerformance, autopass=True, ordinal=self.ordinal)
        reporter.in_predictions = input_feeds['predictions']
        reporter.in_full_predictions = input_feeds['full_predictions']
        reporter.in_labels = input_feeds['labels']
        reporter.in_trainlabels = input_feeds['trainlabels']
        reporter.in_documents = input_feeds['documents']

        return reporter

@registercomponent
class RegressionReporterComponent(WorkflowComponent):

    predictions = Parameter()
    labels = Parameter()

    def accepts(self):
        return [ ( InputFormat(self, format_id='predictions', extension='.predictions.txt',inputparameter='predictions'), InputFormat(self, format_id='labels', extension='.txt', inputparameter='labels') ) ]

    def setup(self, workflow, input_feeds):

        regreporter = workflow.new_task('report_regression_performance', ReportRegressionPerformance, autopass=True)
        regreporter.in_predictions = input_feeds['predictions']
        regreporter.in_labels = input_feeds['labels']

        return regreporter
    
class ReportDocpredictionsComponent(WorkflowComponent):

    predictions = Parameter()
    full_predictions = Parameter()
    documents = Parameter()

    def accepts(self):
        return [ ( InputFormat(self, format_id='predictions', extension='.predictions.txt',inputparameter='predictions'), InputFormat(self, format_id='full_predictions', extension='.full_predictions.txt',inputparameter='full_predictions'), InputFormat(self, format_id='documents', extension='.txt',inputparameter='documents') ) ]

    def setup(self, workflow, input_feeds):

        reportdocs = workflow.new_task('report_performance', ReportDocpredictions, autopass=True)
        reportdocs.in_predictions = input_feeds['predictions']
        reportdocs.in_full_predictions = input_feeds['full_predictions']
        reportdocs.in_documents = input_feeds['documents']

        return reportdocs
