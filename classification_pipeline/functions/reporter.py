
from sklearn.metrics import auc

from pynlpl import evaluation

class Reporter:

    def __init__(self, predictions, labels, probabilities=False, documents=False):
        if len(predictions) != len(labels):
            print('The number of predictions (', len(predictions), ') does not align with the number of labels (', len(labels), '); exiting program')
            quit()
        self.ce = self.save_classifier_output(labels, predictions)
        self.labels = labels 
        self.unique_labels = list(set(labels))
        self.predictions = predictions
        if documents:
            if len(predictions) != len(documents):
                print('The number of documents (', len(documents), ') does not align with the number of predictions and labels (', len(predictions), '); exiting program')
                quit()
            else:
                self.documents = documents
        else:
            self.documents = ['-'] * len(labels)
        if probabilities:
            if len(predictions) != len(probabilities):
                print('The number of probabilities (', len(probabilities), ') does not align with the number of predictions and labels (', len(predictions), '); exiting program')
                quit()
            else:
                self.probabilities = probabilities
        else:
            self.probabilities = ['-'] * len(labels) 

    def save_classifier_output(self, labels, predictions):
        ce = evaluation.ClassEvaluation()
        for i, instance in enumerate(labels):
            ce.append(labels[i], predictions[i])
        return ce

    def assess_label_performance(self, label):
        label_performance = [self.ce.precision(cls=label), self.ce.recall(cls=label), self.ce.fscore(cls=label),self.ce.tp_rate(cls=label), self.ce.fp_rate(cls=label), auc([0, self.ce.fp_rate(cls=label), 1], [0, self.ce.tp_rate(cls=label), 1]), self.ce.tp[label] + self.ce.fn[label], self.ce.tp[label] + self.ce.fp[label], self.ce.tp[label]]
        label_performance = [label] + [round(x,2) for x in label_performance]
        return label_performance

    def assess_micro_performance(self):
        micro_performance = [self.ce.precision(), self.ce.recall(), self.ce.fscore(), self.ce.tp_rate(), self.ce.fp_rate(), auc([0, self.ce.fp_rate(), 1], [0, self.ce.tp_rate(), 1]), len(self.ce.observations), len(self.ce.observations), sum([self.ce.tp[label] for label in self.labels])]
        micro_performance = ['micro'] + [round(x,2) for x in micro_performance]        
        return micro_performance

    def assess_performance(self):
        performance_headers = ["Cat", "Pr", "Re", "F1", "TPR", "FPR", "AUC", "Tot", "Clf", "Cor"] 
        performance = [performance_headers]
        for label in self.unique_labels:
            performance.append(self.assess_label_performance(label))
        performance.append(self.assess_micro_performance())
        return performance
    
    def predictions_by_document(self):
        predictions = [['document', 'target', 'prediction', 'prob']] 
        for index in range(len(self.documents)):
            predictions.append([self.documents[index], self.labels[index], self.predictions[index], self.probabilities[index]])
        return predictions

    def return_confusion_matrix(self):
        confusion_matrix = self.ce.confusionmatrix()
        return confusion_matrix.__str__()

    def return_ranked_fps(self, label):
        ranked_fps = sorted([[self.documents[i], self.labels[i], self.predictions[i], self.probabilities[i]] for i in range(len(self.labels)) if self.predictions[i] == label and self.labels[i] != label], key=lambda k : k[3], reverse = True)
        return ranked_fps

    def return_ranked_tps(self, label):
        ranked_tps = sorted([[self.documents[i], self.labels[i], self.predictions[i], self.probabilities[i]] for i in range(len(self.labels)) if self.predictions[i] == label and self.labels[i] == label], key=lambda k : k[3], reverse = True)
        return ranked_tps
