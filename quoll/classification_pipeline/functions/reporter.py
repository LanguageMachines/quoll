
from sklearn.metrics import auc

from pynlpl import evaluation

class Reporter:

    def __init__(self, predictions, full_predictions, label_order, labels=False, unique_labels = False, ordinal=False, documents=False, strictness=1):
        self.predictions = predictions
        if len(predictions) != len(full_predictions):
            print('The number of full predictions (', len(full_predictions), ') does not align with the number of predictions and labels (', len(predictions), '); exiting program')
            quit()
        else:
            self.full_predictions = full_predictions
            self.label_order = label_order
        if labels:
            if len(predictions) != len(labels):
                print('The number of predictions (', len(predictions), ') does not align with the number of labels (', len(labels), '); exiting program')
                quit()
            if unique_labels:
                self.unique_labels = unique_labels
            else:
                self.unique_labels = list(set(self.labels))
            if ordinal:
                self.ce = evaluation.OrdinalEvaluation()
                self.labels = [int(label) for label in labels]
                self.unique_labels = [int(label) for label in unique_labels]
                self.predictions = [int(prediction) for prediction in predictions]
            else:
                self.ce = evaluation.ClassEvaluation()
                self.labels = labels
            self.save_classifier_output(self.labels, self.predictions, self.full_predictions, strictness)
        else:
            self.labels = ['-'] * len(self.predictions)
            self.unique_labels = ['-']
        if documents:
            if len(predictions) != len(documents):
                print('The number of documents (', len(documents), ') does not align with the number of predictions and labels (', len(predictions), '); exiting program')
                quit()
            else:
                self.documents = documents
        else:
            self.documents = ['-'] * len(labels)

    def save_classifier_output(self, labels, predictions, full_predictions, strictness=1):
        for i, instance in enumerate(labels):
            if strictness>1 and full_predictions[i][0] != '-' and len(self.label_order) >= strictness:
                fp_numbered = [[j,x] for j,x in enumerate(full_predictions[i])]
                fp_sorted = sorted(fp_numbered,key = lambda k : k[1],reverse=True)
                top_n_predictions = [self.label_order[fp_sorted[j][0]] for j in list(range(strictness))]
                if labels[i] in top_n_predictions:
                    self.ce.append(labels[i],labels[i])
                else:
                    self.ce.append(labels[i],predictions[i])
            else:
                self.ce.append(labels[i], predictions[i])

    def assess_ordinal_label_performance(self, label):
        self.ce.compute()
        mae = self.ce.mae(label) if len(self.ce.error[label]) > 0 else 0.0
        rmse = self.ce.rmse(label) if len(self.ce.error[label]) > 0 else 0.0
        ordinal_label_performance = [self.ce.precision(cls=label), self.ce.recall(cls=label), self.ce.fscore(cls=label),self.ce.tp_rate(cls=label), self.ce.fp_rate(cls=label), auc([0, self.ce.fp_rate(cls=label), 1], [0, self.ce.tp_rate(cls=label),1]), mae, rmse, self.ce.accuracy(cls=label), self.ce.tp[label] + self.ce.fn[label], self.ce.tp[label] + self.ce.fp[label], self.ce.tp[label]]
        ordinal_label_performance = [label] + [round(x,2) for x in ordinal_label_performance]
        return ordinal_label_performance       

    def assess_overall_ordinal_performance(self):
        overall_performance = [self.ce.precision(), self.ce.recall(), self.ce.fscore(), self.ce.tp_rate(), self.ce.fp_rate(), auc([0, self.ce.fp_rate(), 1], [0, self.ce.tp_rate(), 1]), self.ce.mae(), self.ce.rmse(), self.ce.accuracy(), len(self.ce.observations), len(self.ce.observations), sum([self.ce.tp[label] for label in list(set(self.labels))])]
        overall_performance = ['overall'] + [round(x,2) for x in overall_performance]
        return overall_performance

    def assess_ordinal_performance(self):
        performance_headers = ["Cat", "Pr", "Re", "F1", "TPR", "FPR", "AUC", "MAE", "RMSE", "ACC", "Tot", "Clf", "Cor"]
        performance = [performance_headers]
        for label in sorted(self.unique_labels):
            if label in self.labels or label in self.predictions:
                performance.append(self.assess_ordinal_label_performance(label))
            else:
                performance.append([label,0,0,0,0,0,0,0,0,0,0,0,0])
        performance.append(self.assess_overall_ordinal_performance())
        return performance

    def assess_label_performance(self, label):
        label_performance = [self.ce.precision(cls=label), self.ce.recall(cls=label), self.ce.fscore(cls=label),self.ce.tp_rate(cls=label), self.ce.fp_rate(cls=label), auc([0, self.ce.fp_rate(cls=label), 1], [0, self.ce.tp_rate(cls=label), 1]), self.ce.tp[label] + self.ce.fn[label], self.ce.tp[label] + self.ce.fp[label], self.ce.tp[label]]
        label_performance = [label] + [round(x,2) for x in label_performance]
        return label_performance

    def assess_micro_performance(self):
        micro_performance = [self.ce.precision(), self.ce.recall(), self.ce.fscore(), self.ce.tp_rate(), self.ce.fp_rate(), auc([0, self.ce.fp_rate(), 1], [0, self.ce.tp_rate(), 1]), len(self.ce.observations), len(self.ce.observations), sum([self.ce.tp[label] for label in list(set(self.labels))])]
        micro_performance = ['micro'] + [round(x,2) for x in micro_performance]
        return micro_performance

    def assess_performance(self):
        performance_headers = ["Cat", "Pr", "Re", "F1", "TPR", "FPR", "AUC", "Tot", "Clf", "Cor"]
        performance = [performance_headers]
        for label in sorted(self.unique_labels):
            if label in self.labels or label in self.predictions:
                performance.append(self.assess_label_performance(label))
            else:
                performance.append([label,0,0,0,0,0,0,0,0,0])
        performance.append(self.assess_micro_performance())
        return performance

    def predictions_by_document(self):
        docpredictions = [['document', 'target', 'prediction'] + ['prediction prob for ' + x for x in self.label_order]] 
        for index in range(len(self.documents)):
            docpredictions.append([self.documents[index], self.labels[index], self.predictions[index]] + self.full_predictions[index])
        return docpredictions

    
    
    def return_confusion_matrix(self):
        confusion_matrix = self.ce.confusionmatrix()
        return confusion_matrix.__str__()                

    def return_ranked_fps(self, label):
        try:
            label_index = self.label_order.index(label)
            ranked_fps = sorted([[self.documents[i], self.labels[i], self.predictions[i], self.full_predictions[i][label_index]] for i in range(len(self.labels)) if self.predictions[i] == label and self.labels[i] != label], key=lambda k : k[3], reverse = True)
        except:
            ranked_fps = []
        return ranked_fps

    def return_ranked_tps(self, label):
        try:
            label_index = self.label_order.index(label)
            ranked_tps = sorted([[self.documents[i], self.labels[i], self.predictions[i], self.full_predictions[i][label_index]] for i in range(len(self.labels)) if self.predictions[i] == label and self.labels[i] == label], key=lambda k : k[3], reverse = True)
        except:
            ranked_tps = []
        return ranked_tps

    def return_ranked_fns(self, label):
        try:
            label_index = self.label_order.index(label)
            ranked_fns = sorted([[self.documents[i], self.labels[i], self.predictions[i], self.full_predictions[i][label_index]] for i in range(len(self.labels)) if self.predictions[i] != label and self.labels[i] == label], key=lambda k : k[3], reverse = True)
        except:
            ranked_fns = []
        return ranked_fns

    def return_ranked_tns(self, label):
        try:
            label_index = self.label_order.index(label)
            ranked_tns = sorted([[self.documents[i], self.labels[i], self.predictions[i], self.full_predictions[i][label_index]] for i in range(len(self.labels)) if self.predictions[i] != label and self.labels[i] != label], key=lambda k : k[3], reverse = True)
        except:
            ranked_tns = []
        return ranked_tns

