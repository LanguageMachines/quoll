
from sklearn import preprocessing
from sklearn import svm, naive_bayes, tree
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import OutputCodeClassifier

class SKLearnClassifier:

    def __init__(self):
        self.label_encoder = preprocessing.LabelEncoder()
        
    def set_label_encoder(self, labels):
        self.label_encoder.fit(labels)

    def return_label_encoding(self, labels):
        label_encoding = []
        for label in list(set(labels)):
            label_encoding.append((label, str(self.label_encoder.transform(label))))
        return label_encoding

    def train_classifier(self, classifier, trainvectors, labels):
        classifier_dict = {'naive_bayes':self.train_naive_bayes, 'tree':self.train_decision_tree}
        clf = classifier_dict[classifier](trainvectors, labels)
        return clf

    def train_naive_bayes(self, trainvectors, labels):
        clf = naive_bayes.MultinomialNB()
        clf.fit(trainvectors, self.label_encoder.transform(labels))        
        return clf

    def train_decision_tree(self, trainvectors, labels):
        clf = tree.DecisionTreeClassifier()
        clf.fit(trainvectors, self.label_encoder.transform(labels))
        return clf

    def apply_model(self, clf, testvectors):
        predictions = []
        probabilities = []
        for i, instance in enumerate(testvectors):
            predictions.append(clf.predict(instance)[0])
            try:
                probabilities.append(clf.predict_proba(instance)[0][prediction])
            except:
                probabilities.append('-')
        output = list(zip(list(self.label_encoder.inverse_transform(predictions)),probabilities))
        return output
