
from sklearn import preprocessing
from sklearn import svm, naive_bayes, tree
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import OutputCodeClassifier
import mord
import numpy

class AbstractSKLearnClassifier:

    def __init__(self):
        self.label_encoder = preprocessing.LabelEncoder()

    def set_label_encoder(self, labels):
        self.label_encoder.fit(labels)

    def return_label_encoder(self):
        return self.label_encoder

    def return_label_encoding(self, labels):
        encoding = [str(x) for x in self.label_encoder.transform(list(set(labels)))]
        label_encoding = list(zip(labels,encoding))
        return label_encoding

    def apply_model(self, clf, testvectors):
        predictions = []
        full_predictions = [list(self.label_encoder.classes_)]
        for i, instance in enumerate(testvectors):
            prediction = self.label_encoder.inverse_transform([clf.predict(instance)[0]])[0]
            predictions.append(prediction)
            try:
                full_predictions.append([clf.predict_proba(instance)[0][c] for c in self.label_encoder.transform(list(self.label_encoder.classes_))])
            except: # no probabilities connected to predictions
                full_predictions.append(['-' for c in self.label_encoder.classes_])
        # try:
        #     output = list(zip(list(self.label_encoder.inverse_transform(predictions)),probabilities))
        # except: # might be negatives in there
        #     predictions = [int(x) for x in predictions]
        #     output = list(zip(list(self.label_encoder.inverse_transform(predictions)),probabilities))
        return predictions, full_predictions

class NaiveBayesClassifier(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)
    
    def train_classifier(self, trainvectors, labels, alpha='default', fit_prior=True, iterations=10):
        if alpha == '':
            paramsearch = GridSearchCV(estimator=naive_bayes.MultinomialNB(), param_grid=dict(alpha=numpy.linspace(0,2,20)[1:]), n_jobs=6)
            paramsearch.fit(trainvectors,self.label_encoder.transform(labels))
            selected_alpha = paramsearch.best_estimator_.alpha
        elif alpha == 'default':
            selected_alpha = 1.0
        else:
            selected_alpha = alpha
        if fit_prior == 'False':
            fit_prior = False
        else:
            fit_prior = True
        self.model = naive_bayes.MultinomialNB(alpha=selected_alpha,fit_prior=fit_prior)
        self.model.fit(trainvectors, self.label_encoder.transform(labels))

    def return_classifier(self):
        return self.model

    def return_model_insights(self):
        return [['feature_log_prob.txt','\n'.join([str(x) for x in self.model.feature_log_prob_.T.tolist()])]]

    def apply_classifier(self, testvectors):
        classifications = AbstractSKLearnClassifier.apply_model(self, self.model, testvectors)
        return classifications

class SVMClassifier(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)

    def train_classifier(self, trainvectors, labels, c='', kernel='', gamma='', degree='', iterations=10):
        if len(self.label_encoder.classes_) > 2: # more than two classes to distinguish
            parameters = ['estimator__C', 'estimator__kernel', 'estimator__gamma', 'estimator__degree']
            multi = True
        else: # only two classes to distinguish
            parameters = ['C', 'kernel', 'gamma', 'degree']
            multi = False
        c_values = [0.001, 0.005, 0.01, 0.5, 1, 5, 10, 50, 100, 500, 1000] if c == '' else [float(x) for x in c.split()]
        kernel_values = ['linear', 'rbf', 'poly'] if kernel == '' else [k for  k in kernel.split()]
        gamma_values = [0.0005, 0.002, 0.008, 0.032, 0.128, 0.512, 1.024, 2.048] if gamma == '' else [float(x) for x in gamma.split()]
        degree_values = [1, 2, 3, 4] if degree == '' else [int(x) for x in degree.split()]
        grid_values = [c_values, kernel_values, gamma_values, degree_values]
        if not False in [len(x) == 1 for x in grid_values]: # only sinle parameter settings
            settings = {}
            for i, parameter in enumerate(parameters):
                settings[parameter] = grid_values[i][0]
        else:
            iterations=int(iterations)
            param_grid = {}
            for i, parameter in enumerate(parameters):
                param_grid[parameter] = grid_values[i]
            model = svm.SVC(probability=True)
            if multi:
                model = OutputCodeClassifier(model)
            paramsearch = RandomizedSearchCV(model, param_grid, cv = 5, verbose = 2, n_iter = iterations, n_jobs = 10, pre_dispatch = 4)
            paramsearch.fit(trainvectors, self.label_encoder.transform(labels))
            settings = paramsearch.best_params_
        # train an SVC classifier with the settings that led to the best performance
        self.model = svm.SVC(
           probability = True,
           C = settings[parameters[0]],
           kernel = settings[parameters[1]],
           gamma = settings[parameters[2]],
           degree = settings[parameters[3]],
           cache_size = 1000,
           verbose = 2
           )
        if multi:
            self.model = OutputCodeClassifier(self.model)
        self.model.fit(trainvectors, self.label_encoder.transform(labels))

    def return_classifier(self):
        return self.model

    def return_model_insights(self):
        return [['support_vectors.txt','\n'.join(self.model.support_vectors_.T.tolist())],['feature_weights.txt','\n'.join([' '.join(l) for l in self.model.coef_.T.tolist()])]]

    def apply_classifier(self, testvectors):
        classifications = AbstractSKLearnClassifier.apply_model(self, self.model, testvectors)
        return classifications

class LogisticRegressionClassifier(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)

    def train_classifier(self, trainvectors, labels, c='', solver='', dual='', penalty='', multiclass='', max_iterations=1000, iterations=10):
        if len(self.label_encoder.classes_) > 2: # more than two classes to distinguish
            parameters = ['estimator__C', 'estimator__solver', 'estimator__penalty', 'estimator__dual', 'estimator__multi_class']
            # multi = True
        else: # only two classes to distinguish
            parameters = ['C', 'solver', 'penalty', 'dual', 'multi_class']
            # multi = False
        c_values = [0.001, 0.005, 0.01, 0.5, 1, 5, 10, 50, 100, 500, 1000] if c == '' else [float(x) for x in c.split()]
        solver_values = ['newton-cg', 'lbfgs', 'liblinear', 'sag'] if solver == '' else [s for  s in solver.split()]
        if penalty == '':
            if not set(['newton-cg','lbfgs','sag']) & set(solver_values):
                penalty_values = ['l1', 'l2']
            else:
                penalty_values = ['l2']
        else:
            penalty_values = [penalty]
        if dual == '':
            if len(solver_values) == 1 and solver_values[0] == 'liblinear':
                if len(penalty_values) == 1 and penalty_values[0] == 'l2':
                    dual_values = [True,False]
            else:
                dual_values = [False]
        else:
            dual_values = [int(dual)] # 1 or 0
        if multiclass == '':
            if 'liblinear' not in solver_values:
                multiclass_values = ['ovr', 'multinomial']
            else:
                multiclass_values = ['ovr']
        else:
            multiclass_values = [multiclass]
        grid_values = [c_values, solver_values, penalty_values, dual_values, multiclass_values]
        max_iterations = int(max_iterations)
        if not False in [len(x) == 1 for x in grid_values]: # only sinle parameter settings
            settings = {}
            for i, parameter in enumerate(parameters):
                settings[parameter] = grid_values[i][0]
        else: # try different parameter combinations
            iterations=int(iterations)
            param_grid = {}
            for i, parameter in enumerate(parameters):
                param_grid[parameter] = grid_values[i]
            model = LogisticRegression(max_iter=max_iterations)
            # if multi:
            #     model = OutputCodeClassifier(model)
            paramsearch = RandomizedSearchCV(model, param_grid, cv = 5, verbose = 2, n_iter = iterations, n_jobs = 10, pre_dispatch = 4)
            paramsearch.fit(trainvectors, self.label_encoder.transform(labels))
            settings = paramsearch.best_params_
        # train a logistic regression classifier with the settings that led to the best performance
        self.model = LogisticRegression(
            C = settings[parameters[0]],
            solver = settings[parameters[1]],
            penalty = settings[parameters[2]],
            dual = settings[parameters[3]],
            multi_class = settings[parameters[4]],
            max_iter = max_iterations,
            verbose = 2
        )
        # if multi:
        #     self.model = OutputCodeClassifier(self.model)
        self.model.fit(trainvectors, self.label_encoder.transform(labels))

    def return_classifier(self):
        return self.model

    def return_model_insights(self):
        return [['feature_weights.txt','\n'.join([str(l) for l in self.model.coef_.T.tolist()])]]

    def apply_classifier(self, testvectors):
        classifications = AbstractSKLearnClassifier.apply_model(self, self.model, testvectors)
        return classifications

class TreeClassifier(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)
    
    def train_classifier(self, trainvectors, labels, class_weight=None):
        self.model = tree.DecisionTreeClassifier(class_weight=class_weight)
        self.model.fit(trainvectors, self.label_encoder.transform(labels))

    def return_classifier(self):
        return self.model

    def return_model_insights(self):
        return [['feature_importances_gini.txt','\n'.join([str(x) for x in self.model.feature_importances_.T.tolist()])],['tree.txt',self.model.tree_.__str__()]]

    def apply_classifier(self, testvectors):
        classifications = AbstractSKLearnClassifier.apply_model(self, self.model, testvectors)
        return classifications

class PerceptronLClassifier(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)

    def train_classifier(self, trainvectors, labels, alpha='', iterations=50, jobs=10):
        iterations = int(iterations)
        jobs = int(jobs)
        if alpha == '':
            paramsearch = GridSearchCV(estimator=Perceptron(), param_grid=dict(alpha=numpy.linspace(0,2,20)[1:],n_iter=[iterations]), n_jobs=jobs)
            paramsearch.fit(trainvectors,self.label_encoder.transform(labels))
            selected_alpha = paramsearch.best_estimator_.alpha
        elif alpha == 'default':
            selected_alpha = 1.0
        else:
            selected_alpha = alpha
        # train a perceptron with the settings that led to the best performance
        self.model = Perceptron(alpha=selected_alpha,n_iter=iterations,n_jobs=jobs)
        self.model.fit(trainvectors, self.label_encoder.transform(labels))

    def return_classifier(self):
        return self.model

    def apply_classifier(self, testvectors):
        classifications = AbstractSKLearnClassifier.apply_model(self, self.model, testvectors)
        return classifications

class OrdinalRidge(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)

    def train_classifier(self, trainvectors, labels):
        self.model = mord.OrdinalRidge()
        self.model.fit(trainvectors,self.label_encoder.transform(labels))

    def return_classifier(self):
        return self.model

    def apply_classifier(self, testvectors):
        classifications = AbstractSKLearnClassifier.apply_model(self, self.model, testvectors)
        return classifications

class OrdinalLogisticAT(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)

    def train_classifier(self, trainvectors, labels):
        self.model = mord.LogisticAT(alpha=1.,max_iter=50000)
        self.model.fit(trainvectors,self.label_encoder.transform(labels))

    def return_classifier(self):
        return self.model

    def apply_classifier(self, testvectors):
        classifications = AbstractSKLearnClassifier.apply_model(self, self.model, testvectors)
        return classifications

class OrdinalLogisticIT(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)

    def train_classifier(self, trainvectors, labels):
        self.model = mord.LogisticIT(alpha=1.,max_iter=50000)
        self.model.fit(trainvectors,self.label_encoder.transform(labels))

    def return_classifier(self):
        return self.model

    def apply_classifier(self, testvectors):
        classifications = AbstractSKLearnClassifier.apply_model(self, self.model, testvectors)
        return classifications

class OrdinalLogisticSE(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)

    def train_classifier(self, trainvectors, labels):
        self.model = mord.LogisticSE(alpha=1.,max_iter=50000)
        self.model.fit(trainvectors,self.label_encoder.transform(labels))

    def return_classifier(self):
        return self.model

    def apply_classifier(self, testvectors):
        classifications = AbstractSKLearnClassifier.apply_model(self, self.model, testvectors)
        return classifications
