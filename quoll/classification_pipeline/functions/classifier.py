
from sklearn import preprocessing
from sklearn import svm, naive_bayes, tree
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import OutputCodeClassifier

import numpy
import multiprocessing

class AbstractSKLearnClassifier:

    def __init__(self):
        self.label_encoder = preprocessing.LabelEncoder()

    def set_label_encoder(self, labels):
        self.label_encoder.fit(labels)

    def return_label_encoder(self):
        return self.label_encoder

    def return_label_encoding(self, labels):
        encoding = [str(x) for x in self.label_encoder.transform(sorted(list(set(labels))))]
        label_encoding = list(zip(labels,encoding))
        return label_encoding

    def predict(self,clf,testvector):
        prediction = self.label_encoder.inverse_transform([clf.predict(testvector)[0]])[0]
        full_prediction = [clf.predict_proba(testvector)[0][c] for c in self.label_encoder.transform(sorted(list(self.label_encoder.classes_)))]
        return prediction, full_prediction

    def apply_model(self, clf, testvectors):
        predictions = []
        full_predictions = [list(self.label_encoder.classes_)]
        for i, instance in enumerate(testvectors):
            prediction, full_prediction = self.predict(clf,instance)
            predictions.append(prediction)
            full_predictions.append(full_prediction)
        return predictions, full_predictions

class NaiveBayesClassifier(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)
    
    def train_classifier(self, trainvectors, labels, alpha='default', fit_prior=True, jobs=4):
        fit_prior = False if fit_prior == 'False' else True
        jobs = int(jobs)
        if alpha == '':
            paramsearch = GridSearchCV(estimator=naive_bayes.MultinomialNB(), param_grid=dict(alpha=numpy.linspace(0,2,20)[1:]), n_jobs=jobs)
            paramsearch.fit(trainvectors,self.label_encoder.transform(labels))
            selected_alpha = paramsearch.best_estimator_.alpha
        elif alpha == 'default':
            selected_alpha = 1.0
        else:
            selected_alpha = float(alpha)
        self.model = naive_bayes.MultinomialNB(alpha=selected_alpha,fit_prior=fit_prior)
        self.model.fit(trainvectors, self.label_encoder.transform(labels))

    def return_classifier(self):
        return self.model

    def return_feature_count(self,vocab=False):
        feature_count = []
        feature_count.append('\t'.join([''] + [self.label_encoder.inverse_transform(self.model.classes_[i]) for i in range(len(self.model.classes_))])) # de namen van de klassen
        if vocab:
            for i,vals in enumerate(self.model.feature_count_.T.tolist()):
                feature_count.append('\t'.join([vocab[i]] + [str(x) for x in vals]))
        else:
            for i,vals in enumerate(self.model.feature_count_.T.tolist()):
                feature_count.append('\t'.join([str(i)] + [str(x) for x in vals]))
        return '\n'.join(feature_count) + '\n'

    def return_feature_log_prob(self,vocab=False):
        feature_log_prob = []
        feature_log_prob.append('\t'.join([''] + [self.label_encoder.inverse_transform(self.model.classes_[i]) for i in range(len(self.model.classes_))])) # de namen van de klassen
        if vocab:
            for i,vals in enumerate(self.model.feature_log_prob_.T.tolist()):
                feature_log_prob.append('\t'.join([vocab[i]] + [str(x) for x in vals]))
        else:
            for i,vals in enumerate(self.model.feature_log_prob_.T.tolist()):
                feature_log_prob.append('\t'.join([str(i)] + [str(x) for x in vals]))
        return '\n'.join(feature_log_prob) + '\n'

    def return_ranked_features(self,vocab=False):
        ranked_features = []
        ranked_features.append('\t'.join([' '.join([self.label_encoder.inverse_transform(self.model.classes_[i]),'log prob']) for i in range(len(self.model.classes_))])) # de namen van de klassen
        sorted_classes = sorted([[self.label_encoder.inverse_transform(self.model.classes_[i]),i] for i,x in enumerate(self.model.class_count_.tolist())],key = lambda k : k[0])
        sorted_features_all_classes = []
        for i,c in enumerate(sorted_classes):
            if vocab:
                features_log_prob_class = [[vocab[j],vals[i]] for j,vals in enumerate(self.model.feature_log_prob_.T.tolist())]
            else:
                features_log_prob_class = [[str(j),vals[i]] for j,vals in enumerate(self.model.feature_log_prob_.T.tolist())]
            sorted_features_log_prob_class = sorted(features_log_prob_class,key = lambda k : k[1],reverse=True)
            sorted_features_log_prob_class_str = [' '.join([x,str(y)]) for x,y in sorted_features_log_prob_class]
            sorted_features_all_classes.append(sorted_features_log_prob_class_str)
        for i,s in enumerate(sorted_features_all_classes[0]):
            ranked_features.append('\t'.join([sorted_features_all_classes[j][i] for j,x in enumerate(sorted_features_all_classes)]))
        return '\n'.join(ranked_features) + '\n'

    def return_class_count(self):
        class_count = []
        sorted_classes = sorted([[self.label_encoder.inverse_transform(self.model.classes_[i]),i] for i,x in enumerate(self.model.class_count_.tolist())],key = lambda k : k[0])
        for i,c in enumerate(sorted_classes):
            class_count.append('\t'.join([c[0],str(int(self.model.class_count_.tolist()[i]))]))
        return '\n'.join(class_count) + '\n'

    def return_class_log_prior(self):
        class_log_prior = []
        sorted_classes = sorted([[self.label_encoder.inverse_transform(self.model.classes_[i]),i] for i,x in enumerate(self.model.class_log_prior_.tolist())],key = lambda k : k[0])
        for i,c in enumerate(sorted_classes):
            class_log_prior.append('\t'.join([c[0],str(self.model.class_log_prior_.tolist()[i])]))
        return '\n'.join(class_log_prior) + '\n'

    def return_model_insights(self,vocab):
        model_insights = [['feature_count.txt',self.return_feature_count(vocab)],['feature_log_prob.txt',self.return_feature_log_prob(vocab)],['ranked_features.txt',self.return_ranked_features(vocab)],['class_count.txt',self.return_class_count()],['class_log_prior.txt',self.return_class_log_prior()]]
        return model_insights
        

class GaussianNaiveBayesClassifier(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)
    
    def train_classifier(self, trainvectors, labels, no_label_encoding=False, alpha='default', fit_prior=True, jobs=4):
        fit_prior = False if fit_prior == 'False' else True
        jobs = int(jobs)
        if alpha == '':
            paramsearch = GridSearchCV(estimator=naive_bayes.MultinomialNB(), param_grid=dict(alpha=numpy.linspace(0,2,20)[1:]), n_jobs=6)
            paramsearch.fit(trainvectors,self.label_encoder.transform(labels))
            selected_alpha = paramsearch.best_estimator_.alpha
        elif alpha == 'default':
            selected_alpha = 1.0
        else:
            selected_alpha = alpha
        self.model = naive_bayes.MultinomialNB(alpha=selected_alpha,fit_prior=fit_prior)
        self.model.fit(trainvectors, self.label_encoder.transform(labels))

    def return_classifier(self):
        return self.model

    def return_feature_count(self,vocab=False):
        feature_count = []
        feature_count.append('\t'.join([''] + [self.label_encoder.inverse_transform(self.model.classes_[i]) for i in range(len(self.model.classes_))])) # de namen van de klassen
        if vocab:
            for i,vals in enumerate(self.model.feature_count_.T.tolist()):
                feature_count.append('\t'.join([vocab[i]] + [str(x) for x in vals]))
        else:
            for i,vals in enumerate(self.model.feature_count_.T.tolist()):
                feature_count.append('\t'.join([str(i)] + [str(x) for x in vals]))
        return '\n'.join(feature_count) + '\n'

    def return_feature_log_prob(self,vocab=False):
        feature_log_prob = []
        feature_log_prob.append('\t'.join([''] + [self.label_encoder.inverse_transform(self.model.classes_[i]) for i in range(len(self.model.classes_))])) # de namen van de klassen
        if vocab:
            for i,vals in enumerate(self.model.feature_log_prob_.T.tolist()):
                feature_log_prob.append('\t'.join([vocab[i]] + [str(x) for x in vals]))
        else:
            for i,vals in enumerate(self.model.feature_log_prob_.T.tolist()):
                feature_log_prob.append('\t'.join([str(i)] + [str(x) for x in vals]))
        return '\n'.join(feature_log_prob) + '\n'

    def return_ranked_features(self,vocab=False):
        ranked_features = []
        ranked_features.append('\t'.join([' '.join([self.label_encoder.inverse_transform(self.model.classes_[i]),'log prob']) for i in range(len(self.model.classes_))])) # de namen van de klassen
        sorted_classes = sorted([[self.label_encoder.inverse_transform(self.model.classes_[i]),i] for i,x in enumerate(self.model.class_count_.tolist())],key = lambda k : k[0])
        sorted_features_all_classes = []
        for i,c in enumerate(sorted_classes):
            if vocab:
                features_log_prob_class = [[vocab[j],vals[i]] for j,vals in enumerate(self.model.feature_log_prob_.T.tolist())]
            else:
                features_log_prob_class = [[str(j),vals[i]] for j,vals in enumerate(self.model.feature_log_prob_.T.tolist())]
            sorted_features_log_prob_class = sorted(features_log_prob_class,key = lambda k : k[1],reverse=True)
            sorted_features_log_prob_class_str = [' '.join([x,str(y)]) for x,y in sorted_features_log_prob_class]
            sorted_features_all_classes.append(sorted_features_log_prob_class_str)
        for i,s in enumerate(sorted_features_all_classes[0]):
            ranked_features.append('\t'.join([sorted_features_all_classes[j][i] for j,x in enumerate(sorted_features_all_classes)]))
        return '\n'.join(ranked_features) + '\n'

    def return_class_count(self):
        class_count = []
        sorted_classes = sorted([[self.label_encoder.inverse_transform(self.model.classes_[i]),i] for i,x in enumerate(self.model.class_count_.tolist())],key = lambda k : k[0])
        for i,c in enumerate(sorted_classes):
            class_count.append('\t'.join([c[0],str(int(self.model.class_count_.tolist()[i]))]))
        return '\n'.join(class_count) + '\n'

    def return_class_prior(self):
        class_log_prior = []
        sorted_classes = sorted([[self.label_encoder.inverse_transform(self.model.classes_[i]),i] for i,x in enumerate(self.model.class_log_prior_.tolist())],key = lambda k : k[0])
        for i,c in enumerate(sorted_classes):
            class_log_prior.append('\t'.join([c[0],str(self.model.class_log_prior_.tolist()[i])]))
        return '\n'.join(class_log_prior) + '\n'

    def return_model_insights(self,vocab):
        model_insights = [['feature_count.txt',self.return_feature_count(vocab)],['feature_log_prob.txt',self.return_feature_log_prob(vocab)],['ranked_features.txt',self.return_ranked_features(vocab)],['class_count.txt',self.return_class_count()],['class_log_prior.txt',self.return_class_log_prior()]]
        return model_insights

    def apply_classifier(self, testvectors):
        classifications = AbstractSKLearnClassifier.apply_model(self, self.model, testvectors)
        return classifications




class LinearRegressionClassifier(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)


    # def set_label_encoder(self, raw_labels,labels):
    #     self.label_encoder = {}
    #     for label in sorted(list(set(labels))):
    #         raw_values = [raw_labels[index] for index, lab in enumerate(labels) if lab == label]
    #         self.label_encoder[label] = [min(raw_values),max(raw_values)]
                
    # def return_label_encoder(self):
    #     return self.label_encoder
    
    # def transform_labels(self,labels):
    #     transformed_labels = []
    #     for label in labels:
    #         for target_label in self.label_encoder.keys()::
    #             if label > self.label_encoder[target_label][0] and label < self.label_encoder[target_label][1]:
    #                 transformed_labels.append(target_label)
    #                 break
    #     return transformed_labels

    def train_classifier(self, trainvectors, labels, fit_intercept='True', normalize='False', copy_X='True', jobs=4):
        fit_intercept = False if fit_intercept == 'False' else True
        normalize = False if normalize == 'False' else True
        copy_X = False if copy_X == 'False' else True
        jobs = int(jobs)
        self.model = LinearRegression(fit_intercept=fit_intercept,normalize=normalize,copy_X=copy_X,n_jobs=jobs)
        if no_label_encoding:
            self.model.fit(trainvectors, labels)
        else:
            self.model.fit(trainvectors, self.label_encoder.transform(labels))

    def return_classifier(self):
        return self.model

    def return_coef(self,vocab=False):
        sorted_classes = sorted([[self.label_encoder.inverse_transform(self.model.classes_[i]),i] for i,x in enumerate(self.model.class_count_.tolist())],key = lambda k : k[0])
        coef = []
        features_all_targets = []
        for i,c in enumerate(sorted_classes):
            if vocab:
                features_target = [[vocab[j],vals[i]] for j,vals in enumerate(self.model.coef_.T.tolist())]
            else:
                features_target = [[str(j),vals[i]] for j,vals in enumerate(self.model.coef_.T.tolist())]
            features_target_str = [' '.join([x,str(y)]) for x,y in features_target]
            features_all_targets.append(sorted_features_log_prob_class_str)
        for i,s in enumerate(features_all_targets):
            coef.append('\t'.join([features_all_targets[j][i] for j,x in enumerate(features_all_targets)]))
        return '\n'.join(coef) + '\n'

    def return_model_insights(self,vocab):
        model_insights = []
        # model_insights = [['coef.txt',self.return_coef(vocab)]]
        return model_insights
        
        # return [['feature_log_prob.txt','\n'.join([str(x) for x in self.model.feature_log_prob_.T.tolist()])],['class_log_prior.txt','\n'.join(['\t'.join([self.label_encoder.inverse_transform(self.model.classes_[i]),str(x)]) for i,x in enumerate(self.model.class_log_prior_.tolist())])],['class_count.txt','\n'.join(['\t'.join([self.label_encoder.inverse_transform(self.model.classes_[i]),str(x)]) for i,x in enumerate(self.model.class_count_.tolist())])]]

    # def apply_classifier(self, testvectors):
    #     predictions_raw = []
    #     predictions = []
    #     full_predictions = [sorted(self.label_encoder.keys())]
    #     for i, instance in enumerate(testvectors):
    #         prediction_raw = self.model.predict(instance)[0]
    #         predictions_raw.append(prediction_raw)
    #         prediction = self.transform_labels([prediction_raw])[0]
    #         predictions.append(prediction)
    #         try:
    #             full_predictions.append([clf.score(numpy.array([instance]),numpy.array(self.transform_labels([c])[0])) for c in self.transform_labels(full_predictions[0])])
    #         except:
    #             full_predictions.append(['-' for c in full_predictions[0]])
    #     return predictions_raw, predictions, full_predictions




class SVMClassifier(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)

    def train_classifier(self, trainvectors, labels, no_label_encoding=False, c='', kernel='', gamma='', degree='', class_weight='', iterations=10, jobs=4):
        jobs = int(jobs)
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
            if class_weight == '':
                class_weight = 'balanced'
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
           class_weight = class_weight,
           cache_size = 1000,
           verbose = 2
           )
        self.model.fit(trainvectors, self.label_encoder.transform(labels))

    def return_classifier(self):
        return self.model

    def return_support_vectors(self):
        return '\n'.join(self.model.support_vectors_.T.tolist())

    def return_feature_weights(self):
        return '\n'.join([' '.join(l) for l in self.model.coef_.T.tolist()])

    def return_model_insights(self,vocab=False):
        model_insights = []
        try:
            model_insights.append(['support_vectors.txt',self.return_support_vectors()])
        except:
            print('could not return support vectors')
        try:
            model_insights.append(['feature_weights',self.return_feature_weights()])
        except:
            print('could not return feature weights')
        return model_insights

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

    def train_classifier(self, trainvectors, labels, no_label_encoding=False, c='', solver='', dual='', penalty='', multiclass='', max_iterations=1000, iterations=10, jobs=4):
        jobs = int(jobs)
        parameters = ['C', 'solver', 'penalty', 'dual', 'multi_class']
        c_values = [0.001, 0.005, 0.01, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0] if c == '' else [float(x) for x in c.split()]
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
        elif dual == 'False':
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
            paramsearch = RandomizedSearchCV(model, param_grid, cv = 5, verbose = 2, n_iter = iterations, n_jobs = jobs, pre_dispatch = 4)
            paramsearch.fit(trainvectors.toarray(), self.label_encoder.transform(labels))
            settings = paramsearch.best_params_
        # train a logistic regression classifier with the settings that led to the best performance
        self.model = LogisticRegression(
            C = settings[parameters[0]],
            solver = settings[parameters[1]],
            penalty = settings[parameters[2]],
            dual = settings[parameters[3]],
            multi_class = settings[parameters[4]],
            max_iter = max_iterations,
            verbose = 2,
            n_jobs = 10
        )
        self.model.fit(trainvectors, self.label_encoder.transform(labels))

    def return_classifier(self):
        return self.model

    def return_feature_coef(self,vocab=False):
        feature_coef = []
        feature_coef.append('\t'.join([''] + [self.label_encoder.inverse_transform(self.model.classes_[i]) for i in range(len(self.model.classes_))])) # de names of the classes
        if vocab:
            for i,vals in enumerate(self.model.coef_.T.tolist()):
                feature_coef.append('\t'.join([vocab[i]] + [str(x) for x in vals]))
        else:
            for i,vals in enumerate(self.model.coef_.T.tolist()):
                feature_log_prob.append('\t'.join([str(i)] + [str(x) for x in vals]))
        return '\n'.join(feature_coef) + '\n'
    
    def return_model_insights(self,vocab=False):
        model_insights = [['coef.txt',self.return_feature_coef(vocab)]]
        return model_insights
        
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
    
    def train_classifier(self, trainvectors, labels, no_label_encoding=False, class_weight=None):
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

    def train_classifier(self, trainvectors, labels, no_label_encoding=False, alpha='', iterations=50, jobs=10):
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

    def return_model_insights(self,vocab):
        model_insights = [['coef.txt',self.return_coef(vocab)]]
        return model_insights
    
class KNNClassifier(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)

    def train_classifier(self, trainvectors, labels, no_label_encoding=False, n_neighbors=3, weights='uniform', algorithm='auto', jobs=8):
        jobs = int(jobs)
        if n_neighbors == 'default' or n_neighbors == '':
            n_neighbors = 3
        if weights == 'default' or weights == '':
            weights = 'uniform'
        if algorithm == 'default' or algorithm == '':
            algorithm = 'auto'
        # train
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,n_jobs=jobs)
        self.model.fit(trainvectors, self.label_encoder.transform(labels))

    def return_classifier(self):
        return self.model

    def apply_classifier(self, testvectors):
        classifications = AbstractSKLearnClassifier.apply_model(self, self.model, testvectors)
        return classifications

    def return_model_insights(self,vocab):
        model_insights = [['coef.txt',self.return_coef(vocab)]]
        return model_insights

