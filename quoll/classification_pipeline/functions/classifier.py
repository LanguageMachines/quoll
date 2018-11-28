
from sklearn import preprocessing
from sklearn import svm, naive_bayes, tree
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import OutputCodeClassifier
import warnings

import numpy
import multiprocessing

class AbstractSKLearnClassifier:

    def __init__(self):
        self.label_encoder = preprocessing.LabelEncoder()
        warnings.filterwarnings(action='ignore', category=DeprecationWarning) # to ignore an annoying warning message during label transforming
        
    def set_label_encoder(self, labels):
        self.label_encoder.fit(labels)

    def return_label_encoder(self):
        return self.label_encoder

    def return_label_encoding(self, labels):
        encoding = [str(x) for x in self.label_encoder.transform(sorted(list(set(labels))))]
        label_encoding = list(zip(labels,encoding))
        return label_encoding

    def predict(self,clf,testvector):
        try:
            prediction = clf.predict(testvector)[0]
            full_prediction = [clf.predict_proba(testvector)[0][c] for c in self.label_encoder.transform(sorted(list(self.label_encoder.classes_)))]
        except ValueError: # classifier trained on dense data
            prediction = self.label_encoder.inverse_transform([clf.predict(testvector.todense())[0]])[0]
            full_prediction = [clf.predict_proba(testvector.toarray())[0][c] for c in self.label_encoder.transform(sorted(list(self.label_encoder.classes_)))]
        except AttributeError: # classifier does not support predict_proba
            try:
                prediction = self.label_encoder.inverse_transform([clf.predict(testvector.todense())[0]])[0]
                full_prediction = [1.0 for c in self.label_encoder.classes_]
            except: # no label encoding
                prediction = clf.predict(testvector)[0]
                full_prediction = ['-']
        return prediction, full_prediction

    def apply_model(self,clf,testvectors):
        predictions = []
        try:
            full_predictions = [list(self.label_encoder.classes_)]
        except:
            full_predictions = ['-']
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
    
    def train_classifier(self, trainvectors, labels, alpha='1.0', fit_prior=False, jobs=1, v=2):
        if alpha == 'search':
            paramsearch = GridSearchCV(estimator=naive_bayes.MultinomialNB(), param_grid=dict(alpha=numpy.linspace(0,2,20)[1:]), n_jobs=jobs)
            paramsearch.fit(trainvectors,labels)
            selected_alpha = paramsearch.best_estimator_.alpha
        else:
            selected_alpha = float(alpha)
        self.model = naive_bayes.MultinomialNB(alpha=selected_alpha,fit_prior=fit_prior)


        self.model.fit(trainvectors, labels)

    def return_classifier(self):
        return self.model

    def return_class_names(self):
        return self.label_encoder.inverse_transform(self.model.classes_)
        
    def return_feature_count(self,vocab=False):
        feature_count = ['\t'.join(self.return_class_names())]
        if vocab:
            l=vocab
        else:
            l=list(range(len(self.model.feature_count_.T.tolist())))
        feature_count.extend(['\t'.join([str(l[i])] + [str(x) for x in vals]) for i,vals in enumerate(self.model.feature_count_.T.tolist())])
        return '\n'.join(feature_count) + '\n'

    def return_feature_log_prob(self,vocab=False):
        feature_log_prob = ['\t'.join(self.return_class_names())]
        if vocab:
            l=vocab
        else:
            l=list(range(len(self.model.feature_log_prob_.T.tolist())))
        feature_log_prob.extend(['\t'.join([str(l[i])] + [str(x) for x in vals]) for i,vals in enumerate(self.model.feature_log_prob_.T.tolist())])
        return '\n'.join(feature_log_prob) + '\n'

    def return_ranked_features(self,class_index,vocab=False):
        if vocab:
            l=vocab
        else:
            l=list(range(len(self.model.feature_log_prob_.T.tolist())))
        sorted_features_class = [' '.join([str(x),str(y)]) for x,y in sorted([[l[j],vals[class_index]] for j,vals in enumerate(self.model.feature_log_prob_.T.tolist())],key=lambda k : k[1],reverse=True)]
        return '\n'.join(sorted_features_class) + '\n'

    def return_class_count(self):
        sorted_classes = sorted(self.return_class_names(),key=lambda k : k[0])
        class_count = ['\t'.join([c[0],str(self.model.class_count_.tolist()[i])]) for i,c in enumerate(sorted_classes)]
        return '\n'.join(class_count) + '\n'

    def return_class_log_prior(self):
        sorted_classes = sorted(self.return_class_names(),key=lambda k : k[0])
        class_log_prior = ['\t'.join([c[0],str(self.model.class_log_prior_.tolist()[i])]) for i,c in enumerate(sorted_classes)]
        return '\n'.join(class_log_prior) + '\n'

    def return_parameter_settings(self):
        parameter_settings = []
        for param in ['alpha','fit_prior']:
            parameter_settings.append([param,str(self.model.get_params()[param])])
        return '\n'.join([': '.join(x) for x in parameter_settings])

    def return_model_insights(self,vocab):
        model_insights = [['parameter_settings.txt',self.return_parameter_settings()],['feature_count.txt',self.return_feature_count(vocab)],['feature_log_prob.txt',self.return_feature_log_prob(vocab)],['class_count.txt',self.return_class_count()],['class_log_prior.txt',self.return_class_log_prior()]]
        sorted_classes = sorted(self.return_class_names(),key=lambda k : k[0])
        for i,cl in enumerate(sorted_classes):
            model_insights.append(['ranked_features_' + cl + '.txt',self.return_ranked_features(i,vocab)])
        return model_insights
        

class SVMClassifier(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)

    def train_classifier(self, trainvectors, labels, c='1.0', kernel='linear', gamma='0.1', degree='1', class_weight='balanced', iterations=10, jobs=1, v=2):
        if len(self.label_encoder.classes_) > 2: # more than two classes to distinguish
            parameters = ['estimator__C', 'estimator__kernel', 'estimator__gamma', 'estimator__degree']
            multi = True
        else: # only two classes to distinguish
            parameters = ['C', 'kernel', 'gamma', 'degree']
            multi = False
        if len(class_weight.split(':')) > 1: # dictionary
            class_weight = dict([label_weight.split(':') for label_weight in class_weight.split()])
        c_values = [0.001, 0.005, 0.01, 0.5, 1, 5, 10, 50, 100, 500, 1000] if c == 'search' else [float(x) for x in c.split()]
        kernel_values = ['linear', 'rbf', 'poly'] if kernel == 'search' else [k for  k in kernel.split()]
        gamma_values = [0.0005, 0.002, 0.008, 0.032, 0.128, 0.512, 1.024, 2.048] if gamma == 'search' else [float(x) for x in gamma.split()]
        degree_values = [1, 2, 3, 4] if degree == 'search' else [int(x) for x in degree.split()]
        grid_values = [c_values, kernel_values, gamma_values, degree_values]
        if not False in [len(x) == 1 for x in grid_values]: # only sinle parameter settings
            settings = {}
            for i, parameter in enumerate(parameters):
                settings[parameter] = grid_values[i][0]
        else:
            param_grid = {}
            for i, parameter in enumerate(parameters):
                param_grid[parameter] = grid_values[i]
            model = svm.SVC(probability=True)
            if multi:
                model = OutputCodeClassifier(model)
                trainvectors = trainvectors.todense()
            paramsearch = RandomizedSearchCV(model, param_grid, cv = 5, verbose = v, n_iter = iterations, n_jobs = jobs, pre_dispatch = 4)
            paramsearch.fit(trainvectors, labels)
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
            verbose = v
        )
        self.model.fit(trainvectors, labels)

    def return_classifier(self):
        return self.model

    def return_class_names(self):
        return self.label_encoder.inverse_transform(self.model.classes_)

    def return_feature_weights(self,vocab=False):
        if self.model.get_params()['kernel'] == 'linear':
            if vocab:
                l=vocab
            else:
                l=list(range(self.model.coef_.shape[1]))
            features_weights=[]
            for i in range(len(l)):
                feature_weights=[l[i]]
                for j in range(self.model.coef_.shape[0]):
                    feature_weights.append(str(self.model.coef_[j,i]))
                features_weights.append(' '.join(feature_weights))
            return '\n'.join(features_weights)
        else:
            return 'Feature weights can only be requested in case of a linear kernel, not for a ' + str(self.model.get_params()['kernel']) + ' kernel.'

    def return_parameter_settings(self):
        parameter_settings = []
        for param in ['C','kernel','gamma','degree']:
            parameter_settings.append([param,str(self.model.get_params()[param])])
        return '\n'.join([': '.join(x) for x in parameter_settings])

    def return_model_insights(self,vocab=False):
#        model_insights = [['feature_weights.txt',self.return_feature_weights(vocab)],['parameter_settings.txt',self.return_parameter_settings()]]
        model_insights = [['parameter_settings.txt',self.return_parameter_settings()]]
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

    def train_classifier(self, trainvectors, labels, c='1.0', solver='liblinear', dual=False, penalty='l2', multiclass='ovr', max_iterations='1000', iterations=10, jobs=4, v=2):
        parameters = ['C', 'solver', 'penalty', 'dual', 'multi_class']
        c_values = [0.001, 0.005, 0.01, 0.5, 1, 5, 10, 50, 100, 500, 1000] if c == 'search' else [float(x) for x in c.split()]
        solver_values = ['newton-cg', 'lbfgs', 'liblinear', 'sag'] if solver == 'search' else [s for  s in solver.split()]
        if penalty == 'search':
            if not set(['newton-cg','lbfgs','sag']) & set(solver_values):
                penalty_values = ['l1', 'l2']
            else:
                penalty_values = ['l2']
        else:
            if (set(['newton-cg','lbfgs','sag']) & set(solver_values)) and 'l1' in penalty:
                print('L1 penalty does not fit with solver, fixing l2')
                penalty_values = ['l2']
            else:
                penalty_values = [penalty]
        if dual:
            if dual == 'search':
                if len(solver_values) == 1 and solver_values[0] == 'liblinear':
                    if len(penalty_values) == 1 and penalty_values[0] == 'l2':
                        dual_values = [True,False]
                else:
                    dual_values = [False]
            else:
                dual_values = [int(dual)] # 1 or 0
        else: # dual is False
            dual_values = [False]
        if multiclass == 'search':
            if 'liblinear' not in solver_values:
                multiclass_values = ['ovr', 'multinomial']
            else:
                multiclass_values = ['ovr']
        else:
            if 'liblinear' not in solver_values:
                multiclass_values = [multiclass]
            else:
                if 'multinomial' in multiclass:
                    print('Multinomial is not an option when using liblinear solver, switching to \'ovr\' setting for \'multiclass\'')
                    multiclass_values = ['ovr']
                else:
                    multiclass_values = [multiclass]
        grid_values = [c_values, solver_values, penalty_values, dual_values, multiclass_values]
        max_iterations = int(max_iterations)
        if not False in [len(x) == 1 for x in grid_values]: # only single parameter settings
            settings = {}
            for i, parameter in enumerate(parameters):
                settings[parameter] = grid_values[i][0]
        else: # try different parameter combinations
            iterations=int(iterations)
            combis = len(c_values) * len(solver_values) * len(penalty_values) * len(dual_values) * len(multiclass_values)
            if combis < iterations:
                iterations=combis
            param_grid = {}
            for i, parameter in enumerate(parameters):
                param_grid[parameter] = grid_values[i]
            model = LogisticRegression(max_iter=max_iterations)
            paramsearch = RandomizedSearchCV(model, param_grid, cv = 5, verbose = v, n_iter = iterations, n_jobs = jobs, pre_dispatch = 4)
            paramsearch.fit(trainvectors.toarray(), labels)
            settings = paramsearch.best_params_
        # train a logistic regression classifier with the settings that led to the best performance
        self.model = LogisticRegression(
            C = settings[parameters[0]],
            solver = settings[parameters[1]],
            penalty = settings[parameters[2]],
            dual = settings[parameters[3]],
            multi_class = settings[parameters[4]],
            max_iter = max_iterations,
            verbose = v,
            n_jobs = jobs
        )
        self.model.fit(trainvectors, labels)

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
#        model_insights = [['coef.txt',self.return_feature_coef(vocab)]]
        model_insights = []
        return model_insights
        
    def apply_classifier(self, testvectors):
        classifications = AbstractSKLearnClassifier.apply_model(self, self.model, testvectors)
        return classifications


class XGBoostClassifier(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)

    def train_classifier(self, trainvectors, labels, 
        booster='gbtree', silent='1', nthread='12', 
        learning_rate='0.1', min_child_weight='1', max_depth='6', gamma='0', max_delta_step='0', 
        subsample='1', colsample_bytree='1', reg_lambda='1', reg_alpha='0', scale_pos_weight='1',
        objective='binary:logistic', seed=7, n_estimators='100',
        scoring='roc_auc', jobs=12, v=2):
        # prepare grid search
        if len(self.label_encoder.classes_) > 2: # more than two classes to distinguish
            parameters = ['estimator__n_estimators','estimator__min_child_weight', 'estimator__max_depth', 'estimator__gamma', 'estimator__subsample','estimator__colsample_bytree','estimator__reg_alpha','estimator__scale_pos_weight']
            multi = True
        else: # only two classes to distinguish
            parameters = ['n_estimators','min_child_weight', 'max_depth', 'gamma', 'subsample','colsample_bytree','reg_alpha', 'scale_pos_weight'] 
            multi = False
        silent = int(silent)
        nthread=int(nthread)
        learning_rate = float(learning_rate)
        max_delta_step = float(max_delta_step)
        reg_lambda = float(reg_lambda)
        n_estimators_values = list(range(100,1000,100)) if n_estimators == 'search' else [int(x) for x in n_estimators.split()]
        min_child_weight_values = list(range(1,6,1)) if min_child_weight == 'search' else [int(x) for x in min_child_weight.split()]
        max_depth_values = list(range(3,10,1)) if max_depth == 'search' else [int(x) for x in max_depth.split()]
        gamma_values = [i/10 for i in range(0,5)] if gamma == 'search' else [float(x) for x in gamma.split()]
        subsample_values = [i/10 for i in range(6,10)] if subsample == 'search' else [float(x) for x in subsample.split()]
        colsample_bytree_values = [i/10 for i in range(6,10)] if colsample_bytree == 'search' else [float(x) for x in colsample_bytree.split()]
        reg_alpha_values = [1e-5,1e-2,0.1,1,100] if reg_alpha == 'search' else [float(x) for x in reg_alpha.split()]
        scale_pos_weight_values = [1,3,5,7,9] if scale_pos_weight == 'search' else [int(x) for x in scale_pos_weight.split()]
        grid_values = [n_estimators_values,min_child_weight_values, max_depth_values, gamma_values, subsample_values, colsample_bytree_values, reg_alpha_values, scale_pos_weight_values]
        if not False in [len(x) == 1 for x in grid_values]: # only sinle parameter settings
            settings = {}
            for i, parameter in enumerate(parameters):
                settings[parameter] = grid_values[i][0]
        else:
            param_grid = {}
            for i, parameter in enumerate(parameters):
                param_grid[parameter] = grid_values[i]
            model = XGBClassifier(silent=silent,nthread=nthread,learning_rate=learning_rate,max_delta_step=max_delta_step,reg_lambda=reg_lambda,scale_pos_weight=scale_pos_weight)
            if multi:
                model = OutputCodeClassifier(model)
                trainvectors = trainvectors.todense()
            if [len(x) > 1 for x in grid_values].count(True) <= 3: # exhaustive grid search with one to three variant parameters
                paramsearch = GridSearchCV(model, param_grid, verbose=v, scoring=scoring, cv=5, n_jobs=1)
            else: # random grid search
                paramsearch = RandomizedSearchCV(model, param_grid, verbose=v, scoring=scoring, cv=5, n_jobs=1)
            paramsearch.fit(trainvectors, labels)
            settings = paramsearch.best_params_
        self.model = XGBClassifier(
            learning_rate = learning_rate, 
            max_delta_step = max_delta_step, 
            reg_lambda = reg_lambda, 
            silent = silent,
            nthread = nthread,
            n_estimators = settings[parameters[0]], 
            min_child_weight = settings[parameters[1]], 
            max_depth = settings[parameters[2]],
            gamma = settings[parameters[3]],
            subsample = settings[parameters[4]],
            colsample_bytree = settings[parameters[5]],
            reg_alpha = settings[parameters[6]],
            scale_pos_weight = settings[parameters[7]],
            verbose = v
        )
        self.model.fit(trainvectors, labels)

    def return_classifier(self):
        return self.model

    def return_feature_importance(self,vocab=False):
        feature_importance = []
        if vocab:
            for i,val in enumerate(self.model.feature_importances_.T.tolist()):
                feature_importance.append([vocab[i],val])
        else:
            for i,val in enumerate(self.model.coef_.T.tolist()):
                feature_importance.append([str(i),val])
        sorted_feature_importance = sorted(feature_importance,key = lambda k : k[1],reverse=True)
        sorted_feature_importance_str = '\n'.join(['\t'.join([str(x) for x in row]) for row in sorted_feature_importance])
        return sorted_feature_importance_str
    
    def return_parameter_settings(self):
        parameter_settings = []
        for param in ['n_estimators','min_child_weight', 'max_depth', 'gamma', 'subsample','colsample_bytree',
            'reg_alpha', 'scale_pos_weight','learning_rate','max_delta_step','reg_lambda']:
            parameter_settings.append([param,str(self.model.get_params()[param])])
        return '\n'.join([': '.join(x) for x in parameter_settings])

    def return_model_insights(self,vocab=False):
        model_insights = [['feature_importance.txt',self.return_feature_importance(vocab)],['parameter_settings.txt',self.return_parameter_settings()]]
        return model_insights

class KNNClassifier(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)

    def train_classifier(self, trainvectors, labels, n_neighbors='3', weights='uniform', algorithm='auto', leaf_size='30', metric='euclidean', p=2, scoring='roc_auc', jobs=1, v=2):
        if len(self.label_encoder.classes_) > 2: # more than two classes to distinguish
            parameters = ['estimator__n_neighbors','estimator__weights', 'estimator__leaf_size', 'estimator__metric']
            multi = True
        else: # only two classes to distinguish
            parameters = ['n_neighbors','weights', 'leaf_size', 'metric'] 
            multi = False
        n_neighbours = [3,5,7,9] if n_neighbors == 'search' else [int(x) for x in n_neighbors.split()]
        weights = ['uniform','distance'] if weights == 'search' else weights.split()
        leaf_size = [10,20,30,40,50] if n_neighbors == 'search' else [int(x) for x in leaf_size.split()]
        metric = ['minkowski','euclidean','manhattan','hamming'] if metric == 'search' else metric.split()
        grid_values = [n_neighbors, weights, leaf_size, metric]
        if not False in [len(x) == 1 for x in grid_values]: # only sinle parameter settings
            settings = {}
            for i, parameter in enumerate(parameters):
                settings[parameter] = grid_values[i][0]
        else:
            param_grid = {}
            for i, parameter in enumerate(parameters):
                param_grid[parameter] = grid_values[i]
            model = KNeighborsClassifier(algorithm=algorithm,p=p) 
            if multi:
                model = OutputCodeClassifier(model)
                trainvectors = trainvectors.todense()
            paramsearch = RandomizedSearchCV(model, param_grid, verbose=v, scoring=scoring, cv=5, n_jobs=jobs)
            paramsearch.fit(trainvectors, labels)
            settings = paramsearch.best_params_
        self.model = KNeighborsClassifier(
            algorithm=algorithm,
            p=p,
            n_neighbors=settings[parameters[0]],
            weights=settings[parameters[1]],
            leaf_size=settings[parameters[2]],
            metric=settings[parameters[3]],
            verbose=v
        )
        self.model.fit(trainvectors, labels)

    def return_classifier(self):
        return self.model
    
    def return_model_insights(self,vocab):
        model_insights = []
        return model_insights

class RandomForestClassifier(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)

    def train_classifier(self, trainvectors, labels, n_neighbors='3', weights='uniform', algorithm='auto', jobs=8, v=2):
        jobs = int(jobs)
        if n_neighbors == 'default' or n_neighbors == '':
            n_neighbors = 3
        if weights == 'default' or weights == '':
            weights = 'uniform'
        if algorithm == 'default' or algorithm == '':
            algorithm = 'auto'
        # train
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,n_jobs=jobs)
        self.model.fit(trainvectors, labels)

    def return_classifier(self):
        return self.model

    def apply_classifier(self, testvectors):
        classifications = AbstractSKLearnClassifier.apply_model(self, self.model, testvectors)
        return classifications

    def return_model_insights(self,vocab):
        model_insights = [['coef.txt',self.return_coef(vocab)]]
        return model_insights


class LinearRegressionClassifier(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)

    def train_classifier(self, trainvectors, labels, fit_intercept=True, normalize=False, copy_X=True, jobs=4, v=2):
        fit_intercept = False if fit_intercept == '0' else True
        normalize = False if normalize == '0' else True
        copy_X = False if copy_X == '0' else True
        jobs = int(jobs)
        self.model = LinearRegression(fit_intercept=fit_intercept,normalize=normalize,copy_X=copy_X,n_jobs=jobs)
        self.model.fit(trainvectors, labels)

    def return_classifier(self):
        return self.model

    def return_model_insights(self,vocab):
        model_insights = []
        return model_insights
        
class TreeClassifier(AbstractSKLearnClassifier):

    def __init__(self):
        AbstractSKLearnClassifier.__init__(self)
        self.model = False

    def set_label_encoder(self, labels):
        AbstractSKLearnClassifier.set_label_encoder(self, labels)

    def return_label_encoding(self, labels):
        return AbstractSKLearnClassifier.return_label_encoding(self, labels)
    
    def train_classifier(self, trainvectors, labels, no_label_encoding=False, class_weight=None, v=2):
        self.model = tree.DecisionTreeClassifier(class_weight=class_weight)
        self.model.fit(trainvectors, labels)

    def return_classifier(self):
        return self.model

    def return_model_insights(self,vocab):
        #model_insights = [['feature_importances_gini.txt','\n'.join([str(x) for x in self.model.feature_importances_.T.tolist()])],['tree.txt',self.model.tree_.__str__()]]
        model_insights = []
        return model_insights
        
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

    def train_classifier(self, trainvectors, labels, no_label_encoding=False, alpha='', iterations=50, jobs=10, v=2):
        iterations = int(iterations)
        jobs = int(jobs)
        if alpha == '':
            paramsearch = GridSearchCV(estimator=Perceptron(), param_grid=dict(alpha=numpy.linspace(0,2,20)[1:],n_iter=[iterations]), n_jobs=jobs)
            paramsearch.fit(trainvectors,labels)
            selected_alpha = paramsearch.best_estimator_.alpha
        elif alpha == 'default':
            selected_alpha = 1.0
        else:
            selected_alpha = alpha
        # train a perceptron with the settings that led to the best performance
        self.model = Perceptron(alpha=selected_alpha,n_iter=iterations,n_jobs=jobs)
        self.model.fit(trainvectors, labels)

    def return_classifier(self):
        return self.model

    def apply_classifier(self, testvectors):
        classifications = AbstractSKLearnClassifier.apply_model(self, self.model, testvectors)
        return classifications

    def return_model_insights(self,vocab):
#        model_insights = [['coef.txt',self.return_coef(vocab)]]
        model_insights = []
        return model_insights
    
