
import numpy
from scipy import sparse
import glob
from collections import defaultdict

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter, FloatParameter

from quoll.classification_pipeline.modules.classify import Classify, VectorizeTrain, VectorizeTrainTest, FitTransformScale, TransformScale 

from quoll.classification_pipeline.functions import reporter, linewriter, docreader

#################################################################
### Tasks #######################################################
#################################################################

class ReportPerformance(Task):

    in_predictions = InputSlot()
    in_testlabels = InputSlot()
    in_testdocuments = InputSlot()

    ordinal = BoolParameter()
    teststart = IntParameter()
    
    def in_full_predictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.full_predictions.txt')

    def out_report(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report')

    def out_performance(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report/performance.csv')

    def out_performance_at_dir(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report/performance_at_dir')

    def out_docpredictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report/docpredictions.csv')

    def out_confusionmatrix(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report/confusion_matrix.csv')

    def out_fps_dir(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report/ranked_fps')

    def out_tps_dir(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report/ranked_tps')

    def out_fns_dir(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report/ranked_fns')

    def out_tns_dir(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.report/ranked_tns')

    def run(self):

        # setup reporter output directory
        self.setup_output_dir(self.out_report().path)
        self.setup_output_dir(self.out_performance_at_dir().path)
        
        # load predictions and full_predictions
        with open(self.in_predictions().path) as infile:
            predictions = infile.read().strip().split('\n')

        with open(self.in_full_predictions().path) as infile:
            lines = [line.split('\t') for line in infile.read().strip().split('\n')]
        label_order = lines[0]
        full_predictions = lines[1:]

        # load testlabels
        with open(self.in_testlabels().path) as infile:
            testlabels = infile.read().strip().split('\n')

        # load documents
        with open(self.in_testdocuments().path,'r',encoding='utf-8') as infile:
            testdocuments = infile.read().strip().split('\n')[self.teststart:]
            
        # initiate reporter
        rp = reporter.Reporter(predictions, full_predictions, label_order, testlabels, self.ordinal, testdocuments)

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
        for label in label_order:
            ranked_fps = rp.return_ranked_fps(label)
            outfile = self.out_fps_dir().path + '/' + label + '.csv'
            lw = linewriter.Linewriter(ranked_fps)
            lw.write_csv(outfile)

        # report fps per label
        self.setup_output_dir(self.out_tps_dir().path)
        for label in label_order:
            ranked_tps = rp.return_ranked_tps(label)
            outfile = self.out_tps_dir().path + '/' + label + '.csv'
            lw = linewriter.Linewriter(ranked_tps)
            lw.write_csv(outfile)

        # report fns per label
        self.setup_output_dir(self.out_fns_dir().path)
        for label in label_order:
            ranked_fns = rp.return_ranked_fns(label)
            outfile = self.out_fns_dir().path + '/' + label + '.csv'
            lw = linewriter.Linewriter(ranked_fns)
            lw.write_csv(outfile)

        # report tns per label
        self.setup_output_dir(self.out_tns_dir().path)
        for label in label_order:
            ranked_tns = rp.return_ranked_tns(label)
            outfile = self.out_tns_dir().path + '/' + label + '.csv'
            lw = linewriter.Linewriter(ranked_tns)
            lw.write_csv(outfile)

        # report confusion matrix
        if self.ordinal: # to make a confusion matrix, the labels should be formatted as string
            rp = reporter.Reporter(predictions, full_predictions, label_order, testlabels, False, testdocuments)
        confusion_matrix = rp.return_confusion_matrix()
        with open(self.out_confusionmatrix().path,'w') as cm_out:
            cm_out.write(confusion_matrix)

        # report performance-at
        if len(label_order) >= 9:
            prat_opts = [3,5,7,9]
        elif len(label_order) >= 7:
            prat_opts = [3,5,7]
        elif len(label_order) >= 5:
            prat_opts = [3,5]
        elif len(label_order) >= 3:
            prat_opts = [3]
        else:
            prat_opts = []
        for po in prat_opts:
            outfile = self.out_performance_at_dir().path + '/performance_at_' + str(po) + '.txt'
            rp = reporter.Reporter(predictions, full_predictions, label_order, testlabels, self.ordinal, testdocuments, po)
            if self.ordinal:
                performance = rp.assess_ordinal_performance()
            else:
                performance = rp.assess_performance()
            lw = linewriter.Linewriter(performance)
            lw.write_csv(outfile)

            
class ReportDocpredictions(Task):

    in_predictions = InputSlot()
    in_testdocuments = InputSlot()

    def in_full_predictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.full_predictions.txt')

    def out_docpredictions(self):
        return self.outputfrominput(inputformat='predictions', stripextension='.predictions.txt', addextension='.docpredictions.csv')

    def run(self):

        # load predictions and full_predictions
        with open(self.in_predictions().path) as infile:
            predictions = infile.read().strip().split('\n')

        with open(self.in_full_predictions().path) as infile:
            lines = [line.split('\t') for line in infile.read().strip().split('\n')]
        label_order = lines[0]
        full_predictions = lines[1:]
        
        # load documents
        with open(self.in_testdocuments().path,'r',encoding='utf-8') as infile:
            testdocuments = infile.read().strip().split('\n')

        # initiate reporter
        rp = reporter.Reporter(predictions, full_predictions, label_order, documents=testdocuments)

        # report predictions by document
        predictions_by_document = rp.predictions_by_document()
        lw = linewriter.Linewriter(predictions_by_document)
        lw.write_csv(self.out_docpredictions().path)


class ReportFolds(Task):

    in_exp = InputSlot()
    
    def out_predictions(self):
        return self.outputfrominput(inputformat='exp', stripextension='.exp', addextension='.validated.predictions.txt')  

    def out_labels(self):
        return self.outputfrominput(inputformat='exp', stripextension='.exp', addextension='.validated.labels')  

    def out_docs(self):
        return self.outputfrominput(inputformat='exp', stripextension='.exp', addextension='.validated.docs.txt')  

    def out_full_predictions(self):
        return self.outputfrominput(inputformat='exp', stripextension='.exp', addextension='.validated.full_predictions.txt')        

    def out_macro_performance(self):
        return self.outputfrominput(inputformat='exp', stripextension='.exp', addextension='.macro_performance.csv')  
 
    def run(self):
       
        # gather fold reports
        print('gathering fold reports')
        performance_files = [ filename for filename in glob.glob(self.in_exp().path + '/fold*/*.report/performance.csv') ]
        docprediction_files = [ filename for filename in glob.glob(self.in_exp().path + '/fold*/*.report/docpredictions.csv') ]
        
        # calculate average performance
        dr = docreader.Docreader()
        performance_combined = [dr.parse_csv(performance_file) for performance_file in performance_files]
        all_performance = [performance_combined[0][0]] # headers
        label_performance = defaultdict(list)
        for p in performance_combined:
            for i in range(1,len(p)): # labels  
                no_float = []
                performance = []
                label = p[i][0] # name of label
                for j in range(1,len(p[i])): # report values
                    if j not in no_float:
                        try:
                            performance.append(float(p[i][j]))
                        except:
                            no_float.append(j)
                            performance.append('nan')
                            for lp in label_performance[label]:
                                lp[j] = 'nan'
                    else:
                        performance.append('nan')
                label_performance[label].append(performance)

        # compute mean and sum per label
        if 'micro' in label_performance.keys():
            labels_order = [label for label in label_performance.keys() if label != 'micro'] + ['micro']
        else:
            labels_order = sorted(label_performance.keys())

        for label in labels_order:
            average_performance = [label]
            for j in range(0,len(label_performance[label][0])-3):
                if label_performance[label][0][j] != 'nan':
                    average_performance.append(str(round(numpy.mean([float(p[j]) for p in label_performance[label]]),2)) + '(' + str(round(numpy.std([float(p[j]) for p in label_performance[label]]),2)) + ')')
                else:
                    average_performance.append('nan')
            for j in range(len(label_performance[label][0])-3,len(label_performance[label][0])):
                average_performance.append(str(sum([int(p[j]) for p in label_performance[label]])))
            all_performance.append(average_performance)

        lw = linewriter.Linewriter(all_performance)
        lw.write_csv(self.out_macro_performance().path)

        # write labels, predictions and full predictions
        label_order = [x.split('prediction prob for ')[1] for x in dr.parse_csv(docprediction_files[0])[0][3:]]
        docpredictions = sum([dr.parse_csv(docprediction_file)[1:] for docprediction_file in docprediction_files], [])
        docs = [line[0] for line in docpredictions]
        labels = [line[1] for line in docpredictions]
        predictions = [line[2] for line in docpredictions]
        full_predictions = [label_order] + [line[3:] for line in docpredictions]

        with open(self.out_docs().path,'w',encoding='utf-8') as l_out:
            l_out.write('\n'.join(docs))

        with open(self.out_labels().path,'w',encoding='utf-8') as l_out:
            l_out.write('\n'.join(labels))

        with open(self.out_predictions().path,'w',encoding='utf-8') as pr_out:
            pr_out.write('\n'.join(predictions))

        with open(self.out_full_predictions().path,'w',encoding='utf-8') as fpr_out:
            fpr_out.write('\n'.join(['\t'.join([str(prob) for prob in full_prediction]) for full_prediction in full_predictions]))


class ClassifyTask(Task):

    in_trainvectors = InputSlot()
    in_testvectors = InputSlot()
    in_trainlabels = InputSlot()

    ga = BoolParameter()
    num_iterations = IntParameter()
    population_size = IntParameter()
    elite = Parameter()
    crossover_probability = Parameter()
    mutation_rate = Parameter()
    tournament_size = IntParameter()
    n_crossovers = IntParameter()
    stop_condition = IntParameter()
    weight_feature_size = Parameter()
    instance_steps = IntParameter()

    classifier = Parameter()
    ordinal = BoolParameter()
    jobs = IntParameter()
    iterations = IntParameter()
    scoring = Parameter()
    linear_raw = BoolParameter()
    scale = BoolParameter()
    min_scale = Parameter()
    max_scale = Parameter()
    
    nb_alpha = Parameter()
    nb_fit_prior = BoolParameter()
    
    svm_c = Parameter()
    svm_kernel = Parameter()
    svm_gamma = Parameter()
    svm_degree = Parameter()
    svm_class_weight = Parameter()

    lr_c = Parameter()
    lr_solver = Parameter()
    lr_dual = BoolParameter()
    lr_penalty = Parameter()
    lr_multiclass = Parameter()
    lr_maxiter = Parameter()

    linreg_fit_intercept = Parameter()
    linreg_normalize = Parameter()
    linreg_copy_X = Parameter()

    xg_booster = Parameter() 
    xg_silent = Parameter()
    xg_learning_rate = Parameter() 
    xg_min_child_weight = Parameter() 
    xg_max_depth = Parameter() 
    xg_gamma = Parameter() 
    xg_max_delta_step = Parameter()
    xg_subsample = Parameter() 
    xg_colsample_bytree = Parameter() 
    xg_reg_lambda = Parameter()
    xg_reg_alpha = Parameter() 
    xg_scale_pos_weight = Parameter()
    xg_objective = Parameter() 
    xg_seed = IntParameter()
    xg_n_estimators = Parameter() 
     
    knn_n_neighbors = Parameter()
    knn_weights = Parameter()
    knn_algorithm = Parameter()
    knn_leaf_size = Parameter()
    knn_metric = Parameter()
    knn_p = IntParameter()

    def out_predictions(self):
        return self.outputfrominput(inputformat='testvectors', stripextension='.vectors.npz', addextension='.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.transformed.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.translated.predictions.txt' if self.ga and self.linear_raw else '.featuresize_' + str(self.weight_feature_size) + '.' + self.classifier + '.ga.transformed.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.predictions.txt' if self.ga else '.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.translated.predictions.txt' if self.linear_raw else '.labels_' + self.in_trainlabels().path.split('/')[-1].split('.')[-2] + '.' + self.classifier + '.predictions.txt')
    
    def run(self):
        
        if self.complete(): # necessary as it will not complete otherwise
            return True
        else:
            yield Classify(traininstances=self.in_trainvectors().path,trainlabels=self.in_trainlabels().path,testinstances=self.in_testvectors().path,
                classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,linear_raw=self.linear_raw,scale=self.scale,min_scale=self.min_scale,max_scale=self.max_scale,
                ga=self.ga, instance_steps=self.instance_steps,num_iterations=self.num_iterations, population_size=self.population_size, elite=self.elite, crossover_probability=self.crossover_probability,
                mutation_rate=self.mutation_rate,tournament_size=self.tournament_size,n_crossovers=self.n_crossovers,stop_condition=self.stop_condition,weight_feature_size=self.weight_feature_size,
                nb_alpha=self.nb_alpha,nb_fit_prior=self.nb_fit_prior,
                svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
                lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter,
                xg_booster=self.xg_booster, xg_silent=self.xg_silent, xg_learning_rate=self.xg_learning_rate, xg_min_child_weight=self.xg_min_child_weight, 
                xg_max_depth=self.xg_max_depth, xg_gamma=self.xg_gamma, xg_max_delta_step=self.xg_max_delta_step, xg_subsample=self.xg_subsample, 
                xg_colsample_bytree=self.xg_colsample_bytree, xg_reg_lambda=self.xg_reg_lambda, xg_reg_alpha=self.xg_reg_alpha, xg_scale_pos_weight=self.xg_scale_pos_weight,
                xg_objective=self.xg_objective, xg_seed=self.xg_seed, xg_n_estimators=self.xg_n_estimators,
                knn_n_neighbors=self.knn_n_neighbors, knn_weights=self.knn_weights, knn_algorithm=self.knn_algorithm, knn_leaf_size=self.knn_leaf_size,
                knn_metric=self.knn_metric, knn_p=self.knn_p,
                linreg_normalize=self.linreg_normalize, linreg_fit_intercept=self.linreg_fit_intercept, linreg_copy_X=self.linreg_copy_X
            )

#################################################################
### Component ###################################################
#################################################################

@registercomponent
class Report(WorkflowComponent):

    train = Parameter()
    test = Parameter()
    trainlabels = Parameter()
    testlabels = Parameter(default = 'xxx.xxx')
    testdocs = Parameter(default = 'xxx.xxx') # can also be fed through 'test'

    # featureselection parameters
    ga = BoolParameter()
    num_iterations = IntParameter(default=300)
    population_size = IntParameter(default=100)
    elite = Parameter(default='0.1')
    crossover_probability = Parameter(default='0.9')
    mutation_rate = Parameter(default='0.3')
    tournament_size = IntParameter(default=2)
    n_crossovers = IntParameter(default=1)
    stop_condition = IntParameter(default=5)
    weight_feature_size = Parameter(default='0.0')
    instance_steps = IntParameter(default=1)

    # classifier parameters
    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter()
    jobs = IntParameter(default=1)
    iterations = IntParameter(default=10)
    scoring = Parameter(default='roc_auc')
    linear_raw = BoolParameter()
    scale = BoolParameter()
    
    nb_alpha = Parameter(default='1.0')
    nb_fit_prior = BoolParameter()
    
    svm_c = Parameter(default='1.0')
    svm_kernel = Parameter(default='linear')
    svm_gamma = Parameter(default='0.1')
    svm_degree = Parameter(default='1')
    svm_class_weight = Parameter(default='balanced')

    lr_c = Parameter(default='1.0')
    lr_solver = Parameter(default='liblinear')
    lr_dual = BoolParameter()
    lr_penalty = Parameter(default='l2')
    lr_multiclass = Parameter(default='ovr')
    lr_maxiter = Parameter(default='1000')

    linreg_fit_intercept = Parameter(default='1')
    linreg_normalize = Parameter(default='0')
    linreg_copy_X = Parameter(default='1')

    xg_booster = Parameter(default='gbtree') # choices: ['gbtree', 'gblinear']
    xg_silent = Parameter(default='1') # set to '1' to mute printed info on progress
    xg_learning_rate = Parameter(default='0.1') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_min_child_weight = Parameter(default='1') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_max_depth = Parameter(default='6') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_gamma = Parameter(default='0') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_max_delta_step = Parameter(default='0')
    xg_subsample = Parameter(default='1') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_colsample_bytree = Parameter(default='1.0') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_reg_lambda = Parameter(default='1')
    xg_reg_alpha = Parameter(default='0') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 
    xg_scale_pos_weight = Parameter('1')
    xg_objective = Parameter(default='binary:logistic') # choices: ['binary:logistic', 'multi:softmax', 'multi:softprob']
    xg_seed = IntParameter(default=7)
    xg_n_estimators = Parameter(default='100') # choose 'search' for automatic grid search, define grid values manually by giving them divided by space 

    knn_n_neighbors = Parameter(default='3')
    knn_weights = Parameter(default='uniform')
    knn_algorithm = Parameter(default='auto')
    knn_leaf_size = Parameter(default='30')
    knn_metric = Parameter(default='euclidean')
    knn_p = IntParameter(default=2)

    # vectorizer parameters
    weight = Parameter(default = 'frequency') # options: frequency, binary, tfidf
    prune = IntParameter(default = 5000) # after ranking the topfeatures in the training set, based on frequency or idf weighting
    balance = BoolParameter()
    delimiter = Parameter(default=',')

    # featurizer parameters
    ngrams = Parameter(default='1 2 3')
    blackfeats = Parameter(default=False)
    lowercase = BoolParameter()    
    minimum_token_frequency = IntParameter(default=1)
    featuretypes = Parameter(default='tokens')

    # ucto / frog parameters
    tokconfig = Parameter(default=False)
    frogconfig = Parameter(default=False)
    strip_punctuation = BoolParameter(default=True)

    def accepts(self):
        return [tuple(x) for x in numpy.array(numpy.meshgrid(*
            [
                (
                InputFormat(self, format_id='modeled_train',extension ='.model.pkl',inputparameter='train'),
                InputFormat(self, format_id='vectorized_train',extension='.vectors.npz',inputparameter='train'),
                InputFormat(self, format_id='featurized_train',extension='.features.npz',inputparameter='train'),
                InputFormat(self, format_id='featurized_train_csv',extension='.csv',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txt',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.tok.txtdir',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.json',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.frog.jsondir',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.txtdir',inputparameter='train'),
                InputFormat(self, format_id='pre_featurized_train',extension='.txt',inputparameter='train'),
                ),
                (
                InputFormat(self, format_id='classified_test',extension='.predictions.txt',inputparameter='test'),
                InputFormat(self, format_id='vectorized_test',extension='.vectors.npz',inputparameter='test'),
                InputFormat(self, format_id='featurized_test',extension='.features.npz',inputparameter='test'),
                InputFormat(self, format_id='featurized_test_csv',extension='.csv',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.tok.txt',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.tok.txtdir',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.frog.json',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.frog.jsondir',inputparameter='test'),
                InputFormat(self, format_id='pre_featurized_test',extension='.txtdir',inputparameter='test'),
                InputFormat(self, format_id='docs_test',extension='.txt',inputparameter='test')
                ),
                (
                InputFormat(self, format_id='labels_train',extension='.labels',inputparameter='trainlabels')
                ),
                (
                InputFormat(self, format_id='labels_test',extension='.labels',inputparameter='testlabels')
                ),
                (
                InputFormat(self, format_id='docs',extension='.txt',inputparameter='testdocs')
                )
            ]
            )).T.reshape(-1,5)]

    def setup(self, workflow, input_feeds):

        if 'classified_test' in input_feeds.keys(): # reporter can be started
            predictions = input_feeds['classified_test']
        
        else:

            ############################
            ### Prepare vectors ###
            ############################

            trainlabels = input_feeds['labels_train']

            if 'vectorized_train' in input_feeds.keys():
                traininstances = input_feeds['vectorized_train']

            elif 'vectorized_train_csv' in input_feeds.keys():
                traininstances = input_feeds['vectorized_train_csv']

            elif 'featurized_train_csv' in input_feeds.keys():
                traininstances = input_feeds['featurized_train']

            elif 'pre_featurized_train' in input_feeds.keys():
                trainfeaturizer = workflow.new_task('featurize_train',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                trainfeaturizer.in_pre_featurized = input_feeds['pre_featurized_train']

                traininstances = trainfeaturizer.out_featurized

            traincsv=True if ('vectorized_train_csv' in input_feeds.keys()) or ('vectorized_train' in input_feeds.keys() and not 'weight' in [x.split('_')[0] for x in traininstances().path.split('.')]) else False
            trainshort=True if ('vectorized_train' in input_feeds.keys() and not 'weight' in [x.split('_')[0] for x in traininstances().path.split('.')]) else False
    
            if 'vectorized_test' in input_feeds.keys():
                testinstances = input_feeds['vectorized_test']

            elif 'vectorized_test_csv' in input_feeds.keys():
                testinstances = input_feeds['vectorized_test_csv']

            elif 'featurized_test' in input_feeds.keys():
                testinstances = input_feeds['featurized_test']

            else:
                testfeaturizer = workflow.new_task('featurize_test',FeaturizeTask,autopass=False,ngrams=self.ngrams,blackfeats=self.blackfeats,lowercase=self.lowercase,minimum_token_frequency=self.minimum_token_frequency,featuretypes=self.featuretypes,tokconfig=self.tokconfig,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation)
                testfeaturizer.in_pre_featurized = input_feeds['pre_featurized_test']

                testinstances = trainfeaturizer.out_featurized

            if (self.select in testinstances().path.split('.')) or (self.balance and 'balanced' in testinstances().path.split('.')):
                trainvectors = traininstances
                testvectors = testinstances
            elif 'classifier_model' in input_feeds.keys(): # not trainfile to base vectorization on
                if not 'vectorized_test' in input_feeds.keys():
                    print('Testinstances can not be vectorized when the classifier model is given as input (traininstances required), exiting program...')
                    quit()
                else:
                    testvectors = testinstances
            else:
                testcsv=True if ('vectorized_test_csv' in input_feeds.keys() or ('vectorized_test' in input_feeds.keys() and not 'weight' in [x.split('_')[0] for x in testinstances().path.split('.')])) else False
                vectorizer = workflow.new_task('vectorize_traintest',VectorizeTrainTest,autopass=True,weight=self.weight,prune=self.prune,balance=self.balance,select=self.select,select_threshold=self.select_threshold,delimiter=self.delimiter,traincsv=traincsv,testcsv=testcsv,trainshort=trainshort)
                vectorizer.in_train = traininstances
                vectorizer.in_trainlabels = trainlabels
                vectorizer.in_test = testinstances

                trainvectors = vectorizer.out_train
                trainlabels = vectorizer.out_trainlabels
                testvectors = vectorizer.out_test

            ############################
            ### Classify vectors ###
            ############################

            classifier = workflow.new_task('classify',ClassifyTask,autopass=True,
                ga=self.ga, instance_steps=self.instance_steps,num_iterations=self.num_iterations, population_size=self.population_size, elite=self.elite, crossover_probability=self.crossover_probability,
                mutation_rate=self.mutation_rate,tournament_size=self.tournament_size,n_crossovers=self.n_crossovers,stop_condition=self.stop_condition,weight_feature_size=self.weight_feature_size,
                classifier=self.classifier,ordinal=self.ordinal,jobs=self.jobs,iterations=self.iterations,scoring=self.scoring,linear_raw=self.linear_raw,scale=self.scale,scale_min=self.scale_min,scale_max=self.scale_max,
                nb_alpha=self.nb_alpha,nb_fit_prior=self.nb_fit_prior,
                svm_c=self.svm_c,svm_kernel=self.svm_kernel,svm_gamma=self.svm_gamma,svm_degree=self.svm_degree,svm_class_weight=self.svm_class_weight,
                lr_c=self.lr_c,lr_solver=self.lr_solver,lr_dual=self.lr_dual,lr_penalty=self.lr_penalty,lr_multiclass=self.lr_multiclass,lr_maxiter=self.lr_maxiter,
                xg_booster=self.xg_booster, xg_silent=self.xg_silent, xg_learning_rate=self.xg_learning_rate, xg_min_child_weight=self.xg_min_child_weight, 
                xg_max_depth=self.xg_max_depth, xg_gamma=self.xg_gamma, xg_max_delta_step=self.xg_max_delta_step, xg_subsample=self.xg_subsample, 
                xg_colsample_bytree=self.xg_colsample_bytree, xg_reg_lambda=self.xg_reg_lambda, xg_reg_alpha=self.xg_reg_alpha, xg_scale_pos_weight=self.xg_scale_pos_weight,
                xg_objective=self.xg_objective, xg_seed=self.xg_seed, xg_n_estimators=self.xg_n_estimators,
                knn_n_neighbors=self.knn_n_neighbors, knn_weights=self.knn_weights, knn_algorithm=self.knn_algorithm, knn_leaf_size=self.knn_leaf_size,
                knn_metric=self.knn_metric, knn_p=self.knn_p,
                linreg_normalize=self.linreg_normalize, linreg_fit_intercept=self.linreg_fit_intercept, linreg_copy_X=self.linreg_copy_X
            )
            classifier.in_trainvectors = trainvectors
            classifier.in_testvectors = testvectors
            classifier.in_trainlabels = trainlabels

            predictions = classifier.out_predictions 

        ######################
        ### Reporting phase ##
        ######################

        if 'docs' in input_feeds.keys():
            docs = input_feeds['docs']

        if 'labels_test' in input_feeds.keys():

            if self.linear_raw:
                self.ordinal=False
            reporter = workflow.new_task('report_performance',ReportPerformance,autopass=True,ordinal=self.ordinal,teststart=0)
            reporter.in_predictions = predictions
            reporter.in_testlabels = input_feeds['labels_test']
            reporter.in_testdocuments = docs

        else: # report docpredictions

            reporter = workflow.new_task('report_docpredictions',ReportDocpredictions,autopass=True)
            reporter.in_predictions = predictions
            reporter.in_testdocuments = docs

        return reporter
