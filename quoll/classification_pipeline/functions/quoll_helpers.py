
##################################
### Nfold CV functions ###########
##################################

def return_fold_indices(num_instances, num_folds, steps=1, startindex=0):
    folds = []
    for i in range(num_folds):
        j = startindex + (i*steps)
        fold = []
        while j < num_instances:
            k = 1
            fold.append(j)
            l = j
            while k < steps:
                l+=1
                k+=1
                fold.append(l)
            j += (num_folds*steps)
        folds.append(fold)
    return folds

##################################
### Parameter encode functions ###
##################################

def prepare_preprocess_input(**kwargs):
    return '--'.join([str(x) for x in [kwargs['tokconfig'],kwargs['frogconfig'],kwargs['strip_punctuation']]])

def prepare_featurize_input(**kwargs):
    return '--'.join([str(x) for x in [kwargs['ngrams'],kwargs['blackfeats'],kwargs['lowercase'],kwargs['minimum_token_frequency'],kwargs['featuretypes']]])

def prepare_vectorize_input(**kwargs):
    return '--'.join([str(x) for x in [kwargs['weight'],kwargs['prune'],kwargs['balance'],kwargs['delimiter'],kwargs['select'],kwargs['selector'],kwargs['select_threshold']]])

def prepare_classify_input(**kwargs):
    return '--'.join([str(x) for x in [
        kwargs['classifier'],kwargs['ordinal'],kwargs['jobs'],kwargs['iterations'],kwargs['scoring'],kwargs['linear_raw'],
        kwargs['random_clf'],
        kwargs['nb_alpha'],kwargs['nb_fit_prior'],
        kwargs['svm_c'],kwargs['svm_kernel'],kwargs['svm_gamma'],kwargs['svm_degree'],kwargs['svm_class_weight'],
        kwargs['lr_c'],kwargs['lr_solver'],kwargs['lr_dual'],kwargs['lr_penalty'],kwargs['lr_multiclass'],kwargs['lr_maxiter'],
        kwargs['linreg_normalize'],kwargs['linreg_fit_intercept'],kwargs['linreg_copy_X'],
        kwargs['xg_booster'],kwargs['xg_silent'],kwargs['xg_learning_rate'],kwargs['xg_min_child_weight'],kwargs['xg_max_depth'],kwargs['xg_gamma'],kwargs['xg_max_delta_step'],kwargs['xg_subsample'], 
        kwargs['xg_colsample_bytree'],kwargs['xg_reg_lambda'],kwargs['xg_reg_alpha'],kwargs['xg_scale_pos_weight'],kwargs['xg_objective'],kwargs['xg_seed'],kwargs['xg_n_estimators'], 
        kwargs['knn_n_neighbors'],kwargs['knn_weights'],kwargs['knn_algorithm'],kwargs['knn_leaf_size'],kwargs['knn_metric'],kwargs['knn_p']
    ]])

def prepare_ga_input(**kwargs):
    return '--'.join([str(x) for x in [
        kwargs['num_iterations'],kwargs['population_size'],kwargs['elite'],kwargs['crossover_probability'],kwargs['mutation_rate'],kwargs['tournament_size'],
        kwargs['n_crossovers'],kwargs['stop_condition'],kwargs['weight_feature_size'],kwargs['instance_steps'],kwargs['sampling'],kwargs['samplesize']
    ]])

def prepare_validate_input(**kwargs):
    return '--'.join([str(x) for x in [kwargs['n'],kwargs['steps'],kwargs['teststart'],kwargs['testend']]])

def prepare_task_input(tasks, **kwargs):
    modules = {
        'preprocess': prepare_preprocess_input,
        'featurize' : prepare_featurize_input,
        'vectorize' : prepare_vectorize_input,
        'classify'  : prepare_classify_input,
        'ga'        : prepare_ga_input,
        'validate'  : prepare_validate_input
    }
    task_args = zip(tasks,[modules[task](**kwargs) for task in tasks])
    print(task_args)
    return task_args

##################################
### Parameter decode functions ###
##################################

def prepare_preprocess_input(paramstring):
    return zip(['tokconfig','frogconfig','strip_punctuation'],paramstring.split('--'))

def prepare_featurize_input(paramstring):
    return zip(['ngrams','blackfeats','lowercase','minimum_token_frequency','featuretypes'],paramstring.split('--'))

def prepare_vectorize_input(paramstring):
    return zip(['weight','prune','balance','delimiter','select','selector','select_threshold'],paramstring.split('--'))

def prepare_classify_input(paramstring):
    return zip(['classifier','ordinal','jobs','iterations','scoring','linear_raw','random_clf','nb_alpha','nb_fit_prior','svm_c','svm_kernel','svm_gamma','svm_degree','svm_class_weight',
        'lr_c','lr_solver','lr_dual','lr_penalty','lr_multiclass','lr_maxiter','linreg_normalize','linreg_fit_intercept','linreg_copy_X','xg_booster','xg_silent','xg_learning_rate',
        'xg_min_child_weight','xg_max_depth','xg_gamma','xg_max_delta_step','xg_subsample','xg_colsample_bytree','xg_reg_lambda','xg_reg_alpha','xg_scale_pos_weight','xg_objective',
        'xg_seed','xg_n_estimators','knn_n_neighbors','knn_weights','knn_algorithm','knn_leaf_size','knn_metric','knn_p'],paramstring.split('--'))

def prepare_ga_input(paramstring):
    return zip(['num_iterations','population_size','elite','crossover_probability','mutation_rate','tournament_size','n_crossovers','stop_condition','weight_feature_size','instance_steps','sampling',
        'samplesize'],paramstring.split('--'))

def decode_validate_input(paramstring):
    return zip(['n','steps','teststart','testend'],paramstring.split('--'))

def decode_task_input(tasks, encoded_args):
    modules = {
        'preprocess': decode_preprocess_input,
        'featurize' : decode_featurize_input,
        'vectorize' : decode_vectorize_input,
        'classify'  : decode_classify_input,
        'ga'        : decode_ga_input,
        'validate'  : decode_validate_input
    }
    parameters_args = {**modules[task](encoded_args[i]) for i,task in enumerate(tasks)}
    print(parameter_args)
    return parameter_args
