



def return_fold_indices(num_instances, num_folds):
    folds = []
    for i in range(num_folds):
        j = i
        fold = []
        while j < num_instances:
            fold.append(j)
            j += num_folds
        folds.append(fold)
    return folds
