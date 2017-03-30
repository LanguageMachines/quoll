



def return_fold_indices(num_instances, num_folds, steps=1):
    folds = []
    for i in range(num_folds):
        j = i+(steps-1)
        fold = []
        while j < num_instances:
            k = 0
            while k < steps:
                fold.append(j)
                j+=1
                k+=1
            j += num_folds
        folds.append(fold)
    return folds
