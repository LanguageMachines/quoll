

def return_fold_indices(num_instances, num_folds, steps=1):
    folds = []
    for i in range(num_folds):
        j = i*steps
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
