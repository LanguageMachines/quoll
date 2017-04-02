
import numpy

from luiginlp.engine import Task, InputSlot, IntParameter

import quoll.classification_pipeline.functions.nfold_cv_functions as nfold_cv_functions
import quoll.classification_pipeline.functions.linewriter as linewriter

################################################################################
###Binner
################################################################################

class MakeBins(Task):

    in_labels = InputSlot()

    n = IntParameter()
    steps = IntParameter(default = 1)

    def out_bins(self):
        return self.outputfrominput(inputformat='labels', stripextension='.labels', addextension='.' + str(self.n) + 'folds.bins.csv')
        
    def run(self):

        # open labels
        with open(self.in_labels().path,'r',encoding='utf-8') as infile:
            labels = numpy.array(infile.read().strip().split('\n'))

        # select rows per fold based on shape of the features
        num_instances = len(labels)
        fold_indices = nfold_cv_functions.return_fold_indices(num_instances,self.n,self.steps)        
        
        # write indices of bins to file
        lw = linewriter.Linewriter(fold_indices)
        lw.write_csv(self.out_bins().path)

class MakeBinsRestr(Task):

    in_labels = InputSlot()
    in_meta = InputSlot()

    n = IntParameter()
    

    def out_bins(self):
        return self.outputfrominput(inputformat='labels', stripextension='.labels', addextension='.' + str(self.n) + 'folds.bins.csv')
        
    def run(self):

        # open labels
        with open(self.in_labels().path,'r',encoding='utf-8') as infile:
            labels = numpy.array(infile.read().strip().split('\n'))

        # select rows per fold based on shape of the features
        num_instances = len(labels)
        fold_indices = nfold_cv_functions.return_fold_indices(num_instances,self.n,self.steps)        
        
        # write indices of bins to file
        lw = linewriter.Linewriter(fold_indices)
        lw.write_csv(self.out_bins().path)