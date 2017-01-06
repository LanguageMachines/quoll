
import numpy

from luiginlp.engine import Task, InputSlot, IntParameter

import functions.nfold_cv_functions as nfold_cv_functions
import functions.linewriter as linewriter

################################################################################
###Binner
################################################################################

class MakeBins(Task):

    in_labels = InputSlot()

    n = IntParameter()

    def out_folds(self):
        return self.outputfrominput(inputformat='labels', stripextension='.labels', addextension='.' + str(self.n) + 'fold_cv')

    def out_bins(self):
        return self.outputfrominput(inputformat='labels', stripextension='.labels', addextension='.' + str(self.n) + 'fold_cv/folds.bins.csv')
        
    def run(self):

        # make nfold_cv directory
        self.setup_output_dir(self.out_folds().path)

        # open labels
        with open(self.in_labels().path,'r',encoding='utf-8') as infile:
            labels = numpy.array(infile.read().strip().split('\n'))

        # select rows per fold based on shape of the features
        num_instances = len(labels)
        fold_indices = nfold_cv_functions.return_fold_indices(num_instances, self.n)        
        
        # write indices of bins to file
        lw = linewriter.Linewriter(fold_indices)
        lw.write_csv(self.out_bins().path)
