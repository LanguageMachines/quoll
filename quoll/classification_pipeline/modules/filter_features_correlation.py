
from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, IntParameter, BoolParameter
import numpy
from scipy import sparse

import quoll.classification_pipeline.functions.vectorizer as vectorizer

################################################################################
###Component to thread the tasks together
################################################################################

@registercomponent
class FilterFeatures(WorkflowComponent):
    
    vectors = Parameter()
    featurenames = Parameter()
    featureranks = Parameter()
    featurecorrelation = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='vectors',extension='.vectors.npz',inputparameter='vectors'), InputFormat(self, format_id='featureranks', extension='.ranked_features.txt', inputparameter='featrank'), InputFormat(self, format_id='featurecorrelation', extension='.corr.txt', inputparameter='featcorr'), InputFormat(self, format_id='featurenames', extension='.txt', inputparameter='featurenames') ) ]
    
    def setup(self, workflow, input_feeds):

        feature_filter = workflow.new_task('filter_features',FilterFeaturesTask,autopass=True)

        return feature_filter

    
################################################################################
###Feature filter
################################################################################

class FilterFeaturesTask(Task):

    in_vectors = InputSlot()
    in_featurenames = InputSlot()
    in_featrank = InputSlot()
    in_featcorr = InputSlot()

    def out_filtered_vectors(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.filtered.vectors.npz')
        
    def run(self):

        # open instances
        loader = numpy.load(self.in_vectors().path)
        instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # open feature names
        with open(self.in_featurenames().path,'r',encoding='utf-8') as infile:
            fn = infile.read().strip().split('\n')

        # open feature ranks
        with open(self.in_featrank().path,'r',encoding='utf-8') as infile:
            feature_ranks = infile.read().strip().split('\n')
            fr = [f[0] for f in feature_ranks]

        # open feature correlation
        featcorr = {}
        featsub = {}
        with open(self.in_featcorr().path,'r',encoding='utf-8') as infile:
            lines = infile.read().strip().split('\n')
            for line in lines:
                featcorr[lines[0]] = line[1].split(', ')
                featsub[lines[0]] = line[2].split(', ')

        # filter features
        filtered_features = vectorizer.filter_features_correlation(fr,featcorr,featsub)

        # obtain indices filtered features
        filtered_features_indices = [fn.index(feature) for feature in filtered_features]

        # transform vectors
        transformed_vectors = vectorizer.compress_vectors(instances,filtered_features_indices)

        # write vectors
        numpy.savez(self.out_filtered_vectors().path, data=transformed_vectors.data, indices=transformed_vectors.indices, indptr=transformed_vectors.indptr, shape=transformed_vectors.shape)
