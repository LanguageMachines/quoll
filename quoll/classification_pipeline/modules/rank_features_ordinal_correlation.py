
from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, IntParameter, BoolParameter
import numpy
from scipy import sparse

import quoll.classification_pipeline.functions.vectorizer as vectorizer

################################################################################
###Component to thread the tasks together
################################################################################

@registercomponent
class RankFeaturesOrdinal(WorkflowComponent):
    
    vectors = Parameter()
    labels = Parameter()
    featurenames = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='vectors',extension='.vectors.npz',inputparameter='vectors'), InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), InputFormat(self, format_id='featurenames', extension='.txt', inputparameter='featurenames') ) ]
    
    def setup(self, workflow, input_feeds):

        feature_ranker = workflow.new_task('rank_features',RankFeaturesOrdinalTask,autopass=True)

        return feature_ranker

    
################################################################################
###Feature ranker
################################################################################

class RankFeaturesOrdinalTask(Task):

    in_vectors = InputSlot()
    in_labels = InputSlot()
    in_featurenames = InputSlot()

    def out_ranked_features(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.ranked_features.txt')
        
    def run(self):

        # open instances
        loader = numpy.load(self.in_vectors().path)
        instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # open labels
        with open(self.in_labels().path,'r',encoding='utf-8') as infile:
            labels = numpy.array(infile.read().strip().split('\n'))

        # load feature names
        with open(self.in_featurenames().path,'r',encoding='utf-8') as infile:
            fn = infile.read().strip().split('\n')

        # calculate feature correlations
        sorted_feature_correlations = vectorizer.calculate_ordinal_correlation_feature_labels(instances,labels)

        # write to file
        with open(self.out_ranked_features().path,'w',encoding='utf-8') as out:
            out.write('\n'.join(['\t'.join([fn[fc[0]],str(fc[2]),str(fc[3])]) for fc in sorted_feature_correlation]))
