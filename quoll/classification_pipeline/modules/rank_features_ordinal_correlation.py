
from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, IntParameter, BoolParameter
import numpy
from scipy import sparse

import quoll.classification_pipeline.functions.vectorizer as vectorizer

  
################################################################################
###Feature ranker
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
        feature_ranker.in_vectors = input_feeds['vectors']
        feature_ranker.in_labels = input_feeds['labels']
        feature_ranker.in_featurenames = input_feeds['featurenames']

        return feature_ranker

class RankFeaturesOrdinalTask(Task):

    in_vectors = InputSlot()
    in_labels = InputSlot()
    in_featurenames = InputSlot()

    def out_ranked_features(self):
        return self.outputfrominput(inputformat='featurenames', stripextension='.txt', addextension='.ranked.txt')

    def run(self):

        # open instances
        loader = numpy.load(self.in_vectors().path)
        instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # open labels
        with open(self.in_labels().path,'r',encoding='utf-8') as infile:
            labels = numpy.array(infile.read().strip().split('\n'))
        labels_np = numpy.array([float(label) for label in labels])

        # load feature names
        with open(self.in_featurenames().path,'r',encoding='utf-8') as infile:
            fn = infile.read().strip().split('\n')

        # calculate feature correlations
        sorted_feature_correlation = vectorizer.calculate_ordinal_correlation_feature_labels(instances,labels_np)

        # write to file
        with open(self.out_ranked_features().path,'w',encoding='utf-8') as out:
            # for fc in sorted_feature_correlation:
            #     out.write('\t'.join([fc[0],fn[fc[0]],str(fc[2]),str(fc[3])]) + '\n')
            out.write('\n'.join(['\t'.join([str(fc[0]),fn[fc[0]],str(fc[2]),str(fc[3])]) for fc in sorted_feature_correlation]))


    
################################################################################
###Feature correlation 
################################################################################

@registercomponent
class CalculateFeatureCorrelation(WorkflowComponent):
    
    vectors = Parameter()
    featurenames = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='vectors',extension='.vectors.npz',inputparameter='vectors'), InputFormat(self, format_id='featurenames', extension='.txt', inputparameter='featurenames') ) ]
    
    def setup(self, workflow, input_feeds):

        feature_correlation = workflow.new_task('feature_correlation',CalculateFeatureCorrelationTask,autopass=True)
        feature_correlation.in_vectors = input_feeds['vectors']
        feature_correlation.in_featurenames = input_feeds['featurenames']

        return feature_correlation

class CalculateFeatureCorrelationTask(Task):

    in_vectors = InputSlot()
    in_featurenames = InputSlot()

    def out_feature_correlation(self):
        return self.outputfrominput(inputformat='featurenames', stripextension='.txt', addextension='.correlations.txt')

    def run(self):

        # open instances
        loader = numpy.load(self.in_vectors().path)
        instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load feature names
        with open(self.in_featurenames().path,'r',encoding='utf-8') as infile:
            fn = infile.read().strip().split('\n')

        # calculate feature correlations
        feature_correlation = vectorizer.calculate_feature_correlation(instances)

        # write to file
        with open(self.out_feature_correlation().path,'w',encoding='utf-8') as out:
            # for fc in feature_correlation:
            #     print(fc,len(fc))
            #     out.write('\t'.join([str(fc[0]),str(fc[1]),fn[fc[0]],fn[fc[1]],str(fc[2]),str(fc[3]),str(fc[4])]) + '\n')
           out.write('\n'.join(['\t'.join([str(fc[0]),str(fc[1]),fn[fc[0]],fn[fc[1]],str(fc[2]),str(fc[3]),str(fc[4])]) for fc in feature_correlation]))
