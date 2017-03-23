
from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, IntParameter, BoolParameter
import numpy
from scipy import sparse

import quoll.classification_pipeline.functions.vectorizer as vectorizer

################################################################################
###Component to thread the tasks together
################################################################################

@registercomponent
class FilterFeatures(WorkflowComponent):
    
    featurenames = Parameter()
    featureranks = Parameter()
    featurecorrelation = Parameter()

    cutoff = IntParameter(default=0)

    def accepts(self):
        return [ ( InputFormat(self, format_id='featrank', extension='.ranked_features.txt', inputparameter='featureranks'), InputFormat(self, format_id='featcorr', extension='.txt', inputparameter='featurecorrelation'), InputFormat(self, format_id='featurenames', extension='.txt', inputparameter='featurenames') ) ]
    
    def setup(self, workflow, input_feeds):

        feature_filter = workflow.new_task('filter_features',FilterFeaturesTask,autopass=True, cutoff=self.cutoff)
        feature_filter.in_featurenames = input_feeds['featurenames']
        feature_filter.in_featrank = input_feeds['featrank']
        feature_filter.in_featcorr = input_feeds['featcorr']

        return feature_filter

    
################################################################################
###Feature filter
################################################################################

class FilterFeaturesTask(Task):

    in_featurenames = InputSlot()
    in_featrank = InputSlot()
    in_featcorr = InputSlot()

    cutoff = IntParameter()

    def out_filtered_features(self):
        return self.outputfrominput(inputformat='featrank', stripextension='.ranked_features.txt', addextension='.filtered_features.txt')

    def out_filtered_features_index(self):
        return self.outputfrominput(inputformat='featrank', stripextension='.ranked_features.txt', addextension='.filtered_features_indices.txt')
        
    def run(self):

        # open feature names
        with open(self.in_featurenames().path,'r',encoding='utf-8') as infile:
            fn = infile.read().strip().split('\n')

        # open feature ranks
        with open(self.in_featrank().path,'r',encoding='utf-8') as infile:
            feature_ranks = infile.read().strip().split('\n')
            fr = [f.split('\t')[0] for f in feature_ranks]

        # open feature correlation
        featcorr = {}
        featsub = {}
        with open(self.in_featcorr().path,'r',encoding='utf-8') as infile:
            lines = infile.read().strip().split('\n')
            for line in lines:
                tokens = line.split('\t')
                if len(tokens) > 1:
                    featcorr[tokens[0]] = tokens[1].split(', ')
                if len(tokens) > 2:
                    featsub[tokens[0]] = tokens[2].split(', ')

        # filter features
        filtered_features = vectorizer.filter_features_correlation(fr,featcorr,featsub)

        # select top n
        if self.cutoff > 0:
            filtered_features = filtered_features[:self.cutoff]

        # obtain indices filtered features
        filtered_features_indices = sorted([fn.index(feature) for feature in filtered_features])

        # write filtered feature names
        with open(self.out_filtered_features().path,'w',encoding='utf-8') as out:
            out.write('\n'.join([f.strip() for f in filtered_features]))

        # write filtered feature indices
        with open(self.out_filtered_features_index().path,'w',encoding='utf-8') as out:
            out.write(' '.join([str(i) for i in filtered_features_indices]))
