
from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, IntParameter, BoolParameter
from collections import defaultdict
import numpy
import copy
from scipy import sparse

import quoll.classification_pipeline.functions.vectorizer as vectorizer

    
################################################################################
###Feature filter
################################################################################

@registercomponent
class FilterFeatures(WorkflowComponent):
    
    featurenames = Parameter()
    featureranks = Parameter()
    featurecorrelation = Parameter()

    cutoff = IntParameter(default=0)

    def accepts(self):
        return [ ( InputFormat(self, format_id='featrank', extension='.ranked.txt', inputparameter='featureranks'), InputFormat(self, format_id='featcorr', extension='.txt', inputparameter='featurecorrelation'), InputFormat(self, format_id='featurenames', extension='.txt', inputparameter='featurenames') ) ]
    
    def setup(self, workflow, input_feeds):

        feature_filter = workflow.new_task('filter_features',FilterFeaturesTask,autopass=True, cutoff=self.cutoff)
        feature_filter.in_featurenames = input_feeds['featurenames']
        feature_filter.in_featrank = input_feeds['featrank']
        feature_filter.in_featcorr = input_feeds['featcorr']

        return feature_filter

class FilterFeaturesTask(Task):

    in_featurenames = InputSlot()
    in_featrank = InputSlot()
    in_featcorr = InputSlot()

    cutoff = IntParameter()

    def out_filtered_features(self):
        return self.outputfrominput(inputformat='featrank', stripextension='.ranked.txt', addextension='.filtered_features.txt')

    def out_filtered_features_index(self):
        return self.outputfrominput(inputformat='featrank', stripextension='.ranked.txt', addextension='.filtered_features_indices.txt')
        
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
                tokens = line.strip().split('\t')
                if len(tokens) > 1:
                    featcorr[tokens[0]] = tokens[1].strip().split(', ')
                if len(tokens) > 2:
                    featsub[tokens[0]] = tokens[2].strip().split(', ')

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


################################################################################
###Feature filter Groups
################################################################################

@registercomponent
class FilterFeaturesGroups(WorkflowComponent):
    
    featurenames = Parameter()
    featureranks = Parameter()
    featuregroups = Parameter()

    def accepts(self):
        return [ ( InputFormat(self, format_id='featrank', extension='.ranked.txt', inputparameter='featureranks'), InputFormat(self, format_id='featgroups', extension='.txt', inputparameter='featuregroups'), InputFormat(self, format_id='featurenames', extension='.txt', inputparameter='featurenames') ) ]
    
    def setup(self, workflow, input_feeds):

        feature_filter = workflow.new_task('filter_features',FilterFeaturesTask,autopass=True)
        feature_filter.in_featurenames = input_feeds['featurenames']
        feature_filter.in_featrank = input_feeds['featrank']
        feature_filter.in_featgroups = input_feeds['featgroups']

        return feature_filter

class FilterFeaturesGroupsTask(Task):

    in_featurenames = InputSlot()
    in_featrank = InputSlot()
    in_featgroups = InputSlot()

    def out_filtered_features(self):
        return self.outputfrominput(inputformat='featrank', stripextension='.ranked.txt', addextension='.filtered_features.txt')

    def out_filtered_features_index(self):
        return self.outputfrominput(inputformat='featrank', stripextension='.ranked.txt', addextension='.filtered_features_indices.txt')
        
    def run(self):

        # open feature names
        with open(self.in_featurenames().path,'r',encoding='utf-8') as infile:
            fn = infile.read().strip().split('\n')

        # open feature ranks
        with open(self.in_featrank().path,'r',encoding='utf-8') as infile:
            feature_ranks = infile.read().strip().split('\n')
            fr = [f.split('\t')[1] for f in feature_ranks]

        # open feature groups
        feature_group = {}
        with open(self.in_featgroups().path,'r',encoding='utf-8') as infile:
            lines = infile.read().strip().split('\n')
            for line in lines:
                features = line.strip().split()
                for feature in features:
                    group_features = copy.deepcopy(features)
                    group_features.remove(feature)
                    feature_group[feature] = group_features

        # filter features
        filtered_features = vectorizer.filter_features_groups(fr,feature_group)

        # obtain indices filtered features
        filtered_features_indices = sorted([fn.index(feature) for feature in filtered_features])

        # write filtered feature names
        with open(self.out_filtered_features().path,'w',encoding='utf-8') as out:
            out.write('\n'.join([f.strip() for f in filtered_features]))

        # write filtered feature indices
        with open(self.out_filtered_features_index().path,'w',encoding='utf-8') as out:
            out.write(' '.join([str(i) for i in filtered_features_indices]))


################################################################################
###Feature filter F
################################################################################

@registercomponent
class FilterFeaturesF(WorkflowComponent):
    
    featurenames = Parameter()
    featurestrength = Parameter()
    featurecorrelation = Parameter()

    threshold_strength = Parameter()
    threshold_correlation = Parameter()

    def accepts(self):
        return [ ( InputFormat(self, format_id='featstrength', extension='.ranked.txt', inputparameter='featurestrength'), InputFormat(self, format_id='featcorr', extension='.correlations.txt', inputparameter='featurecorrelation'), InputFormat(self, format_id='featurenames', extension='.txt', inputparameter='featurenames') ) ]
    
    def setup(self, workflow, input_feeds):

        feature_filter = workflow.new_task('filter_features',FilterFeaturesFTask,autopass=True, threshold_strength=self.threshold_strength,threshold_correlation=self.threshold_correlation)
        feature_filter.in_featurenames = input_feeds['featurenames']
        feature_filter.in_featstrength = input_feeds['featstrength']
        feature_filter.in_featcorr = input_feeds['featcorr']

        return feature_filter

class FilterFeaturesFTask(Task):

    in_featurenames = InputSlot()
    in_featstrength = InputSlot()
    in_featcorr = InputSlot()

    threshold_strength = Parameter()
    threshold_correlation = Parameter()

    def out_filtered_features(self):
        return self.outputfrominput(inputformat='featstrength', stripextension='.ranked.txt', addextension='.filtered_features.txt')

    def out_filtered_features_index(self):
        return self.outputfrominput(inputformat='featstrength', stripextension='.ranked.txt', addextension='.filtered_features_indices.txt')
      
    def out_filter_log(self):
        return self.outputfrominput(inputformat='featstrength', stripextension='.ranked.txt', addextension='.filtered_features_log.txt')

    def run(self):

        # open feature names
        with open(self.in_featurenames().path,'r',encoding='utf-8') as infile:
            fn = infile.read().strip().split('\n')

        # open feature ranks
        feature_strength = {}
        with open(self.in_featstrength().path,'r',encoding='utf-8') as infile:
            feature_ranks = infile.read().strip().split('\n')
            for f in feature_ranks:
                data = f.split('\t')
                feature_strength[int(data[0])] = abs(float(data[2]))

        # open feature correlation
        featcorr = defaultdict(lambda : {})
        with open(self.in_featcorr().path,'r',encoding='utf-8') as infile:
            lines = infile.read().strip().split('\n')
            for line in lines:
                tokens = line.strip().split('\t')
                f1 = int(tokens[0])
                f2 = int(tokens[1])
                corr = float(tokens[4])
                featcorr[f1][f2] = corr
                featcorr[f2][f1] = corr

        # filter features
        filtered_features, log = vectorizer.filter_features_correlation_f(feature_strength,featcorr,float(self.threshold_strength),float(self.threshold_correlation))

        # obtain names of filtered features
        filtered_features_names = [fn[feature_index] for feature_index in filtered_features]

        # write filtered features
        with open(self.out_filtered_features().path,'w',encoding='utf-8') as out:
            out.write('\n'.join([f.strip() for f in filtered_features_names]))

        # write filtered feature indices
        with open(self.out_filtered_features_index().path,'w',encoding='utf-8') as out:
            out.write(' '.join([str(i) for i in filtered_features]))

        # write log
        with open(self.out_filter_log().path,'w',encoding='utf-8') as out:
            out.write(log)
