
import numpy as np
from itertools import combinations
from copy import copy

##########################
### based on Deerishi (https://github.com/deerishi/Decision-Tree-in-Python-for-Continuous-feature_indexibutes.git)
##########################

class DecisionTreeContinuous:
    
    def __init__(self):
        self.Tree=-1*(np.ones((1000,1)))
        self.decisions={}

    def fit(self,instances,labels):
        self.construct_tree(instances,labels,1)

    def transform(self,instances):
        predicted=[]
        print('Performing predictions')
        for i in range(0,instances.shape[0]):
            res=self.findLabel(instances[i],1)
            predicted.append(res)
            print('testing ',i,' predicted= ',res)
            
        return predicted

    def return_tree(self):
        return(self.Tree)

    def apply_segmentation(self,segmentation,instances_column,labels):
        groups = []
        for rule in segmentation:
            selection = [i for i,val in enumerate(instances_column) if (val > rule[0]) and (val < rule[1])]
            groups.append(selection)
        return groups

    def return_segmentations(self,thresholds,num_segments,max_value):
        segmentations = []
        combis = itertools.combinations(thresholds,num_segments-1)
        for combi in combis:
            segmentation = [[0,combi[0]]]
            for bound in range(1,len(combi)):
                segmentation.append([combi[bound-1],combi[bound]])
            segmentation.append([combi[bound],max_value+1])
            segmentations.append(segmentation)
        return segmentations

    def calculate_entropy(self,labelfracs):
        if sum(i > 0 for i in labelfracs) == 1: # all instances in one category
            entropy=0
        else:
            entropy = -1*(labelfracs[0]*np.log2(labelfracs[0]))
            for frac in labelfracs[1:]:
                entropy -= (frac * np.log(frac)) 
            =-1*pH*np.log2(pH) - pC*np.log2(pC)

    def obtain_labelfracs(self,labels):
        labelfracs = []
        for label in self.unique_labels:
            c = np.where(labels==label)
            labelfracs.append(float(c.shape[0])/labels.shape[0])
        return labelfracs

    def calculateIG(self,groups,labels):
        # current entropy
        labelfracs = self.obtain_labelfracs(labels,self.unique_labels)
        current_entropy = self.calculate_entropy(labelfracs)
        # entropy of each grouping
        group_entropy = []
        for group in groups:
            labelfracs = self.obtain_labelfracs(group,unique_labels)
            group_entropy.append(self.calculate_entropy(labelfracs))
        infogain = current_entropy - sum(group_entropy)
        return infogain
       
    def findSegmentationAndIG(self,instances,feature_index,labels):
        instances_column = instances[:,feature_index]
        feature_values = set(instances[:,feature_index])
        feature_values_sorted=sorted(feature_values)
        threshholds=[]
        for i in range(0,len(feature_values_sorted)-1):
           threshholds.append((feature_values_sorted[i]+feature_values_sorted[i+1])/2)
        thresholds=set(candidate_thresholds)

        # check which of the thresholds were already used 
        # if feature_index in self.used_thresholds:
        #     for used in self.used_thresholds[feature_index]:
        #         if used in candidate_thresholds:
        #             candidate_thresholds.remove(used)
        
        IG=[]
        segmentations = self.return_segmentations(candidate_thresholds,len(labels),max(feature_values_sorted))
        for segmentation in segmentations:
            groups = self.apply_segmentation(segmentation,instances_column,labels)
            labelgroups = []
            for group in groups:
                labelgroups.append([labels[i] for i in group])
            IG.append(self.calculateIG(labelgroups,labels))
        maxIG=max(IG)
        maxSegmentation=IG.index(maxIG)
        
        return segmentations[maxSegmentation],maxIG
            
    def construct_tree(self,instances,labels,node_index):
            #since its a recursive function we need to have a base case. return when the number of wrong classes is 0. maybe 
            #we can chane it later
            print('node index is ',node_index)
            label_instances = []
            print('getting label indexes')
            for label in sorted(list(set(labels))):
                li = np.where(labels==label)[0]
                print('number of instances with label',label,'is',li)
                label_instances.append(li)
            if sum(i > 0 for i in label_instances) == 1: # all instances in one category
                self.decisions[node_index]=tuple([li.shape[0] for li in label_instances])
                return
            
            print('Generating segmentations')
            IGA=[]
            feature_segmentations=[]
            for feature_index in range(0,instances.shape[1]):
                segm,IG=self.findSegmentationAndIG(instances,feature_index,labels)
                IGA.append(IG)
                feature_segmentations.append(segm)
                
            maxIG=max(IGA)
            feature_index=IGA.index(maxIG)
            print 'Feature index is',feature_index
            best_feature_segmentation=feature_segmentations[feature_index]
            # self.usedThresholds[feature_index].add(thresh)
            self.Tree[node_index]=feature_index
            self.Segmentations[node_index]=best_feature_segmentation
            
            # apply segmentation to construct new nodes
            print('\nApplying segmentation')
            new_nodes_rows = []
            for i,group in enumerate(self.apply_segmentation(best_feature_segmentation,instances[:,feature_index],labels)):
                group_instances = instances[group]
                label_instances = labels[group]
                self.construct_tree(group_instances,2*node_index+i,label_instances)

        
    def predict(self,instances,node_index):

        if self.Tree[node_index][0]==-1:
            #then we check the decisions 
            decisions = self.decisions[node_index]
            print(decisions)

        elif instances[self.Tree[node_index][0]]>=self.Segmentations[node_index][0]:
            #go left
            res=self.predict(data,2*node)
        else:
            res=self.predict(data,2*nodeNum+1)

        return res
