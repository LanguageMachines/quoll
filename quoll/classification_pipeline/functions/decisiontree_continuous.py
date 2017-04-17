
import math
import numpy as np
from scipy import stats
from itertools import combinations

##########################
### based on Deerishi (https://github.com/deerishi/Decision-Tree-in-Python-for-Continuous-feature_indexibutes.git)
##########################

class DecisionTreeContinuous:
    
    def __init__(self):
        self.Tree={}
        self.predictions={}
        self.train_instances = False
        self.test_instances = False
        self.labels = []
        self.unique_labels = []

    def fit(self,instances,labels):
        self.train_instances=instances
        self.labels = labels
        self.unique_labels = list(set(labels))
        self.construct_tree(list(range(len(labels))),1)

    def transform(self,instances):
        # predicted=[]
        # print('Performing predictions')
        self.test_instances = instances
        self.predict(list(range(instances.shape[0])),1)
        return self.predictions
        # for i in range(0,instances.shape[0]):
        #     res=self.findLabel(instances[i],1)
        #     predicted.append(res)
        #     print('testing ',i,' predicted= ',res)
            
        # return predicted

    def return_tree(self):
        return(self.Tree)

    def apply_segmentation(self,segmentation,instances_column):
        groups = []
        for rule in segmentation:
            # print('Rule',rule)
            # print('INSTANCES_COLUMN',instances_column)
            selection = [i for i,val in enumerate(instances_column) if (val >= rule[0]) and (val < rule[1])]
            # print('Selection:',selection)
            groups.append(selection)
        return groups

    def find_best_segmentation_binrec(self,target_num_segments,saved_thresholds,current_segments_labels,current_segments_thresholds,current_segments_featurevals):
        # print('TARGET',target_num_segments)
        # print('SAVED',saved_thresholds)
        # print('CURRENT (labels)',current_segments_labels)
        # print('CURRENT (thresholds)',current_segments_thresholds)
        # print('CURRENT (featvals)',current_segments_featurevals)
        best_thresholds_by_segment = []
        for i,segment in enumerate(current_segments_thresholds):
            groupings = []
            labels_segment = current_segments_labels[i]
            featvals_segment = current_segments_featurevals[i]
            igs = []
            for threshold in segment:
                # make division
                subsegments = [[j for j,val in enumerate(featvals_segment) if (val < threshold)],[j for j,val in enumerate(featvals_segment) if (val > threshold)]]
                labels_subsegments = [[labels_segment[j] for j in subsegments[0]],[labels_segment[j] for j in subsegments[1]]]
                # print('BEFORE IG:',labels_subsegments,labels_segment)
                if all([len(x)>0 for x in labels_subsegments]):
                    IG = self.calculateIG(labels_subsegments,labels_segment)
                    igs.append(IG)
            # print('ALL',igs)
            if len(igs) > 0:
                best = max(igs)
                # print('BEST',igs)
                best_index = igs.index(best)
                best_thresholds_by_segment.append([i,best,best_index])
        if len(best_thresholds_by_segment) == 0:
            return False
        # print('CHOICES',best_thresholds_by_segment)
        best_thresholds_sorted = sorted(best_thresholds_by_segment,key = lambda k : k[1],reverse=True)
        best_threshold_segment = best_thresholds_sorted[0]
        # print('SELECTION',best_threshold_segment)
        # replace segment
        segment_to_replace_thresholds = current_segments_thresholds[best_threshold_segment[0]]
        best_threshold = segment_to_replace_thresholds[best_threshold_segment[2]]
        # print('SEGMENT TO REPLACE',segment_to_replace_thresholds)
        # print('BEST THRESHOLD',best_threshold)
        segment_to_replace_featvals = current_segments_featurevals[best_threshold_segment[0]]
        segment_to_replace_labels = current_segments_labels[best_threshold_segment[0]]
        new_segments_indices = [[i for i,val in enumerate(segment_to_replace_featvals) if (val < best_threshold)],[i for i,val in enumerate(segment_to_replace_featvals) if (val > best_threshold)]]
        # print('NEW SEGMENTS INDICES',new_segments_indices)
        # print('SEGMENT TO REPLACE THRESHOLDS',segment_to_replace_thresholds)
        # print('LENGTH',len(segment_to_replace_thresholds))
        replacing_segments_featurevals = [[segment_to_replace_featvals[i] for i in new_segments_indices[0]],[segment_to_replace_featvals[i] for i in new_segments_indices[1]]]
        replacing_segments_labels = [[segment_to_replace_labels[i] for i in new_segments_indices[0]],[segment_to_replace_labels[i] for i in new_segments_indices[1]]]
        replacing_segments_thresholds = [[(replacing_segments_featurevals[0][i]+replacing_segments_featurevals[0][i+1])/2 for i in range(0,len(replacing_segments_featurevals[0])-1)],[(replacing_segments_featurevals[1][i]+replacing_segments_featurevals[1][i+1])/2 for i in range(0,len(replacing_segments_featurevals[1])-1)]]
        current_segments_thresholds[best_threshold_segment[0]:best_threshold_segment[0]+1] = replacing_segments_thresholds[0],replacing_segments_thresholds[1] 
        current_segments_featurevals[best_threshold_segment[0]:best_threshold_segment[0]+1] = replacing_segments_featurevals[0],replacing_segments_featurevals[1] 
        current_segments_labels[best_threshold_segment[0]:best_threshold_segment[0]+1] = replacing_segments_labels[0],replacing_segments_labels[1] 
        # print('NEW',current_segments_thresholds)
        saved_thresholds.append(best_threshold)
        if len(saved_thresholds) == target_num_segments:
            return saved_thresholds
        else:
            return self.find_best_segmentation_binrec(target_num_segments,saved_thresholds,current_segments_labels,current_segments_thresholds,current_segments_featurevals)

    def return_segmentations(self,thresholds,num_segments,max_value):
        segmentations = []
        combis = combinations(thresholds,num_segments-1)
        for combi in combis:
            sorted_combi = sorted(combi)
            segmentation = [[0,sorted_combi[0]]]
            for bound in range(1,len(sorted_combi)):
                segmentation.append([sorted_combi[bound-1],sorted_combi[bound]])
            segmentation.append([sorted_combi[bound],max_value+1])
            segmentations.append(segmentation)
        return segmentations

    def calculate_entropy(self,labelfracs):
        # entropy = -sum([- prob * math.log(prob, 2) for prob in labelfracs if prob != 0])

        # if sum(i > 0 for i in labelfracs) == 1: # all instances in one category
        #     entropy=0
        # else:
        #     entropy = 0 if labelfracs[0] == 0 else -1*(labelfracs[0]*np.log2(labelfracs[0]))
        #     for frac in labelfracs[1:]:
        #         subtract = 0 if frac == 0 else frac * np.log2(frac)
        #         entropy -= subtract 
        #     #=-1*pH*np.log2(pH) - pC*np.log2(pC)
        return stats.entropy(labelfracs)

    def obtain_labelfracs(self,labels):
        labelfracs = []
        for label in self.unique_labels:
            c = labels.count(label)
            frac = c/len(labels) if c > 0 else 0 
            labelfracs.append(frac)
        return labelfracs

    def calculateIG(self,groups,labels):
        # current entropy
        labelfracs = self.obtain_labelfracs(labels)
        current_entropy = self.calculate_entropy(labelfracs)
        # entropy of each grouping
        group_entropy = []
        for group in groups:
            labelfracs = self.obtain_labelfracs(group)
            group_entropy.append((len(group)/len(labels)) * self.calculate_entropy(labelfracs))
        infogain = current_entropy - sum(group_entropy)
        return infogain
       
    def findSegmentationAndIG(self,instances,feature_index,labels):
        instances_column = instances[:,feature_index]
        feature_values = set(instances[:,feature_index])
        feature_values_sorted=sorted(feature_values)
        # print('Finding segmentation for',instances_column)
        candidate_thresholds=[]
        # print('Going through sorted feature values to have candidate thresholds')
        for i in range(0,len(feature_values_sorted)-1):
            candidate_thresholds.append((feature_values_sorted[i]+feature_values_sorted[i+1])/2)
        thresholds=list(set(candidate_thresholds))
        # print('Done. Candidate thresholds:',thresholds)

        # check which of the thresholds were already used 
        # if feature_index in self.used_thresholds:
        #     for used in self.used_thresholds[feature_index]:
        #         if used in candidate_thresholds:
        #             candidate_thresholds.remove(used)
        
        IG=[]
        # print('Now obtaining segmentations based on candidate thresholds')
        # segmentations = self.return_segmentations(thresholds,len(list(set(labels))),max(feature_values_sorted))
        # for i,segmentation in enumerate(segmentations):
        #     # print('Applying segmentation',i,'of',len(segmentations))
        #     groups = self.apply_segmentation(segmentation,instances_column,labels)
        #     labelgroups = []
        #     for group in groups:
        #         labelgroups.append([labels[i] for i in group])
        #     lgig = self.calculateIG(labelgroups,labels)
        #     # print('IG',lgig)
        #     IG.append(lgig)
        # maxIG=max(IG)
        try:
            best_segmentation = self.find_best_segmentation_binrec(len(list(set(labels)))-1,[],[labels],[thresholds],[list(instances_column)])

            best_segmentation_begin_end = [0] + sorted(best_segmentation) + [max(feature_values_sorted)+1]
            best_segmentation_formatted = [[best_segmentation_begin_end[i],best_segmentation_begin_end[i+1]] for i in range(len(best_segmentation_begin_end)-1)]
            groups = self.apply_segmentation(best_segmentation_formatted,list(instances_column))
            labelgroups=[]
            for group in groups:
                labelgroups.append([labels[i] for i in group])
                maxIG = self.calculateIG(labelgroups,labels) 
        except:
            best_segmentation_formatted = [0]
            maxIG = 0
        return best_segmentation_formatted,maxIG
            
    def construct_tree(self,selection,node_index):
            # print('node index is ',node_index)
            # print('TREE UPTO NOW:',self.Tree)
            labels = list(np.array(self.labels)[selection])
            label_instances = [labels.count(label) for label in self.unique_labels]
            # print(label_instances)
            if sum(i > 0 for i in label_instances) == 1: # all instances in one category
                self.Tree[node_index] = [False,False,sorted([[label,labels.count(label)] for label in self.unique_labels],key = lambda k : k[1],reverse=True)[0][0]]
                return
            
            # print('Generating segmentations')
            IGA=[]
            feature_segmentations=[]
            instances = self.train_instances[selection,:]
            for feature_index in range(0,self.train_instances.shape[1]):
                # print('generating segmentation for feature index',feature_index)
                segm,IG=self.findSegmentationAndIG(instances,feature_index,labels)
                # print('Done. segmentation:',segm,', Infogain:',IG)
                IGA.append(IG)
                feature_segmentations.append([feature_index,segm,IG])

            # print('All IGs:',IGA)
                
            best = sorted(feature_segmentations,key=lambda k : k[2],reverse=True)[0]
            maxIG = best[2]
            if maxIG == 0: # no improvement
                [False,False,sorted([[label,labels.count(label)] for label in self.unique_labels],key = lambda k : k[1],reverse=True)[0][0]]
                return 
            feature_index=best[0]
            # print('Best feature index is',feature_index)
            best_feature_segmentation=best[1]
            # print('Best feature segmentation is',best_feature_segmentation)
            # self.usedThresholds[feature_index].add(thresh)
            self.Tree[node_index]=[feature_index,best_feature_segmentation,sorted([[label,labels.count(label)] for label in self.unique_labels],key = lambda k : k[1],reverse=True)[0][0]]            
            # apply segmentation to construct new nodes
            # print('\nApplying segmentation')
            new_nodes_rows = []
            # print('BEST SEGMENTATION:',best_feature_segmentation)
            new_groups = self.apply_segmentation(best_feature_segmentation,instances[:,feature_index])
            # print('NEW GROUPS',new_groups)
            for i,group in enumerate(new_groups):
                # print('I',i)
                # print('GROUP',group)
                group_selection = [selection[i] for i in group]
                # print('GROUP_SELECTION',group_selection)
                # label_instances = list(np.array(labels)[group])
                # print('LABEL_INSTANCES',label_instances)
                # print('GROUP_INSTANCES',group_instances.shape)
                # print('Constructing tree with',2*node_index+i)
                self.construct_tree(group_selection,len(self.unique_labels)*node_index+i)
        
    def predict(self,selection,node_index):

        node = self.Tree[node_index]
        if node[0]: # not an end leaf 
            feature_index = node[0]
            instances_column = self.test_instances[selection,feature_index]
            segmentation = node[1]
            groups = self.apply_segmentation(segmentation,instances_column)
            # print('prediction node',node_index)
            # print('groups:',groups)
            if sum(len(g) > 0 for g in groups) == 1: # one group is filled
                predicted_label = node[2]
                for i in selection:
                    self.predictions[i] = predicted_label
                    # print('predicted instance',i)
            else:
                for i,group in enumerate(groups):
                    if len(group) == 0:
                        continue
                    else:
                        selection_group = [selection[i] for i in group]
                        # print('starting prediction for group',selection_group)
                        self.predict(selection_group,len(self.unique_labels)*node_index+i)
        else:
            predicted_label = node[2]
            for i in selection:
                self.predictions[i] = predicted_label
                # print('predicted instance',i)