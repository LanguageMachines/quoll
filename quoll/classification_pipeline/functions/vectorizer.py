
#!/usr/bin/env

from collections import Counter
import math
import random
import numpy
import copy
import operator
from scipy import sparse, stats
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler

class Counts:
    """
    Counter
    =====
    Function to perform general count operations on featurized instances
    Used as parent class in several classes

    Parameters
    ------
    Instances : list
        list of featurized instances, as list with feature frequencies
    Labels : list
        list of labels (str) of the instances,
        each index of a label corresponds to the index of the instance
    """
    def __init__(self, instances, labels):
        self.instances = instances
        self.labels = labels

    def count_document_frequency(self, label = False):
        """
        Feature counter
        =====
        Function to return document counts of all features

        Parameters
        -----
        label : str
            Choose to count the frequency that each feature co-occurs with the given label
            If False, the total document count is returned

        Returns
        -----
        document_frequency : Counter
            Counts of the number of documents or labels with which a feature occurs
            key : The feature index (int)
            value : The document / label count of the feature index (int)
        """
        if label:
            target_instances = self.instances[list(numpy.where(numpy.array(self.labels) == label)[0])]
        else:
            target_instances = self.instances
        feature_indices = range(self.instances.shape[1])
        feature_counts = target_instances.sum(axis = 0).tolist()[0]
        document_frequency = dict(zip(feature_indices, feature_counts))
        return document_frequency

    def count_label_frequency(self):
        """
        Label counter
        =====
        Function to return counts of all document labels

        Returns
        -----
        label_frequency : dict
            Counts of each label
            key : The label (str)
            value : The count of the label (int)
        """
        label_frequency = {}
        for label in set(self.labels):
            label_frequency[label] = self.labels.count(label)
        return label_frequency

    def count_label_feature_frequency(self):
        """
        Frequency calculator
        =====
        Function to calculate the frequency of each feature in combination with specific labels

        Parameters
        -----
        labels : list
            list of labels (str) of the train instances

        Returns
        -----
        label_feature_frequency : dict of dicts
            key1 : label, str
            key2 : feature index, int
            value : number of times the two co-occur on the document level, list
        """
        label_feature_frequency = {}
        for label in self.labels:
            label_feature_frequency[label] = self.count_document_frequency(label)
        return label_feature_frequency

    def count_idf(self):
        """
        Inverse Document Frequency counter
        =====
        Function to calculate the inverse document frequency of every feature

        Returns
        -----
        idf : dict
            The idf of every feature based on the training documents
            key : The feature index
            value : The idf of the feature index
        """
        idf = dict.fromkeys(range(self.instances.shape[1]), 0) # initialize for all features
        num_docs = self.instances.shape[0]
        feature_counts = self.count_document_frequency()
        for feature in feature_counts.keys():
            idf[feature] = math.log((num_docs / feature_counts[feature]), 10) if feature_counts[feature] > 0 else 0
        return idf

def balance_data(instances, labels):
    # identify lowest frequency]
    unique_labels = list(set(labels))
    label_count_sorted = sorted([(label,labels.count(label)) for label in unique_labels], key = lambda k : k[1])
    least_frequent_indices = [i for i,label in enumerate(labels) if label == label_count_sorted[0][0]]
    least_frequent_count = label_count_sorted[0][1]
    balanced_instances = instances[least_frequent_indices,:]
    balanced_labels = [label_count_sorted[0][0]] * least_frequent_count
    # impose lowest frequency on other labels
    for cursorlabel in [lc[0] for lc in label_count_sorted[1:]]:
        label_indices = [i for i,label in enumerate(labels) if label == cursorlabel]
        samples = random.sample(label_indices, least_frequent_count)
        sampled_instances = instances[samples,:]
        balanced_instances = sparse.vstack((balanced_instances,sampled_instances), format='csr')
        balanced_labels.extend([cursorlabel] * least_frequent_count)
    return balanced_instances, balanced_labels

def return_document_frequency(instances, labels):

    cnt = Counts(instances, labels)
    document_frequency = cnt.count_document_frequency()
    return document_frequency

def return_idf(instances, labels):

    transformer = TfidfTransformer(smooth_idf=True)
    transformer.fit(instances)
    idf = dict.fromkeys(range(instances.shape[1]), 0)
    for feature,value in enumerate(list(transformer._idf_diag.data)):
        idf[feature] = value
    return idf
                            
def return_binary_vectors(instances, feature_weights):
    
    binary_values = numpy.array([1 for cell in instances.data])
    binary_vectors = sparse.csr_matrix((binary_values, instances.indices, instances.indptr), shape = instances.shape)
    return binary_vectors

def return_tfidf_vectors(instances, idfs):

    feature_idf_ordered = sparse.csr_matrix([idfs[feature] for feature in sorted(idfs.keys())])
    tfidf_vectors = instances.multiply(feature_idf_ordered)
    return tfidf_vectors

def return_featureselection(featureweights,prune):
    # by default pruning is done based on feature frequency    
    featureselection = sorted(featureweights, key = featureweights.get, reverse = True)[:prune]
    return featureselection

def compress_vectors(instances, featureselection):

    compressed_vectors = instances[:, featureselection]
    return compressed_vectors

def align_vectors(instances, target_vocabulary, source_vocabulary):

    source_feature_indices = dict([(feature, i) for i, feature in enumerate(source_vocabulary)])
    indices = []
    columns = []
    num_instances = instances.shape[0]

    for feature in target_vocabulary:
        try:
            indices.append(source_feature_indices[feature])
        except: # feature not in source vocabulary
            if len(indices) > 0:
                columns.append(compress_vectors(instances,indices))
                indices = []
            columns.append(sparse.csr_matrix([[0]] * num_instances))
    if len(indices) > 0:
        columns.append(compress_vectors(instances,indices))
    aligned_vectors = sparse.hstack(columns).tocsr()
    return aligned_vectors

def normalize_features(instances):

    normalized = normalize(instances,norm='l1')
    return normalized

def fit_scale(vectors):
    scaler = StandardScaler()
    scaler.fit(vectors)
    return scaler

def scale_vectors(vectors,scale):
    return scaler.transform(vectors)
