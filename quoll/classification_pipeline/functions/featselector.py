#!/usr/bin/env python
# encoding: utf-8

import operator
from collections import defaultdict
import numpy as np
from scipy import sparse, stats

from quoll.classification_pipeline.functions import vectorizer

class FCBF:
    """
    Created by Prashant Shiralkar on 2015-02-06.
    Fast Correlation-Based Filter (FCBF) algorithm as described in 
    Feature Selection for High-Dimensional Data: A Fast Correlation-Based
    Filter Solution. Yu & Liu (ICML 2003)
    """

    def __init__(self):
        self.sbest = []

    def entropy(self, vec, base=2):
        " Returns the empirical entropy H(X) in the input vector."
        _, vec = np.unique(vec, return_counts=True)
        prob_vec = np.array(vec/float(sum(vec)))
        if base == 2:
            logfn = np.log2
        elif base == 10:
            logfn = np.log10
        else:
            logfn = np.log
        return prob_vec.dot(-logfn(prob_vec))

    def conditional_entropy(self, x, y):
        "Returns H(X|Y)."
        uy, uyc = np.unique(y, return_counts=True)
        prob_uyc = uyc/float(sum(uyc))
        cond_entropy_x = np.array([self.entropy(x[y == v]) for v in uy])
        return prob_uyc.dot(cond_entropy_x)
        
    def mutual_information(self, x, y):
        " Returns the information gain/mutual information [H(X)-H(X|Y)] between two random vars x & y."
        return self.entropy(x) - self.conditional_entropy(x, y)

    def symmetrical_uncertainty(self, x, y):
        " Returns 'symmetrical uncertainty' (SU) - a symmetric mutual information measure."
        return 2.0*self.mutual_information(x, y)/(self.entropy(x) + self.entropy(y))

    def getFirstElement(self, d):
        """
        Returns tuple corresponding to first 'unconsidered' feature
        
        Parameters:
        ----------
        d : ndarray
            A 2-d array with SU, original feature index and flag as columns.
        
        Returns:
        -------
        a, b, c : tuple
            a - SU value, b - original feature index, c - index of next 'unconsidered' feature
        """
        
        t = np.where(d[:,2]>0)[0]
        if len(t):
            return d[t[0],0], int(d[t[0],1]), t[0]
        return None, None, None

    def getNextElement(self, d, idx):
        """
        Returns tuple corresponding to the next 'unconsidered' feature.
        
        Parameters:
        -----------
        d : ndarray
            A 2-d array with SU, original feature index and flag as columns.
        idx : int
            Represents original index of a feature whose next element is required.
            
        Returns:
        --------
        a, b, c : tuple
            a - SU value, b - original feature index, c - index of next 'unconsidered' feature
        """
        t = np.where(d[:,2]>0)[0]
        t = t[t > idx]
        if len(t):
            return d[t[0],0], int(d[t[0],1]), t[0]
        return None, None, None
        
    def removeElement(self, d, idx):
        """
        Returns data with requested feature removed.
        
        Parameters:
        -----------
        d : ndarray
            A 2-d array with SU, original feature index and flag as columns.
        idx : int
            Represents original index of a feature which needs to be removed.
            
        Returns:
        --------
        d : ndarray
            Same as input, except with specific feature removed.
        """
        d[idx,2] = 0
        return d

    def c_correlation(self, X, y):
        """
        Returns SU values between each feature and class.
        
        Parameters:
        -----------
        X : 2-D ndarray
            Feature matrix.
        y : ndarray
            Class label vector
            
        Returns:
        --------
        su : ndarray
            Symmetric Uncertainty (SU) values for each feature.
        """
        su = np.zeros(X.shape[1])
        for i in np.arange(X.shape[1]):
            su[i] = self.symmetrical_uncertainty(X[:,i], y)
        return su

    def fit(self, X, y, thresh):
        """
        Perform Fast Correlation-Based Filter solution (FCBF).
        
        Parameters:
        -----------
        X : 2-D ndarray
            Feature matrix
        y : ndarray
            Class label vector
        thresh : float
            A value in [0,1) used as threshold for selecting 'relevant' features. 
            A negative value suggest the use of minimum SU[i,c] value as threshold.
        
        Returns:
        --------
        sbest : 2-D ndarray
            An array containing SU[i,c] values and feature index i.
        """
        thresh = float.thresh
        n = X.shape[1]
        slist = np.zeros((n, 3))
        slist[:, -1] = 1

        # identify relevant features
        slist[:,0] = self.c_correlation(X, y) # compute 'C-correlation'
        idx = slist[:,0].argsort()[::-1]
        slist = slist[idx, ]
        slist[:,1] = idx
        if thresh < 0:
            thresh = np.median(slist[-1,0])
            print('Using minimum SU value as default threshold: {0}'.format(thresh))
        elif thresh >= 1 or thresh > max(slist[:,0]):
            print('No relevant features selected for given threshold.')
            print('Please lower the threshold and try again.')
            exit()
            
        slist = slist[slist[:,0]>thresh,:] # desc. ordered per SU[i,c]
        
        # identify redundant features among the relevant ones
        cache = {}
        m = len(slist)
        p_su, p, p_idx = self.getFirstElement(slist)
        for i in range(m):
            q_su, q, q_idx = self.getNextElement(slist, p_idx)
            if q:
                while q:
                    if (p, q) in cache:
                        pq_su = cache[(p,q)]
                    else:
                        
                        pq_su = self.symmetrical_uncertainty(X[:,p], X[:,q])
                        cache[(p,q)] = pq_su

                    if pq_su >= q_su:
                        slist = self.removeElement(slist, q_idx)
                    q_su, q, q_idx = self.getNextElement(slist, q_idx)
                    
            p_su, p, p_idx = self.getNextElement(slist, p_idx)
            if not p_idx:
                break

        self.sbest = slist[slist[:,2]>0, :2]
        self.indices = [int(x[1]) for x in self.sbest]
        self.sus = [round(x[0],2) for x in self.sbest]

    def transform(self, X):
        return vectorizer.compress_vectors(X,self.indices)

    def fit_transform(self, X, y, threshold):
        self.fit(X.toarray(), y, threshold)
        filtered_X = self.transform(X)
        return filtered_X, self.sus, self.indices


class MRMRLinear:

    def __init__(self):
        self.feature_strength = []
        self.feature_correlation = []

    def rank_relevance(self,X,y):
        # calculate correlation by feature
        feature_strength = {}
        for i in range(X.shape[1]):
            feature_vals = X[:,i].transpose().toarray()[0]
            corr,p = stats.pearsonr(feature_vals,y)
            feature_strength[i] = abs(corr)
        self.feature_strength = feature_strength

    def compute_correlations(self,X):
        # calculate correlation by feature
        correlations = defaultdict(list)
        for i in range(X.shape[1]):
            feature_vals_i = X[:,i].transpose().toarray()[0]
            for j in range(X.shape[1]):
                feature_vals_j = X[:,j].transpose().toarray()[0]
                corr,p = stats.pearsonr(feature_vals_i,feature_vals_j)
                correlations[i].append([j,abs(corr),corr,p])
        self.feature_correlation = correlations

    def summarize_feature_weights(self):
        feature_weights = []
        for i in range(len(self.feature_strength.keys())):
            strength = self.feature_strength[i]
            correlations = [x[1] for x in self.feature_correlation[i]]
            feature_weights.append([str(x) for x in [strength] + correlations])
        return feature_weights

    def filter_features(self, strength_threshold, correlation_threshold):
        selected_features = []
        discarded = []
        print('Starting feature filter with strength threshold ' + str(strength_threshold) + ' and correlation_threshold ' + str(correlation_threshold))
        sorted_keys = [kv[0] for kv in sorted(self.feature_strength.items(), key=operator.itemgetter(1), reverse=True)]
        report=[]
        for feature_index in sorted_keys:
            report.append('Inspecting feature ' + str(feature_index))
            if not feature_index in discarded:
                strength = self.feature_strength[feature_index]
                report.append('Strength = ' + str(strength))
                if (strength_threshold > 1.0 and len(selected_features) < strength_threshold) or (strength_threshold <= 1.0 and strength > strength_threshold):
                    report.append('Threshold met, adding to selected features.')
                    selected_features.append(feature_index)
                    correlating_features = [feature[0] for feature in self.feature_correlation[feature_index] if feature[1] > correlation_threshold]
                    report.append('Correlating features: ' + ', '.join([str(x) for x in correlating_features]) + '; deleting...')
                    discarded.extend(correlating_features)
                    report.append('Current set of discarded features: ' + ', '.join([str(f) for f in discarded]))
                else:
                    report.append('Feature did not meet threshold, filtering done')
                    break
            else:
                report.append('Feature was discarded, moving on to next feature')
        return selected_features

    def fit(self, X, y, thresh):
        """
        Perform filtering
        
        Parameters:
        -----------
        X : 2-D ndarray
            Feature matrix
        y : ndarray
            Class label vector
        thresh : float
            A value in [0,1) used as threshold for selecting 'relevant' features. 
            A negative value suggest the use of minimum SU[i,c] value as threshold.
        
        Returns:
        --------
        sbest : 2-D ndarray
            An array containing SU[i,c] values and feature index i.
        """
        y = [float(x) for x in y]
        self.rank_relevance(X,y)
        self.compute_correlations(X)
        strength_threshold = float(thresh.split('_')[0])
        correlation_threshold = float(thresh.split('_')[1])
        self.indices = self.filter_features(strength_threshold,correlation_threshold)
        print('Selected features:',self.indices)

    def transform(self, X):
        return vectorizer.compress_vectors(X,self.indices)

    def fit_transform(self, X, y, threshold):
        self.fit(X, y, threshold)
        filtered_X = self.transform(X)
        return filtered_X, self.summarize_feature_weights(), self.indices
