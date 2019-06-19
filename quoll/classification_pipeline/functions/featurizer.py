#!/usr/bin/env

import os
import numpy as np
from scipy import sparse
import operator
import re
import multiprocessing
from collections import Counter

import colibricore

#####################################################
### Helpers ####
#####################################################

def extract_tokentype(docs, tokentype):
    documents_tokentype = []
    for document in docs:
        documents_tokentype.append([[token[tokentype] for token in sentence] for sentence in document['processed']])
    return documents_tokentype


#####################################################
### Featurizer ####
#####################################################

class Featurizer:
    """
    Featurizer
    =====
    Class to extract features from raw and tagged text files

    Parameters
    -----
    instances : list
        The instances come in an array where each entry represents a text instance
            for token n-grams, lemma n-grams or pos-ngrams, make sure units are split by whitespace
            for character ngrams, input the original text
    features : dict
        Subset any of the entries in the following dictionary:

        features = {
            'simple_stats' : {},
            'token_ngrams' : {'n_list' : [1, 2, 3]},
            'char_ngrams' : {'n_list' : [1, 2, 3]},
            'lemma_ngrams' : {'n_list' : [1, 2, 3]},
            'pos_ngrams' : {'n_list' : [1, 2, 3]}
        }

    Attributes
    -----
    self.instances : list
        The instances parameter
    self.modules : dict
        Template dict with all the helper classes
    self.helpers : list
        List to call all the helper classes
    self.features : dict
        Container of the featurized instances for the different feature types
    self.vocabularies : dict
        Container of the name of each feature index for the different feature types
    """
    def __init__(self, documents, features):
        self.documents = documents
        self.modules = {
            'chars'   :           CharNgrams,
            'tokens'  :           TokenNgrams,
            'lemmas'  :           LemmaNgrams,
            'pos'     :           PosNgrams
        }
        self.helpers = [v(**features[k]) for k, v in self.modules.items() if k in features.keys()]
        self.features = {}
        self.vocabularies = {}

    def fit_transform(self):
        """
        Featurizer
        =====
        Function to extract every helper feature type

        Transforms
        -----
        self.features : dict
            The featurized instances of every helper are written to this dict
        self.vocabularies : dict
            The name of each feature index for every feature type is written to this dict
        """
        for helper in self.helpers:
            helperfeatures, helpervocabulary = helper.fit_transform(self.documents)
            self.features[helper.name] = helperfeatures
            self.vocabularies[helper.name] = helpervocabulary

    def return_instances(self, helpernames):
        """
        Information extractor
        =====
        Function to extract featurized instances in any combination of feature types

        Parameters
        ------
        helpernames : list
            List of the feature types to combine
            Names of feature types correspond with the keys of self.modules

        Returns
        -----
        instances : scipy csr matrix
            Featurized instances
        Vocabulary : list
            List with the feature name per index
        """
        submatrices = [self.features[name] for name in helpernames]
        instances = sparse.hstack(submatrices).tocsr()
        vocabulary = np.hstack([self.vocabularies[name] for name in helpernames])
        return instances, vocabulary

class CocoNgrams:

    def __init__(self, ngrams, blackfeats):
        self.ngrams = ngrams
        self.blackfeats = set(blackfeats)
        self.documents = False
        self.classdecoder = False
        self.model = False
        self.cursor = {}

    def fit(self, tmpdir, documents, mt):
        self.documents = documents
        coco_file = tmpdir + 'instances.txt'
        linecursor = 0
        with open(coco_file, 'w', encoding = 'utf-8') as txt:
            for doccursor,doc in enumerate(documents):
                indices = []
                for sentence in doc:
                    indices.append(linecursor)
                    linecursor+=1
                    txt.write(' '.join(sentence) + '\n')
                for index in indices:
                    self.cursor[index] = doccursor
        classfile = tmpdir + 'instances.colibri.cls'
        # Build class encoder
        classencoder = colibricore.ClassEncoder()
        classencoder.build(coco_file)
        classencoder.save(classfile)

        # Encode corpus data
        corpusfile = tmpdir + 'instances.colibri.dat'
        classencoder.encodefile(coco_file, corpusfile)

        # Load class decoder
        self.classdecoder = colibricore.ClassDecoder(classfile)

        # Train model
        options = colibricore.PatternModelOptions(mintokens = mt, minlength = min(self.ngrams), maxlength = max(self.ngrams), doreverseindex=True)
        self.model = colibricore.IndexedPatternModel()
        self.model.train(corpusfile, options)
        
    def transform(self, write = False):
        rows = []
        cols = []
        data = []
        vocabulary = []
        items = list(zip(range(self.model.__len__()), self.model.items()))
        for i, (pattern, indices) in items:
            if len(pattern.tolist()) in self.ngrams:
                vocabulary.append(pattern.tostring(self.classdecoder))
                lines = [index[0] - 1 for index in indices]
                docs = [self.cursor[lineindex] for lineindex in lines]
                counts = Counter(docs)
                unique = counts.keys()
                rows.extend(unique)
                cols.extend([i] * len(unique))
                data.extend(counts.values())
        if write:
            with open(write, 'w', encoding = 'utf-8') as sparse_out:
                sparse_out.write(' '.join([str(x) for x in data]) + '\n')
                sparse_out.write(' '.join([str(x) for x in rows]) + '\n')
                sparse_out.write(' '.join([str(x) for x in cols]) + '\n')
                sparse_out.write(str(len(self.documents)) + ' ' + str(self.model.__len__()))
        instances = sparse.csr_matrix((data, (rows, cols)), shape = (len(self.documents), self.model.__len__()))
        if len(self.blackfeats) > 0:
            blackfeats_indices = []
            for bf in self.blackfeats:
                regex = re.compile(bf)
                matches = [i for i, f in enumerate(vocabulary) if regex.match(f)]
                regex_left = re.compile(r'.+' + ' ' + bf + r'$')
                matches += [i for i, f in enumerate(vocabulary) if regex_left.match(f)]
                regex_right = re.compile(r'^' + bf + ' ' + r' .+')
                matches += [i for i, f in enumerate(vocabulary) if regex_right.match(f)]
                regex_middle = re.compile(r'.+' + ' ' + bf + ' ' + r'.+')
                matches += [i for i, f in enumerate(vocabulary) if regex_middle.match(f)]
                blackfeats_indices.extend(matches)
            to_keep = list(set(range(len(vocabulary))) - set(blackfeats_indices))
            instances = sparse.csr_matrix(instances[:, to_keep])
            vocabulary = list(np.array(vocabulary)[to_keep])
        return instances, vocabulary

class TokenNgrams(CocoNgrams):
    """
    Token ngram extractor
    =====
    Class to extract token ngrams from all documents

    Parameters
    -----
    kwargs : dict
        n_list : list
            The values of N (1 - ...)
        blackfeats : list
            Features to exclude

    Attributes
    -----
    self.name : str
        The name of the module
    self.n_list : list
        The n_list parameter
    self.blackfeats : list
        The blackfeats parameter
    self.features : list
        List of feature names, to keep track of the values of feature indices
    """
    def __init__(self, **kwargs):
        self.name = 'tokens'
        self.n_list = [int(x) for x in kwargs['n_list']]
        if 'blackfeats' in kwargs.keys():
            self.blackfeats = kwargs['blackfeats']
        else:
            self.blackfeats = []
        self.mt = kwargs['mt']
        CocoNgrams.__init__(self, self.n_list, self.blackfeats)
        self.features = []

    def fit(self, documents):
        """
        Model fitter
        =====
        Function to make an overview of all the existing features

        """
        documents_text = extract_tokentype(documents,'text')
        tmpdir = os.getcwd() + '/tmp/'
        if not os.path.isdir(tmpdir):
            os.mkdir(tmpdir)
        CocoNgrams.fit(self, tmpdir, documents_text, self.mt)

    def transform(self):
        """
        Model transformer
        =====
        Function to featurize instances based on the fitted features

        """
        instances, features = CocoNgrams.transform(self)
        return(instances, features)

    def fit_transform(self, documents):
        """
        Fit transform
        =====
        Function to perform the fit and transform sequence

        Parameters
        -----

        Returns
        -----
        self.transform(tagged_data) : list
            The featurized instances
        self.features : list
            The vocabulary
        """
        self.fit(documents)
        return self.transform()


class LemmaNgrams(CocoNgrams):
    """
    Lemma ngram extractor
    =====
    Class to extract lemma ngrams from all documents

    Parameters
    -----
    kwargs : dict
        n_list : list
            The values of N (1 - ...)
        blackfeats : list
            Features to exclude

    Attributes
    -----
    self.name : str
        The name of the module
    self.n_list : list
        The n_list parameter
    self.blackfeats : list
        The blackfeats parameter
    self.features : list
        List of feature names, to keep track of the values of feature indices
    """
    def __init__(self, **kwargs):
        self.name = 'lemmas'
        self.n_list = [int(x) for x in kwargs['n_list']]
        if 'blackfeats' in kwargs.keys():
            self.blackfeats = kwargs['blackfeats']
        else:
            self.blackfeats = []
        self.mt = kwargs['mt']
        CocoNgrams.__init__(self, self.n_list, self.blackfeats)
        self.features = []

    def fit(self, documents):
        """
        Model fitter
        =====
        Function to make an overview of all the existing features

        """
        documents_lemmas = extract_tokentype(documents, 'lemma')
        tmpdir = os.getcwd() + '/tmp/'
        if not os.path.isdir(tmpdir):
            os.mkdir(tmpdir)
        CocoNgrams.fit(self, tmpdir, documents_lemmas, self.mt)

    def transform(self):
        """
        Model transformer
        =====
        Function to featurize instances based on the fitted features

        """
        instances, features = CocoNgrams.transform(self)
        return(instances, features)

    def fit_transform(self, documents):
        """
        Fit transform
        =====
        Function to perform the fit and transform sequence

        Parameters
        -----

        Returns
        -----
        self.transform(tagged_data) : list
            The featurized instances
        self.features : list
            The vocabulary
        """
        self.fit(documents)
        return self.transform()

class PosNgrams(CocoNgrams):
    """
    POS ngram extractor
    =====
    Class to extract lemma ngrams from all documents

    Parameters
    -----
    kwargs : dict
        n_list : list
            The values of N (1 - ...)
        blackfeats : list
            Features to exclude

    Attributes
    -----
    self.name : str
        The name of the module
    self.n_list : list
        The n_list parameter
    self.blackfeats : list
        The blackfeats parameter
    self.features : list
        List of feature names, to keep track of the values of feature indices
    """
    def __init__(self, **kwargs):
        self.name = 'pos'
        self.n_list = [int(x) for x in kwargs['n_list']]
        if 'blackfeats' in kwargs.keys():
            self.blackfeats = kwargs['blackfeats']
        else:
            self.blackfeats = []
        self.mt = kwargs['mt']
        CocoNgrams.__init__(self, self.n_list, self.blackfeats)
        self.features = []

    def fit(self, documents):
        """
        Model fitter
        =====
        Function to make an overview of all the existing features

        """
        documents_pos = extract_tokentype(documents, 'pos')
        tmpdir = os.getcwd() + '/tmp/'
        if not os.path.isdir(tmpdir):
            os.mkdir(tmpdir)
        CocoNgrams.fit(self, tmpdir, documents_pos, self.mt)

    def transform(self):
        """
        Model transformer
        =====
        Function to featurize instances based on the fitted features

        """
        instances, features = CocoNgrams.transform(self)
        return(instances, features)

    def fit_transform(self, documents):
        """
        Fit transform
        =====
        Function to perform the fit and transform sequence

        Parameters
        -----

        Returns
        -----
        self.transform(tagged_data) : list
            The featurized instances
        self.features : list
            The vocabulary
        """
        self.fit(documents)
        return self.transform()

class CharNgrams:
    """
    Character ngram extractor
    =====
    Class to extract character ngrams from all documents

    Parameters
    -----
    kwargs : dict
        n_list : list
            The values of N (1 - ...)
        blackfeats : list
            Features to exclude

    Attributes
    -----
    self.name : str
        The name of the module
    self.n_list : list
        The n_list parameter
    self.blackfeats : list
        The blackfeats parameter
    self.features : list
        List of feature names, to keep track of the values of feature indices
    """
    def __init__(self, **kwargs):
        self.name = 'chars'
        self.n_list = [int(x) for x in kwargs['n_list']]
        self.mt = kwargs['mt']
        if 'blackfeats' in kwargs.keys():
            self.blackfeats = kwargs['blackfeats']
        else:
            self.blackfeats = []
        self.features = []

    def freq_dict(self, text):
        """
        Returns a frequency dictionary of the input list
        """
        c = Counter()
        for word in text:
            c[word] += 1
        return c

    def find_ngrams(self,input_list, n):
        """
        Calculate n-grams from a list of tokens/characters with added begin and end
        items. Based on the implementation by Scott Triglia
        http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
        """
        for x in range(n-1):
            input_list.insert(0, '')
            input_list.append('')
        return zip(*[input_list[i:] for i in range(n)])
    
    def fit(self, documents):
        """
        Model fitter
        =====
        Function to make an overview of all the existing features

        Parameters
        -----
        documents : list
            Each entry represents a text instance in the data file

        Attributes
        -----
        features : dict
            dictionary of features and their count
        """
        features = {}
        for document in documents:
            document = list(document['raw'])
            for n in self.n_list:
                features.update(self.freq_dict([''.join(item) for item in self.find_ngrams(document, n)]))
        self.features = [i for i,j in sorted(features.items(), reverse=True, key=operator.itemgetter(1)) if not bool(set(i.split("_")) & set(self.blackfeats)) and not (i == '') and j >= self.mt]
        
    def transform(self, documents):
        """
        Model transformer
        =====
        Function to featurize instances based on the fitted features

        Parameters
        -----
        documents : list
            Each entry represents a text instance in the data file

        Attributes
        -----
        instances : list
            The documents represented as feature vectors
        """
        instances = []
        for document in documents:
            document = list(document['raw'])
            char_dict = {}
            for n in self.n_list:
                char_dict.update(self.freq_dict([''.join(item) for item in self.find_ngrams(document, n)]))
            instances.append([char_dict.get(f,0) for f in self.features])
        return sparse.csr_matrix(instances)

    def fit_transform(self, documents):
        """
        Fit transform
        =====
        Function to perform the fit and transform sequence

        Parameters
        -----
        documents : list
            Each entry represents a text instance in the data file

        Returns
        -----
        self.transform(tagged_data) : list
            The featurized instances
        self.features : list
            The vocabulary
        """
        self.fit(documents)
        return self.transform(documents), self.features
