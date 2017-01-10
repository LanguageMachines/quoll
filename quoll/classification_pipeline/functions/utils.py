#!/usr/bin/env

import re
import os
import datetime
import functools
from collections import Counter
from collections import defaultdict
import numpy as np

def return_folds(instances, split_user = False, n = 10):
    folds = []
    if split_user:
        print('usersplit')
        userindex = 0
        user_instances = defaultdict(list)
        for i, user in enumerate(instances):
            if user == '-':
                user_instances[str(userindex)].append(i)
                userindex += 1
            else:
                user_instances[user].append(i)
        for i in range(n):
            j = i
            fold = []
            users = list(user_instances.keys())
            while j < len(users):
                fold.extend(user_instances[users[j]])
                j += n
            folds.append(fold)
    else:
        num_instances = len(instances)
        for i in range(n):
            j = i
            fold = []
            while j < num_instances:
                fold.append(j)
                j += n
            folds.append(fold)
    runs = []
    foldindices = range(n)
    for run in foldindices:
        testindex = run
        trainindices = list(set(foldindices) - set([testindex]))
        trainfolds = [folds[i] for i in trainindices]
        train = functools.reduce(lambda y, z: y+z, trainfolds)
        test = folds[testindex]
        runs.append([train, test])
    return runs

def find_ngrams(input_list, n):
    """
    Calculate n-grams from a list of tokens/characters with added begin and end
    items. Based on the implementation by Scott Triglia
    http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
    """
    for x in range(n-1):
        input_list.insert(0, '')
        input_list.append('')
    return zip(*[input_list[i:] for i in range(n)])

def freq_dict(text):
    """
    Returns a frequency dictionary of the input list
    """
    c = Counter()
    for word in text:
        c[word] += 1
    return c

def format_table(data, widths):
    output = []
    for row in data:
        try:
            output.append("".join("%-*s" % i for i in zip(widths, row)))
        except:
            print('Format table: number of width values and row values do not correspond, using tabs instead')
            output.append('\t'.join(row))
    return output

def save_sparse_csr(filename, array):
    np.savez(filename, data = array.data, indices = array.indices,
             indptr = array.indptr, shape = array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def write_lcs_config(savedir, ts, lts, prune):
    expdir = os.getcwd() + '/'
    files = savedir + './files/'
    data = expdir + './data'
    index = expdir + './index'
    config = '\n'.join\
        ([
        'docprof.normalise=NONE',
        'general.analyser=nl.cs.ru.phasar.lcs3.analyzers.FreqAnalyzer',
        'general.autothreshold=true',
        'general.data=' + data,
        'general.files=' + files,
        'general.index=' + index,
        'general.numcpus=16',
        'general.termstrength=' + ts, # hier een parameter
        'gts.mindf=1',
        'gts.mintf=6',
        'lts.algorithm=' + lts, # parameter
        'lts.maxterms=' + prune,
        'profile.memory=false',
        'research.fullconfusion=false',
        'research.writemit=true',
        'research.writemitalliters=false',
        'general.algorithm=WINNOW',
        'general.docext=',
        'general.fbeta=1.0',
        'general.fullranking=true',
        'general.maxranks=1',
        'general.minranks=1',
        'general.preprocessor=',
        'general.rankalliters=false',
        'general.saveclassprofiles=true',
        'general.threshold=1.0',
        'general.writetestrank=true',
        'gts.maxdf=1000000',
        'gts.maxtf=1000000',
        'lts.aggregated=true',
        'naivebayes.smoothing=1.0',
        'positivenaivebayes.classprobability=0.2',
        'regwinnow.complexity=0.1',
        'regwinnow.initialweight=0.1',
        'regwinnow.iterations=10',
        'regwinnow.learningrate=0.01',
        'regwinnow.ownthreshold=true',
        'research.conservememory=true',
        'research.mitsortorder=MASS',
        'rocchio.beta=1.0',
        'rocchio.gamma=1.0',
        'svmlight.params=',
        'winnow.alpha=1.05',
        'winnow.beta=0.95',
        'winnow.beta.twominusalpha=false',
        'winnow.decreasing.alpha=false',
        'winnow.decreasing.alpha.strategy=LOGARITMIC',
        'winnow.maxiters=3',
        'winnow.negativeweights=true',
        'winnow.seed=-1',
        'winnow.termselect=false',
        'winnow.termselect.epsilon=1.0E-4',
        'winnow.termselect.iterations=1,2,',
        'winnow.thetamin=0.5',
        'winnow.thetaplus=2.5'
        ])
    with open(savedir + 'lcs3.conf', 'w', encoding = 'utf-8') as config_out:
        config_out.write(config)

def tokenized_2_tagged(tokenized_texts):
    tagged = []
    for text in tokenized_texts:
        tagged_line = [[token, '-', '-', '-'] for token in text.split()]
        tagged.append(tagged_line)
    return tagged
