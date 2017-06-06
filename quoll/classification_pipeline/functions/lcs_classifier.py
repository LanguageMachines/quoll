#!/usr/bin/env 

import os
import numpy

from quoll.classification_pipeline.functions import nfold_cv_functions

class LCS_classifier:

    def __init__(self, directory):
        self.expdir = directory
        self.filesdir = False
        self.classifications = []
    
    def experiment(self,traininstances,trainlabels,vocabulary,testinstances=False,testlabels=False):
        self.filesdir = self.expdir + 'files/'
        try:
            os.mkdir(self.filesdir)
        except:
            print('filesdirectory already exists')
        # try:
        if not os.path.exists('train'):
            trainparts = self.prepare(traininstances,trainlabels,vocabulary, 'train/')
            with open('train', 'w', encoding = 'utf-8') as train:
                train.write('\n'.join(trainparts))            
        else:
            print('Trainfile already exists, skipping data preparation')
        if not os.path.exists('test'):
            testparts = self.prepare(testinstances,testlabels,vocabulary, 'test/')
            with open('test', 'w', encoding = 'utf-8') as test:
                test.write('\n'.join(testparts))
        else:
            print('Testfile already exists, skipping data preparation')
        self.classify(self.expdir)
        # except: 
        #     print('Running 10-fold cross-validation')
        #     if not os.path.exists('parts'):
        #         parts = self.prepare(traininstances,trainlabels,vocabulary)
        #         with open('parts','w',encoding='utf-8') as parts_out:
        #             parts_out.write('\n'.join(parts))
        #     else:
        #         with open('parts','r',encoding='utf-8') as parts_in:
        #             parts = parts_in.read().strip().split('\n')
        #         print('Partsfile already exists, skipping data preparation')
        #     # perform tenfold on train
        #     folds = nfold_cv_functions.return_fold_indices(len(parts),10,1)
        #     for i, fold in enumerate(folds):
        #         expdir = self.expdir + 'fold_' + str(i) + '/'
        #         try:
        #             os.mkdir(expdir)
        #         except:
        #             print('folddirectory already exists')
        #         numparts = numpy.array(parts)
        #         trainparts = list([numparts[indices] for j,indices in enumerate(folds) if j != i])
        #         testparts = list(numparts[fold])
        #         with open('train','w',encoding='utf-8') as train_out:
        #             train_out.write('\n'.join(trainparts))
        #         with open('test','w',encoding='utf-8') as test_out:
        #             test_out.write('\n'.join(testparts))
        #         self.classify(expdir)

    def instances_2_ngrams(self,instances,vocabulary):
        instances_ngrams = []
        vocab_numpy = numpy.array(vocabulary)
        indices = instances.indices
        indptr = instances.indptr
        for i in range(instances.shape[0]):
            instances_ngrams.append(list(vocab_numpy[indices[indptr[i]:indptr[i+1]]]))
        return instances_ngrams

    def prepare(self, instances, labels, vocabulary, add_dir=False):
        parts = []
        # transform instances from vectors to vocabularylists
        instances_vocabulary = self.instances_2_ngrams(instances,vocabulary)
        # make directory to write files to
        # make added directory
        if add_dir:
            add = add_dir
            try:
                os.mkdir(self.filesdir + add)
            except:
                print('added dir already exists')
        else:
            add = ''
        # make chunks of 25000 from the data
        data = list(zip(labels,instances_vocabulary))
        print('\n'.join(data[0][1]).encode('utf-8'))
        if len(data) > 25000:
            chunks = [list(t) for t in zip(*[iter(data)]*25000)]
        else:
            chunks = [data]
        for i, chunk in enumerate(chunks):
            # make subdirectory
            subpart = add + 'sd' + str(i) + '/'
            subdir = self.filesdir + subpart
            try:
                os.mkdir(subdir)
            except:
                print('subdirectory already exists')
            for j, instance in enumerate(chunk):
                zeros = 5 - len(str(j))
                filename = subpart + ('0' * zeros) + str(j) + '.txt'
                label = instance[0]
                features = instance[1]
                with open(self.filesdir + filename, 'w', encoding = 'utf-8') as outfile: 
                    outfile.write('\n'.join(features))
                parts.append(filename + ' ' + label)
        return parts

    def classify(self,expdir):
        self.write_config()
        os.system('lcs --verbose')
        self.extract_performance()
        os.system('mv train ' + expdir)
        os.system('mv data/ ' + expdir)
        os.system('mv test* ' + expdir)
        os.system('mv lcs* ' + expdir)
        os.system('mv index/ ' + expdir)

    def extract_performance(self):
        with open('test.rnk') as rnk:
            for line in rnk.readlines():
                tokens = line.strip().split()
                filename = tokens[0].strip()
                classification, score = tokens[1].split()[0].split(':')
                classification = classification.replace('?','')
                self.classifications.append([classification, score])

    def write_config(self):
        fileschunks = self.filesdir.split('/')
        files = '/'.join(fileschunks[:-1]) + '/./' + fileschunks[-1]
        current = os.getcwd()
        current_chunks = current.split('/')
        data = '/'.join(current_chunks) + '/./data'
        index = '/'.join(current_chunks) + '/./index'
        config = '\n'.join\
            ([
            'docprof.normalise=NONE',
            'general.analyser=nl.cs.ru.phasar.lcs3.analyzers.FreqAnalyzer',
            'general.autothreshold=true',
            'general.data=' + data,
            'general.files=' + files,
            'general.index=' + index,
            'general.numcpus=16',
            'general.termstrength=BOOL', # hier een parameter
            'gts.mindf=1',
            'gts.mintf=6',
            'lts.algorithm=INFOGAIN', # parameter
            'lts.maxterms=100000',
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
        with open('lcs3.conf', 'w', encoding = 'utf-8') as config_out:
            config_out.write(config)
