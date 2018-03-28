#!/usr/bin/env 

import os
import numpy
import re

from quoll.classification_pipeline.functions import nfold_cv_functions

class LCS_classifier:

    def __init__(self, directory):
        self.expdir = directory
        self.filesdir = False 
        self.predictions = False
        self.full_predictions = False
        self.docs = False
        self.model = False
        
    def experiment(self,traininstances,trainlabels,vocabulary,testinstances=False,testlabels=False):
        self.filesdir = self.expdir + '/files/'
        try:
            os.mkdir(self.filesdir)
        except:
            print('filesdirectory already exists')
        # try:
        if not os.path.exists('train'):
            trainparts, trainfile_index = self.prepare(traininstances,trainlabels,vocabulary, 'train/')
            with open('train', 'w', encoding = 'utf-8') as train:
                train.write('\n'.join(trainparts))            
            with open(self.expdir + '/trainfile_index.txt','w',encoding='utf-8') as out:
                out.write('\n'.join([' '.join([str(x) for x in line]) for line in trainfile_index]))
        else:
            print('Trainfile already exists, skipping data preparation')
        if not os.path.exists('test'):
            testparts, testfile_index = self.prepare(testinstances,testlabels,vocabulary, 'test/')
            with open('test', 'w', encoding = 'utf-8') as test:
                test.write('\n'.join(testparts))
            with open(self.expdir + '/testfile_index.txt','w',encoding='utf-8') as out:
                out.write('\n'.join([' '.join([str(x) for x in line]) for line in testfile_index]))
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
        filename_index = []
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
        index = 0
        data = list(zip(labels,instances_vocabulary))
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
                filename_index.append([filename, index])
                index += 1
                label = instance[0]
                features = instance[1]
                with open(self.filesdir + filename, 'w', encoding = 'utf-8') as outfile: 
                    outfile.write('\n'.join(features))
                parts.append(filename + ' ' + label)
        return parts, filename_index

    def classify(self,expdir):
        self.write_config()
        os.system('lcs --verbose')
        self.docs, self.predictions, self.full_predictions = self.extract_performance('test.rnk')
        self.model = self.extract_model('data/')
        os.system('mv train ' + expdir)
        os.system('mv data/ ' + expdir)
        os.system('mv test* ' + expdir)
        os.system('mv lcs* ' + expdir)
        os.system('mv index/ ' + expdir)

    def extract_performance(self,testrnk_file):
        with open(testrnk_file) as rnk:
            for i,line in enumerate(rnk.readlines()):
                tokens = line.strip().split()
                filename = tokens[0].strip()
                classifications = tokens[1:]
                if i == 0:
                    prediction_cats = [cl.split(':')[0].replace('?','') for cl in classifications]
                    full_predictions = [prediction_cats]
                    predictions = []
                    docs = []
                # for classification in classifications:
                prediction = classifications[0].split(':')[0].replace('?','')
                full_prediction = [0] * len(prediction_cats)
                for prediction_prob in classifications:
                    cat, prob = prediction_prob.split(':')
                    cat = cat.replace('?','')
                    full_prediction[prediction_cats.index(cat)] = prob
                docs.append(filename)
                predictions.append(prediction)
                full_predictions.append(full_prediction)
            return docs, predictions, full_predictions

    def extract_model(self,datadir):
        categories = list(set(self.predictions))
        all_files = os.listdir(datadir)
        category_models = [f for f in all_files if f.endswith('mitp')]
        out_models = {}
        for category in categories:
            modelfile = max([f for f in category_models if re.match('^'+category,f)])
            with open(datadir + modelfile,'r',encoding='utf-8') as mf_in:
                lines = mf_in.read().strip().split('\n')[6:]
            clines = [line.strip().split('\t')[:2] for line in lines]
            clines_f = [[line[0],float(line[1])] for line in clines]
            clines_f_sorted = sorted(clines_f,key = lambda k : k[1],reverse=True)
            out_models[category] = clines_f_sorted
        return out_models              

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
