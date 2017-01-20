
from luiginlp.engine import Task, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, IntParameter, BoolParameter
import numpy
from scipy import sparse
from collections import defaultdict

import quoll.classification_pipeline.functions.nfold_cv_functions as nfold_cv_functions
import quoll.classification_pipeline.functions.linewriter as linewriter
import quoll.classification_pipeline.functions.docreader as docreader

from quoll.classification_pipeline.modules.tokenize_instances import Tokenize
from quoll.classification_pipeline.modules.featurize_instances import Featurize_tokens
from quoll.classification_pipeline.modules.make_bins import MakeBins 
from quoll.classification_pipeline.modules.run_experiment import ExperimentComponent
from quoll.classification_pipeline.modules.run_nfold_cv_vectors import ReportFolds


@registercomponent
class SparsePipeline(WorkflowComponent):

    instances = Parameter()
    labels = Parameter()
    classifier_args = Parameter()

    tokconfig = Parameter(default=False)
    strip_punctuation = BoolParameter(default=True)
    token_ngrams = Parameter(default='1 2 3')
    blackfeats = Parameter(default=False)
    lowercase = BoolParameter(default=True)   
    n = IntParameter(default=10)
    weight = Parameter(default='frequency')
    prune = IntParameter(default=5000)
    balance = BoolParameter(default=False)
    classifier = Parameter(default='naive_bayes')
    ordinal = BoolParameter(default=False)


    def accepts(self):
        return [ ( InputFormat(self,format_id='instances',extension='.txt',inputparameter='instances'), InputFormat(self, format_id='labels', extension='.labels', inputparameter='labels'), InputFormat(self,format_id='classifier_args',extension='.txt',inputparameter='classifier_args') ) ]

    def setup(self, workflow, input_feeds):

        tokenizer = workflow.new_task('tokenize_instances', Tokenize, autopass=True, tokconfig=self.tokconfig, strip_punctuation=self.strip_punctuation)
        tokenizer.in_txt = input_feeds['instances']

        featurizer = workflow.new_task('featurize_instances', Featurize_tokens, autopass=True, token_ngrams=self.token_ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase)
        featurizer.in_tokenized = tokenizer.out_tokenized

        bin_maker = workflow.new_task('make_bins', MakeBins, autopass=True, n=self.n)
        bin_maker.in_labels = input_feeds['labels']

        fold_runner = workflow.new_task('nfold_cv', RunFolds, autopass=True, n = self.n, weight=self.weight, prune=self.prune, balance=self.balance, classifier=self.classifier, ordinal=self.ordinal)
        fold_runner.in_bins = bin_maker.out_bins
        fold_runner.in_features = featurizer.out_features
        fold_runner.in_labels = input_feeds['labels']
        fold_runner.in_vocabulary = featurizer.out_vocabulary
        fold_runner.in_documents = input_feeds['instances']        
        fold_runner.in_classifier_args = input_feeds['classifier_args']

        folds_reporter = workflow.new_task('report_folds', ReportFolds, autopass = False)
        folds_reporter.in_expdirectory = fold_runner.out_exp

        return folds_reporter