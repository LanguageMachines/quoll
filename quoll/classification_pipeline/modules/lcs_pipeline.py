
from luiginlp.engine import WorkflowComponent, InputFormat, registercomponent, Parameter, IntParameter, BoolParameter

from quoll.classification_pipeline.modules.tokenize_instances import Tokenize_instances
from quoll.classification_pipeline.modules.featurize_instances import Featurize_tokens
from quoll.classification_pipeline.modules.vectorize_sparse_instances import Vectorize_traininstances, Vectorize_testinstances
from quoll.classification_pipeline.modules.classify_instances import BalancedWinnowClassifier
from quoll.classification_pipeline.modules.report_performance import ReportPerformance

@registercomponent
class LCSPipeline_traintest(WorkflowComponent):

    traininstances = Parameter()
    trainlabels = Parameter()
    testinstances = Parameter()
    testlabels = Parameter()
    testdocuments = Parameter()

    tokconfig = Parameter(default=False)
    strip_punctuation = BoolParameter(default=True)
    token_ngrams = Parameter(default='1 2 3')
    blackfeats = Parameter(default=False)
    lowercase = BoolParameter(default=True)   
    weight = Parameter(default='frequency')
    prune = IntParameter(default=5000)
    balance = BoolParameter(default=False)
    lcs_path = Parameter()
    ordinal = BoolParameter(default=False)

    def accepts(self):
        return [ ( InputFormat(self,format_id='traininstances',extension='.txt',inputparameter='traininstances'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self,format_id='testinstances',extension='.txt',inputparameter='testinstances'), InputFormat(self, format_id='testlabels', extension='.labels', inputparameter='testlabels'), InputFormat(self,format_id='testdocuments',extension='.txt',inputparameter='testdocuments') ) ]

    def setup(self, workflow, input_feeds):

        traintokenizer = workflow.new_task('tokenize_traininstances', Tokenize_instances, autopass=True, tokconfig=self.tokconfig, strip_punctuation=self.strip_punctuation)
        traintokenizer.in_txt = input_feeds['traininstances']

        testtokenizer = workflow.new8_task('tokenize_testinstances', Tokenize_instances, autopass=True, tokconfig=self.tokconfig, strip_punctuation=self.strip_punctuation)
        testtokenizer.in_txt = input_feeds['traininstances']

        trainfeaturizer = workflow.new_task('featurize_traininstances', Featurize_tokens, autopass=True, token_ngrams=self.token_ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase)
        trainfeaturizer.in_tokenized = traintokenizer.out_tokenized

        testfeaturizer = workflow.new_task('featurize_testinstances', Featurize_tokens, autopass=True, token_ngrams=self.token_ngrams, blackfeats=self.blackfeats, lowercase=self.lowercase)
        testfeaturizer.in_tokenized = testtokenizer.out_tokenized

        trainvectorizer = workflow.new_task('vectorize_traininstances', Vectorize_traininstances, autopass=True, weight=self.weight, prune=self.prune, balance=self.balance)
        trainvectorizer.in_train = trainfeaturizer.out_features
        trainvectorizer.in_trainlabels = input_feeds['trainlabels']
        trainvectorizer.in_vocabulary = trainfeaturizer.out_vocabulary

        testvectorizer = workflow.new_task('vectorize_testinstances', Vectorize_testinstances, autopass=True, weight=self.weight)
        testvectorizer.in_test = testfeaturizer.out_features
        testvectorizer.in_sourcevocabulary = testfeaturizer.out_vocabulary
        testvectorizer.in_topfeatures = trainvectorizer.out_topfeatures

        lcs_classifier = workflow.new_task('classify_lcs', BalancedWinnowClassifier, autopass=True, lcs_path=self.lcs_path)
        lcs_classifier.in_train = trainvectorizer.out_train
        lcs_classifier.in_trainlabels = trainvectorizer.out_labels
        lcs_classifier.in_test = testvectorizer.out_test
        lcs_classifier.in_testlabels = input_feeds['testlabels']
        lcs_classifier.in_vocabulary = trainvectorizer.out_topfeatures

        reporter = workflow.new_task('report_performance', ReportPerformance, autopass=True, ordinal=self.ordinal)
        reporter.in_labels = input_feeds['testlabels']
        reporter.in_predictions = lcs_classifier.out_classifications
        reporter.in_documents = input_feeds['testdocuments']

        return reporter
