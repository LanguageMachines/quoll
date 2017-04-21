
import numpy
from scipy import sparse

from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter, IntParameter

from quoll.classification_pipeline.functions import linewriter, docreader, fcbf, vectorizer

################################################################################
###Component
################################################################################

@registercomponent
class ApplyFCBF(WorkflowComponent):

    trainvectors = Parameter()
    trainlabels = Parameter()
    featurenames = Parameter()

    threshold = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='trainvectors',extension='.vectors.npz',inputparameter='trainvectors'), InputFormat(self, format_id='trainlabels', extension='.labels', inputparameter='trainlabels'), InputFormat(self, format_id='featurenames', extension='.txt', inputparameter='featurenames') ) ]

    def setup(self, workflow, input_feeds):

        file_formatter = workflow.new_task('trainvectors_2_fcbf_input', Trainvectors2FCBFInput, autopass=False)
        file_formatter.in_vectors = input_feeds['trainvectors']
        file_formatter.in_labels = input_feeds['trainlabels']

        feature_selector = workflow.new_task('select_features_fcbf', FCBFTask, autopass=False, threshold=self.threshold)
        feature_selector.in_instances = file_formatter.out_instances
        feature_selector.in_featurenames = input_feeds['featurenames']

        vector_transformer = workflow.new_task('transform_vectors', vectorizer.TransformVectors, autopass=False)
        vector_transformer.in_vectors = input_feeds['trainvectors']
        vector_transformer.selection = feature_selector.out_feature_indices

        return vector_transformer


################################################################################
###Tasks
################################################################################

class Trainvectors2FCBFInput(Task):

    in_vectors = InputSlot()
    in_labels = InputSlot()

    def out_instances(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.fcbf.csv')

    def run(self):

        # load vectors
        loader = numpy.load(self.in_vectors().path)
        vectors = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load labels
        with open(self.in_labels().path,'r',encoding='utf-8') as file_in:
            labels = file_in.read().strip().split('\n')

        # combine vetors and labels
        if vectors.shape[0] != len(labels):
            print('instances and labels do not align, exiting program...')
        instances_list = vectors.toarray().tolist()
        for i, label in enumerate(labels):
            instances_list[i].append(label)

        # write to file
        lw = linewriter.Linewriter(instances_list)
        lw.write_csv(self.out_instances().path)       

class FCBFTask(Task):

    in_instances = InputSlot()
    in_featurenames = InputSlot()

    threshold = Parameter()

    def out_feature_indices(self):
        return self.outputfrominput(inputformat='instances', stripextension='.txt', addextension='.selected_feature_indices.txt')

    def out_featurenames(self):
        return self.outputfrominput(inputformat='instances', stripextension='.txt', addextension='.selected_features.txt')

    def run(self):

        # run feature selection
        fcbf.fcbf_wrapper(self.in_instances().path, float(self.threshold), delim=',', header=False, classAt=-1)

        # read in output of feature selection
        filetokens = self.in_instances().path.split('/')
        filename = filetokens[-1]
        filepath = '/'.join(filetokens[:-1])
        selected_features_file = filepath + '/features_' + filename
        dr = docreader.Docreader()
        selected_features_lines = dr.parse_csv(selected_features_file)
        print(selected_features_lines)

        # read in featurenames
        with open(self.in_featurenames().path,'r',encoding='utf-8') as file_in:
            featurenames = file_in.read().strip().split('\n')

        # extract selected feature indices and names
        selected_feature_indices = []
        selected_featurenames = numpy.array(featurenames)[selected_feature_indices]

        # write to files
        with open(self.out_feature_indices().path,'w',encoding='utf-8') as file_out:
            file_out.write(' '.join([str(i) for i in selected_feature_indices]))

        with open(self.out_featurenames().path,'w',encoding='utf-8') as file_out:
            file_out.write('\n'.join(selected_featurenames))
