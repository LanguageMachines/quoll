
import numpy
from scipy import sparse
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, registercomponent, InputSlot, BoolParameter

from quoll.classification_pipeline.functions import vectorizer, docreader

class VectorizeTask(Task):

    in_csv = InputSlot()

    normalize = BoolParameter()

    def out_vectors(self):
        return self.outputfrominput(inputformat='csv', stripextension='.csv', addextension='.vectors.npz')

    def run(self):

        # load instances
        loader = docreader.Docreader()
        instances = loader.parse_csv(self.in_csv().path)
        instances_float = [[0.0 if feature == 'NA' else float(feature.replace(',','.')) for feature in instance] for instance in instances]
        instances_sparse = sparse.csr_matrix(instances_float)

        # normalize features
        if self.normalize:
            instances_sparse = vectorizer.normalize_features(instances_sparse)

        # write instances to file
        numpy.savez(self.out_vectors().path, data=instances_sparse.data, indices=instances_sparse.indices, indptr=instances_sparse.indptr, shape=instances_sparse.shape)



@registercomponent
class Vectorize(StandardWorkflowComponent):

    instances = InputSlot()

    normalize = BoolParameter()

    def accepts(self):
        return InputFormat(self, format_id='csv', extension='.csv')

    def autosetup(self):
        return VectorizeTask

class TransformVectorsTask(Task):

    in_vectors = InputSlot()
    in_selection = InputSlot()

    def out_vectors(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.transformed.vectors.npz')

    def run(self):

        # load vectors
        loader = numpy.load(self.in_vectors().path)
        instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

        # load selection
        with open(self.in_selection().path) as infile:
            selection = [int(x) for x in infile.read().strip().split()]

        # transform vectors
        transformed_vectors = vectorizer.compress_vectors(instances,selection)

        # write vectors
        numpy.savez(self.out_vectors().path, data=transformed_vectors.data, indices=transformed_vectors.indices, indptr=transformed_vectors.indptr, shape=transformed_vectors.shape)

@registercomponent
class TransformVectors(WorkflowComponent):

    vectors = Parameter()
    selection = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='vectors',extension='.vectors.npz',inputparameter='vectors'), InputFormat(self, format_id='selection',extension='.txt',inputparameter='selection') ) ]

    def setup(self, workflow, input_feeds):

        transformer = workflow.new_task('transform_vectors', TransformVectorsTask, autopass=True)
        transformer.in_vectors = input_feeds['vectors']
        transformer.in_selection = input_feeds['selection']
