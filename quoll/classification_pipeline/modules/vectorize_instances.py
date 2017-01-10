
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
