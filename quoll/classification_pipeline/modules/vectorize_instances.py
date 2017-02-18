
import numpy
from scipy import sparse
from luiginlp.engine import Task, StandardWorkflowComponent, WorkflowComponent, InputFormat, registercomponent, InputSlot, Parameter, BoolParameter

from quoll.classification_pipeline.functions import vectorizer, docreader, pairwise_functions

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

        return transformer

class TransformVectorsPairwiseTask(Task):

    in_vectors = InputSlot()
    in_labels = InputSlot()
    in_documents = InputSlot()

    def out_vectors(self):
        return self.outputfrominput(inputformat='vectors', stripextension='.vectors.npz', addextension='.pairwise.vectors.npz')

    def out_labels(self):
        return self.outputfrominput(inputformat='labels', stripextension='.labels', addextension='.pairwise.labels')

    def out_documents(self):
        return self.outputfrominput(inputformat='documents', stripextension='.txt', addextension='.pairwise.txt')

    def run(self):

        # load vectors
        loader = numpy.load(self.in_vectors().path)
        instances = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape']).toarray()

        # load labels
        with open(self.in_labels().path,'r',encoding='utf-8') as infile:
            labels = infile.read().strip().split('\n')
        labels_np = numpy.array([float(label) for label in labels])

        # load documents
        with open(self.in_documents().path,'r',encoding='utf-8') as infile:
            documents = infile.read().strip().split('\n')

        # transform vectors
        transformed_vectors, transformed_labels, combinations = pairwise_functions.transform_pairwise(instances,labels_np)
        transformed_vectors_sparse = sparse.csr_matrix(transformed_vectors)
        transformed_labels_txt = [str(label) for label in transformed_labels.tolist()]
        transformed_documents = [str(combi[0]) + ' - ' + str(combi[1]) for combi in combinations]

        # write vectors
        numpy.savez(self.out_vectors().path, data=transformed_vectors_sparse.data, indices=transformed_vectors_sparse.indices, indptr=transformed_vectors_sparse.indptr, shape=transformed_vectors_sparse.shape)

        # write labels
        with open(self.out_labels().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(transformed_labels_txt))

        # write documents
        with open(self.out_documents().path,'w',encoding='utf-8') as outfile:
            outfile.write('\n'.join(transformed_documents))

@registercomponent
class TransformPairwise(WorkflowComponent):
    
    vectors = Parameter()
    labels = Parameter()
    documents = Parameter()

    def accepts(self):
        return [ ( InputFormat(self,format_id='vectors',extension='.vectors.npz',inputparameter='vectors'), InputFormat(self, format_id='labels',extension='.labels',inputparameter='labels'), InputFormat(self, format_id='documents', extension='.txt', inputparameter='documents') ) ]

    def setup(self, workflow, input_feeds):

        pairwise_transformer = workflow.new_task('transform_vectors_pairwise', TransformVectorsPairwiseTask, autopass=True)
        pairwise_transformer.in_vectors = input_feeds['vectors']
        pairwise_transformer.in_labels = input_feeds['labels']
        pairwise_transformer.in_documents = input_feeds['documents']

        return pairwise_transformer
