
from luigi import Parameter, IntParameter
from luiginlp.engine import Task, StandardWorkflowComponent, registercomponent, InputSlot, InputComponent, InputFormat
import glob

from featurize_foliadir import FeaturizerComponent_dir

class Featuredir2CorpusTask(Task):

    in_featuredir = InputSlot()

    def out_corpus(self):
        return self.outputfrominput(inputformat='featuredir', stripextension='.featuredir', addextension='.corpus.txt')

    def run(self):
        inputfiles = [filename for filename in glob.glob(self.in_featuredir().path + '/*.features.txt')]
        docs = []
        for inputfile in inputfiles:
            with open(inputfile, 'r', encoding='utf-8') as infile:
                file_str = infile.read()
                words = file_str.strip()
                docs.append(words)
        with open(self.out_corpus().path, 'w', encoding='utf-8') as outfile:
            outfile.write('\n'.join(docs))

class Featuredir2corpusComponent(StandardWorkflowComponent):

    language = Parameter(default='nl')

    def accepts(self):
        return InputFormat(self, format_id='featuredir', extension='.featuredir', directory=True), InputComponent(self, FeaturizerComponent_dir, language=self.language)

    def autosetup(self):
        return Featuredir2CorpusTask
