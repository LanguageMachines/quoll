
from luigi import Parameter
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, registercomponent, TargetInfo
from luiginlp.util import replaceextension

class featuredir2corpus(Task):

    in_featuredir = InputSlot()

    def out_corpus(self):

        return self.outputfrominput(inputformat='featuredir', stripextension='.featuredir', addextension='.corpus.txt')

    def run(self):

        inputfiles = [filename for filename in glob.glob(self.in_featuredir().path + '/*.features.txt')]
        with open(self.out_corpus().path, 'w', encoding='utf-8') as outfile:
            for inputfile in inputfiles:
                with open(inputfile, 'r', encoding='utf-8') as infile:
                    file_str = infile.read()
                    words = file_str.strip()
                    outfile.write(words+'\n')

