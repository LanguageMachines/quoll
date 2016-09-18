
import sys
import os
import glob
from pynlpl.formats import folia
import luiginlp
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter
from luiginlp.modules.ucto import Ucto_dir
import simple_featurizer

class FeaturizerTask_single(Task):
    """Featurizes a single FoLiA XML file"""


    in_folia = InputSlot() #input slot for a FoLiA document

    def out_featuretxt(self):
        """Output slot -- outputs a single *.features.txt file"""
        return self.outputfrominput(inputformat='folia', stripextension='.folia.xml', addextension='.features.txt')

    def run(self):
        """Run the featurizer"""
        ft = simple_featurizer.SimpleFeaturizer()
        doc = folia.Document(file=self.in_folia().path, encoding = 'utf-8', recover=True)
        features = ft.extract_words(doc)
        with open(self.out_featuretxt().path,'w',encoding = 'utf-8') as f_out:
            f_out.write(' '.join(features))

@registercomponent
class FeaturizerComponent_single(StandardWorkflowComponent):

    def autosetup(self):
        return FeaturizerTask_single

    def accepts(self):
        return InputFormat(self, format_id='folia', extension='folia.xml'),

FeaturizerComponent_single.inherit_parameters(FeaturizerTask_single)

class FeaturizerTask_dir(Task):
 
    in_tokfoliadir = InputSlot() #input slot, directory of FoLiA documents (files must have folia.xml extension)

    def out_featuredir(self):
        """Output slot - Directory of feature files"""
        return self.outputfrominput(inputformat='tokfoliadir', stripextension='.tok.foliadir', addextension='.featuredir')

    def run(self):
        #Set up the output directory, will create it and tear it down on failure automatically
        self.setup_output_dir(self.out_featuredir().path)

        #gather input files
        inputfiles = [ filename for filename in glob.glob(self.in_tokfoliadir().path + '/*.folia.xml') ]

        #inception aka dynamic dependencies: we yield a list of tasks to perform which could not have been predicted statically
        #in this case we run the FeaturizerTask_single component for each input file in the directory
        yield [ FeaturizerComponent_single(inputfile=inputfile,outputdir=self.out_featuredir().path) for inputfile in inputfiles ]

@registercomponent
class FeaturizerComponent_dir(StandardWorkflowComponent):

    language = Parameter()

    def setup(self, workflow, input_feeds):
        featurizertask = workflow.new_task('FeaturizerTask_dir', FeaturizerTask_dir, autopass=True)
        featurizertask.in_tokfoliadir = input_feeds['tokfoliadir']
        return featurizertask

    def accepts(self):
        return InputFormat(self, format_id='tokfoliadir', extension='.tok.foliadir', directory=True), InputComponent(self, Ucto_dir, language = self.language)

if __name__ == '__main__':
    foliadir = sys.argv[1]
    lang = sys.argv[2]

    luiginlp.run(FeaturizerComponent_dir(inputfile=foliadir, language=lang, startcomponent=Ucto_dir))
