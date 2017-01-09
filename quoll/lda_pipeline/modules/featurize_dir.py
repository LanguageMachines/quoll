
import sys
import os
import glob
from pynlpl.formats import folia
import luiginlp
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter
from luiginlp.modules.ucto import Ucto_dir

from quoll.lda_pipeline.modules import simple_featurizer #TODO @fkunneman: this one doesn't seem to exist?

class FeaturizerTask_singlefolia(Task):
    """Featurizes a single FoLiA XML file"""

    in_folia = InputSlot() #input slot for a FoLiA document

    def out_featuretxt(self):
        """Output slot -- outputs a single *.features.txt file"""
        return self.outputfrominput(inputformat='folia', stripextension='.folia.xml', addextension='.features.txt')

    def run(self):
        """Run the featurizer"""
        ft = simple_featurizer.SimpleFeaturizer()
        doc = folia.Document(file=self.in_folia().path, encoding = 'utf-8', recover=True)
        features = ft.extract_words_folia(doc)
        with open(self.out_featuretxt().path,'w',encoding = 'utf-8') as f_out:
            f_out.write(' '.join(features))

class FeaturizerTask_singletxt(Task):
    """Featurizes a single txt file"""

    in_txt = InputSlot() #input slot for a FoLiA document

    def out_featuretxt(self):
        """Output slot -- outputs a single *.features.txt file"""
        return self.outputfrominput(inputformat='txt', stripextension='.tok.txt', addextension='.features.txt')

    def run(self):
        """Run the featurizer"""
        ft = simple_featurizer.SimpleFeaturizer()
        with open(self.in_txt().path,'r',encoding='utf-8') as infile:
            doc = infile.read()
        features = ft.extract_words_txt(doc)
        with open(self.out_featuretxt().path,'w',encoding = 'utf-8') as f_out:
            f_out.write(' '.join(features))

@registercomponent
class FeaturizerComponent_singlefolia(StandardWorkflowComponent):

    def autosetup(self):
        return FeaturizerTask_singlefolia

    def accepts(self):
        return InputFormat(self, format_id='folia', extension='folia.xml'),

FeaturizerComponent_singlefolia.inherit_parameters(FeaturizerTask_singlefolia)

@registercomponent
class FeaturizerComponent_singletxt(StandardWorkflowComponent):

    def autosetup(self):
        return FeaturizerTask_singletxt

    def accepts(self):
        return InputFormat(self, format_id='txt', extension='tok.txt'),

FeaturizerComponent_singletxt.inherit_parameters(FeaturizerTask_singletxt)

class FeaturizerTask_dirfolia(Task):
 
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
        yield [ FeaturizerComponent_singlefolia(inputfile=inputfile,outputdir=self.out_featuredir().path) for inputfile in inputfiles ]

class FeaturizerTask_dirtxt(Task):
 
    in_toktxtdir = InputSlot() #input slot, directory of FoLiA documents (files must have folia.xml extension)

    def out_featuredir(self):
        """Output slot - Directory of feature files"""
        return self.outputfrominput(inputformat='toktxtdir', stripextension='.tok.txtdir', addextension='.featuredir')

    def run(self):
        #Set up the output directory, will create it and tear it down on failure automatically
        print('setting up outputdir')
        self.setup_output_dir(self.out_featuredir().path)

        #gather input files
        print('gathering input files')
        inputfiles = [ filename for filename in glob.glob(self.in_toktxtdir().path + '/*.tok.txt') ]

        print(len(inputfiles), 'inputfiles')
        #inception aka dynamic dependencies: we yield a list of tasks to perform which could not have been predicted statically
        #in this case we run the FeaturizerTask_single component for each input file in the directory
        yield [ FeaturizerComponent_singletxt(inputfile=inputfile,outputdir=self.out_featuredir().path) for inputfile in inputfiles ]

@registercomponent
class FeaturizerComponent_dir(StandardWorkflowComponent):

    language = Parameter(default='nl')

    def setup(self, workflow, input_feeds):
        if 'tokfoliadir' in input_feeds.keys():
            featurizertask = workflow.new_task('FeaturizerTask_dirfolia', FeaturizerTask_dirfolia, autopass=True)
            featurizertask.in_tokfoliadir = input_feeds['tokfoliadir']
        elif 'toktxtdir' in input_feeds.keys():
            featurizertask = workflow.new_task('FeaturizerTask_dirtxt', FeaturizerTask_dirtxt, autopass=True)
            featurizertask.in_toktxtdir = input_feeds['toktxtdir']
        return featurizertask

    def accepts(self):
        return InputFormat(self, format_id='tokfoliadir', extension='.tok.foliadir', directory=True), InputFormat(self, format_id='toktxtdir', extension='.tok.txtdir', directory=True), InputComponent(self, Ucto_dir, language = self.language)
