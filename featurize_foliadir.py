
import sys
import os
import glob
from pynlpl.formats import folia
from luigi import Parameter
import luiginlp
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, registercomponent
import featurizer

#def featurizer_foliadir(foliadir, outdir):
#    files = os.listdir(foliadir)

    #ft = featurizer.Featurizer(lowercase = False, skip_punctuation = False, setname = 'piccl')
#    ft = featurizer.Featurizer()

#    for f in files:
#        print(f, foliadir + f)
#        try:
#            outfile = f[:-4] + '.txt'
#            if outfile in os.listdir(outdir):
#                print('file already generated, skipping')
#                continue
#            else:
#                doc = folia.Document(file = foliadir + f, encoding = 'utf-8')
#                features = ft.extract_words(doc)
#                with open(outfile, 'w', encoding = 'utf-8') as f_out:
#                    f_out.write(' '.join(features))
#        except:
            #exc_type, exc_obj, exc_tb = sys.exc_info()
#            print('Error parsing doc', foliadir + f)

class FeaturizerTask_single(Task):
    """Featurizes a single FoLiA XML file"""

    outputdir = Parameter(default="") #optional output directory (output will be written to same dir as inputfile otherwise)

    in_folia = None #input slot for a FoLiA document

    def out_featuretxt(self):
        """Output slot -- outputs a single *.features.txt file"""
        return self.outputfrominput(inputformat='folia', inputextension='.folia.xml', outputextension='.features.txt')

    def run(self):
        """Run the featurizer"""
        ft = featurizer.Featurizer()
        doc = folia.Document(file=self.in_folia().path, encoding = 'utf-8')
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
 
    in_foliadir = None #input slot, directory of FoLiA documents (files must have folia.xml extension)

    def out_featuredir(self):
        """Output slot - Directory of feature files"""
        return self.outputfrominput(inputformat='foliadir', inputextension='.foliadir', outputextension='.featuredir')

    def run(self):
        #Set up the output directory, will create it and tear it down on failure automatically
        self.setup_output_dir(self.out_featuredir().path)

        #gather input files
        inputfiles = [ filename for filename in glob.glob(self.in_foliadir().path + '/*.folia.xml') ]

        #inception aka dynamic dependencies: we yield a list of tasks to perform which could not have been predicted statically
        #in this case we run the FeaturizerTask_single component for each input file in the directory
        yield [ FeaturizerComponent_single(inputfile=inputfile,outputdir=self.out_featuredir().path) for inputfile in inputfiles ]

@registercomponent
class FeaturizerComponent_dir(StandardWorkflowComponent):
    def autosetup(self):
        return FeaturizerTask_dir

    def accepts(self):
        return InputFormat(self, format_id='foliadir', extension='foliadir', directory=True)

if __name__ == '__main__':
    foliadir = sys.argv[1]

    luiginlp.run(FeaturizerComponent_dir(inputfile = foliadir))
