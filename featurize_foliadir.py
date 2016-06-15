
import sys
import os

from pynlpl.formats import folia
from luigi import Parameter
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, registercomponent, TargetInfo
from luiginlp.util import replaceextension
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

    in_folia = None #input slot

    def out_featuretxt(self):
        if self.outputdir and self.outputdir != '.': #handle optional outputdir
            return TargetInfo(self, os.path.join(self.outputdir, os.path.basename(replaceextension(self.in_folia().path, '.folia.xml','.features.txt'))))
        else:
            return TargetInfo(self, replaceextension(self.in_folia().path, '.folia.xml','.features.txt'))

    def run(self):
        ft = featurizer.Featurizer()
        doc = folia.Document(file=self.in_folia().path, encoding = 'utf-8')
        features = ft.extract_words(doc)
        with open(self.out_featuretxt().path,'w',encoding = 'utf-8') as f_out:
            f_out.write(' '.join(features))

class FeaturizerComponent_single(StandardWorkflowComponent):
    def autosetup(self):
        return FeaturizerTask_single

    def accepts(self):
        return InputFormat(self, format_id='folia', extension='folia.xml'),




if __name__ == '__main__':
    foliadir = sys.argv[1]
    outdir = sys.argv[2]
