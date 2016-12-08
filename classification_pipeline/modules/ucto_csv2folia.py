

import sys
import os
import glob
from pynlpl.formats import folia
import luiginlp
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter
from luiginlp.modules.ucto import Ucto
import featurizer

class Ucto_csv2folia(Task):

    in_csv = InputSlot() #input slot for a csv document

    def out_featuretxt(self):
        return self.outputfrominput(inputformat='csv', stripextension='.csv', addextension='.tokenized.folia')

    def run(self):
        doc = folia.Document(file=self.in_folia().path, encoding = 'utf-8')

        ft = featurizer.Featurizer()
        features = ft.extract_words(doc)
        with open(self.out_featuretxt().path,'w',encoding = 'utf-8') as f_out:
            f_out.write(' '.join(features))

@registercomponent
class UctoCSV(StandardWorkflowComponent):

    def autosetup(self):
        return Ucto_csv2folia

    def accepts(self):
        return InputFormat(self, format_id='csv', extension='.csv'),