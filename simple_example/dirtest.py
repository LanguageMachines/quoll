import os
import luiginlp
from luiginlp.engine import Task,TargetInfo, StandardWorkflowComponent, registercomponent, InputFormat
from luiginlp.util import replaceextension

class DirTestTask(Task):

    in_txtdir = None #input slot placeholder (will be linked to an out_* slot of another module in the workflow specification)

    def out_txt(self):
        return TargetInfo(self, replaceextension(self.in_txtdir().path, '.txtdir','.index'))

    def run(self):
        #execute a shell command, python keyword arguments will be passed as option flags (- for one letter, -- for more)
        # values will be made shell-safe.
        # None or False values will not be propagated at all.

        indexfile = open('w.index','w')

        for filename in os.listdir(self.in_txtdir().path):
            indexfile.write(open(self.in_txtdir().path+'/'+filename).read())

@registercomponent
class DirTestComponent(StandardWorkflowComponent):

    def autosetup(self):
        return (DirTestTask)

    def accepts(self):
        """Returns a tuple of all the initial inputs and other workflows this component accepts as input (a disjunction, only one will be selected)"""
        return (InputFormat(self, format_id='txtdir', extension='txtdir',directory=True))

if __name__ == '__main__':

    luiginlp.run(DirTestComponent(inputfile="w.txtdir"))
