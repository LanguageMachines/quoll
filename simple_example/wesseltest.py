import luiginlp
from luiginlp.engine import Task,TargetInfo, StandardWorkflowComponent, registercomponent, InputFormat
from luiginlp.util import replaceextension

class WesselTestTask(Task):

    in_txt = None #input slot placeholder (will be linked to an out_* slot of another module in the workflow specification)

    def out_txt(self):
        return TargetInfo(self, replaceextension(self.in_txt().path, '.txt','.echoed.txt'))

    def run(self):
        #execute a shell command, python keyword arguments will be passed as option flags (- for one letter, -- for more)
        # values will be made shell-safe.
        # None or False values will not be propagated at all.
        open('w.echoed.txt','w').write(open(self.in_txt().path).read()*10)

@registercomponent
class WesselTestComponent(StandardWorkflowComponent):
    """A workflow component for Frog"""

    def autosetup(self):
        return (WesselTestTask)

    def accepts(self):
        """Returns a tuple of all the initial inputs and other workflows this component accepts as input (a disjunction, only one will be selected)"""
        return (InputFormat(self, format_id='txt', extension='txt'))

if __name__ == '__main__':

    luiginlp.run(WesselTestComponent(inputfile="w.txt"))