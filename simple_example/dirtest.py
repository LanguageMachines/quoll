import os
import luiginlp
from luiginlp.engine import Task,TargetInfo, StandardWorkflowComponent, registercomponent, InputFormat
from luiginlp.util import replaceextension
from luigi import Parameter

#---------------- Tasks -----------------------

class ProcessSingleFileTask(Task):

    in_txt = None
    outputdir = Parameter(default="")

    def out_processedtxt(self):
        #Here we say that we will put the processed file int he output dir, with .processedtxt extension
        return TargetInfo(self, os.path.join(self.outputdir, os.path.basename(replaceextension(self.in_txt().path, '.txt','.processedtxt'))))

    def run(self):
        #Here we actually process the file
        open(self.out_processedtxt().path.replace('.txt','.processedtxt'),'w').write(open(self.in_txt().path).read()+' Wessel is de beste!')

class ProcessFileForFileTask(Task):
    """Processes a whole dir, strategy A... enables parallellization"""

    in_txtdir = None

    def out_processedtxtdir(self):
        return TargetInfo(self, replaceextension(self.in_txtdir().path, '.txtdir','.processedtxtdir'))

    def run(self):
        #Create the output file
        self.setup_output_dir(self.out_processedtxtdir().path)

        #Here we process each file, but giving it to another component, the yield pauses the iteration and thus makes
        #parallellization possible
        for file in os.listdir(self.in_txtdir().path):
            yield SingleFileComponent(inputfile=self.in_txtdir().path+'/'+file,outputdir=self.out_processedtxtdir().path)

class ProcessDirTask(Task):
    """Processes a whole dir, strategy B... without parallellization"""

    in_txtdir = None

    def out_txt(self):
        return TargetInfo(self, replaceextension(self.in_txtdir().path, '.txtdir','.index'))

    def run(self):
        indexfile = open('w.index','w')

        for filename in os.listdir(self.in_txtdir().path):
            indexfile.write(open(self.in_txtdir().path+'/'+filename).read())

#---------------- Components ----------------------

@registercomponent
class FileForFileComponent(StandardWorkflowComponent):
    """Processes a whole dir, strategy A"""

    def autosetup(self):
        return (ProcessFileForFileTask)

    def accepts(self):
        return (InputFormat(self, format_id='txtdir', extension='txtdir',directory=True))

@registercomponent
class SingleFileComponent(StandardWorkflowComponent):

    outputdir = Parameter(default="")

    def autosetup(self):
        return (ProcessSingleFileTask)

    def accepts(self):
        return (InputFormat(self, format_id='txt', extension='txt'))

@registercomponent
class WholeDirComponent(StandardWorkflowComponent):
    """Processes a whole dir, strategy B"""

    def autosetup(self):
        return (ProcessDirTask)

    def accepts(self):
        return (InputFormat(self, format_id='txtdir', extension='txtdir',directory=True))


if __name__ == '__main__':

    luiginlp.run(FileForFileComponent(inputfile="w.txtdir"))
