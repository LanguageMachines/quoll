import os
import os.path

import luiginlp
from luiginlp.engine import Task,TargetInfo, StandardWorkflowComponent, registercomponent, InputFormat
from luiginlp.util import replaceextension

class FolderStructureToLCSIndex(Task):
    """Assumes you have a folder structures with 1 file for every document, organized in folders per document type"""

    in_txtdir = None

    def out_txt(self):
        return TargetInfo(self, replaceextension(self.in_txtdir().path, '.txtdir','.index'))

    def run(self):
        indexfile = open(self.out_txt().path,'w')

        for folder_name in os.listdir(self.in_txtdir().path):

            full_folder_name = self.in_txtdir().path+'/'+folder_name

            if os.path.isfile(full_folder_name):
                continue

            for file_name in os.listdir(full_folder_name):
                indexfile.write(file_name+'\t'+folder_name+'\n')

#This is just to test the tasks above
if __name__ == '__main__':

    @registercomponent
    class BasicComponent(StandardWorkflowComponent):
        """Processes a whole dir, strategy B"""

        def autosetup(self):
            return (FolderStructureToLCSIndex)

        def accepts(self):
            return (InputFormat(self, format_id='txtdir', extension='txtdir',directory=True))

    luiginlp.run(BasicComponent(inputfile="test_data.txtdir"))
