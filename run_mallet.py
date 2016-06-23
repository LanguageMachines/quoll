
from luiginlp.engine import Task, InputSlot
from luiginlp.util import replaceextension
from luigi import Parameter

class RunMalletTask(Task):

    executable = 'mallet'

    mallet_rundir = Parameter()
    num_topics = Parameter(default=100)
    interval = Parameter(default=10)

    #this is the input slot for plaintext files, input slots are
    #connected to output slots of other tasks by a workflow component
    in_dir = InputSlot()

    #Define an output slot, output slots are methods that start with out_
    def out_malletdir(self):
        #Output slots always return TargetInfo instances pointing to the
        #output file, we derive the name of the output file from the input
        #file, and replace the extension
        return self.outputfrominput(inputformat='featuredir', stripextension='.featuredir', addextension='.malletdir')

    #Define the run method, this will be called to do the actual work
    def run(self):
        #Here we run the external tool. This will invoke the executable
        #specified. Keyword arguments are passed as option flags (-L in
        #this case). Positional arguments are passed as such (after option flags).
        #All parameters are available on the Task instance
        #Values will be passed in a shell-safe manner, protecting against injection attacks
        #os.chdir(self.mallet_rundir)
        self.ex('import-dir', keep__sequence=True, input=self.in_dir().path, output=self.output().path)
