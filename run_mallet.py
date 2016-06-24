
from luiginlp.engine import Task, registercomponent, StandardWorkflowComponent, InputFormat, InputSlot, Parameter
from luiginlp.util import replaceextension
import os

class RunMalletTask(Task):

    executable = 'mallet'

    mallet_rundir = Parameter()
    num_topics = Parameter(default=100)
    interval = Parameter(default=10)
    topic_threshold = Parameter(default=0.1)
    rand_var = Parameter(default=123456)
    num_iter = Parameter(default=2000)

    #this is the input slot for plaintext files, input slots are
    #connected to output slots of other tasks by a workflow component
    in_featuredir = InputSlot()

    #Define an output slot, output slots are methods that start with out_
    def out_malletdir(self):
        #Output slots always return TargetInfo instances pointing to the
        #output file, we derive the name of the output file from the input
        #file, and replace the extension
        return self.outputfrominput(inputformat='featuredir', stripextension='.featuredir', addextension='.mallet')

    #Define the run method, this will be called to do the actual work
    def run(self):
        #Here we run the external tool. This will invoke the executable
        #specified. Keyword arguments are passed as option flags (-L in
        #this case). Positional arguments are passed as such (after option flags).
        #All parameters are available on the Task instance
        #Values will be passed in a shell-safe manner, protecting against injection attacks
        #os.chdir(self.mallet_rundir)
        self.ex('import-dir', keep__sequence=True, input=self.in_featuredir().path, output=self.output()[0].path, __options_last=True)
        self.ex('train-topics', num__iterations=self.num_iter, random__seed=self.rand_var, num__topics=self.num_topics, optimize__interval=self.interval, input=self.output()[0].path, doc__topics__threshold=self.topic_threshold, output__doc__topics=self.output()[0].path+'.NT'+self.num_topics+'.I'+self.interval+'.THR'+self.topic_threshold+'.doc-topics', output__topic__keys=self.output()[0].path+'.NT'+self.num_topics+'.I'+self.interval+'.rand'+self.rand_var+'.topic-keys.txt', inferencer__filename=self.output()[0].path+'.NT'+self.num_topics+'.I'+self.interval+'.rand'+self.rand_var+'.inferencer', __options_last=True)   

@registercomponent
class RunMalletComponent(StandardWorkflowComponent):

    mallet_rundir = Parameter()
    num_topics = Parameter(default=100)
    interval = Parameter(default=10)
    topic_threshold = Parameter(default=0.1)
    rand_var = Parameter(default=123456)
    num_iter = Parameter(default=2000)

    def autosetup(self):
        return RunMalletTask

    def accepts(self):
        return InputFormat(self, format_id='featuredir', extension='featuredir', directory=True),
