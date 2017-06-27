
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter

import frog
import glob
import json

class Frog_instances(Task):
    """"Frogs a file one document per line"""
    in_txt = InputSlot()

    frogconfig = Parameter()
    strip_punctuation = BoolParameter()

    def out_frogged(self):
        return self.outputfrominput(inputformat='txt', stripextension='.txt', addextension='.frog.json')

    def run(self):

        # open file
        with open(self.in_txt().path, 'r', encoding = 'utf-8') as file_in:
            lines = file_in.readlines()

        # initialize frogger
        fo = frog.FrogOptions(ner=False, chunking=False, mwu=False, lemma=True, morph=False, daringmorph=False)
        frogger = frog.Frog(fo,self.frogconfig)

        # initialize print reports    
        numlines = len(lines)
        if numlines > 1000:
            reports = range(0, numlines, 1000)
        elif numlines > 100:
            reports = range(0, numlines, 10)
        else:
            reports = range(numlines)

        # for each line
        documents = []
        for i, line in enumerate(lines):
            if i in reports:
                print(i, 'of', numlines, 'lines frogged.')
            # frog
            frogged = frogger.process(line)
            # initialize sentences
            sentences = []
            sentence = []
            # add each token to the sentence...
            for token in frogged:
                if not (self.strip_punctuation and token['pos'] == 'LET()'):
                    sentence.append({'text':token['text'], 'lemma':token['lemma'], 'pos':token['pos']})
                # ...until the sentence ends
                if 'eos' in token:
                    sentences.append(sentence)
                    # initialize a new sentence
                    sentence = []
            # scrape last bits of info
            if len(sentence) > 0:
                sentences.append(sentence)
            documents.append(sentences)

        # write output
        with open(self.out_frogged().path,'w',encoding = 'utf-8') as file_out:
            json.dump(documents,file_out)

class Frog_txtdir(Task):
    """"Tokenizes a directory with files"""

    in_txtdir = InputSlot()

    frogconfig = Parameter()
    strip_punctuation = BoolParameter()

    def out_frogjsondir(self):
        return self.outputfrominput(inputformat='txtdir', stripextension='.txtdir', addextension='.frog.jsondir')

    def run(self):
        #Set up the output directory, will create it and tear it down on failure automatically
        self.setup_output_dir(self.out_frogjsondir().path)

        #gather input files
        inputfiles = [ filename for filename in glob.glob(self.in_txtdir().path + '/*.txt') ]

        print('Running Frogger...')
        yield [ Frog_component(inputfile=inputfile,outputdir=self.out_frogjsondir().path,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation) for inputfile in inputfiles ]


@registercomponent
class Frog_component(StandardWorkflowComponent):

    frogconfig = Parameter()
    strip_punctuation = BoolParameter()

    def autosetup(self):
        return Frog_instances

    def accepts(self):
        return InputFormat(self, format_id='txt', extension='.txt')

@registercomponent
class Frog_dir(StandardWorkflowComponent):

    frogconfig = Parameter()
    strip_punctuation = BoolParameter()

    def autosetup(self):
        return Frog_txtdir

    def accepts(self):
        return InputFormat(self, format_id='txtdir', extension='.txtdir')
