
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter

import frog
import ucto

import glob
import json

class Tokenize_instances(Task):
    """"Tokenizes a file one document per line"""
    in_txt = InputSlot()

    tokconfig = Parameter()
    strip_punctuation = BoolParameter()

    def out_tokenized(self):
        return self.outputfrominput(inputformat='txt', stripextension='.txt', addextension='.tok.txt')

    def run(self):

        with open(self.in_txt().path, 'r', encoding = 'utf-8') as file_in:
            lines = file_in.read().strip().split('\n')

        with open(self.out_tokenized().path,'w',encoding = 'utf-8') as file_out:
            c = 0
            tokenizer = ucto.Tokenizer(self.tokconfig)
            for line in lines:
                tokenizer.process(line)
                tokens = []
                for token in tokenizer:
                    if not (self.strip_punctuation and token.tokentype == 'PUNCTUATION'):
                        tokens.append(token.text)
                file_out.write(' '.join(tokens) + '\n')
                c += 1
            print(len(lines),c)


class Tokenize_document(Task):
    """"Tokenizes a document"""
    in_txt = InputSlot()

    tokconfig = Parameter()
    strip_punctuation = BoolParameter()

    def out_tokenized(self):
        return self.outputfrominput(inputformat='txt', stripextension='.txt', addextension='.tok.txt')

    def run(self):

        with open(self.in_txt().path, 'r', encoding = 'utf-8') as file_in:
            parts = file_in.read().strip().split('\n')

        with open(self.out_tokenized().path,'w',encoding = 'utf-8') as file_out:
            tokenizer = ucto.Tokenizer(self.tokconfig)
            for part in parts:
                tokenizer.process(part)
                tokens = []
                for token in tokenizer:
                    if not (self.strip_punctuation and token.tokentype == 'PUNCTUATION'):
                        tokens.append(token.text)
                    if token.isendofsentence():
                        file_out.write(' '.join(tokens) + '\n')
                        tokens = []

class Tokenize_txtdir(Task):
    """"Tokenizes a directory with files"""

    in_txtdir = InputSlot()

    tokconfig = Parameter()
    strip_punctuation = BoolParameter()

    def out_toktxtdir(self):
        return self.outputfrominput(inputformat='txtdir', stripextension='.txtdir', addextension='.tok.txtdir')

    def run(self):
        #Set up the output directory, will create it and tear it down on failure automatically
        self.setup_output_dir(self.out_toktxtdir().path)

        #gather input files
        inputfiles = [ filename for filename in glob.glob(self.in_txtdir().path + '/*.txt') ]

        print('Running Tokenizer...')
        yield [ Tokenize_doc(inputfile=inputfile,outputdir=self.out_toktxtdir().path,tokconfig=self.tokconfig,strip_punctuation=self.strip_punctuation) for inputfile in inputfiles ]




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

#################################################################
### Components
#################################################################

@registercomponent
class Tokenize_doc(StandardWorkflowComponent): # connection between Tokenize_txtdir and Tokenize_document

    tokconfig = Parameter()
    strip_punctuation = BoolParameter()

    def autosetup(self):
        return Tokenize_document

    def accepts(self):
        return InputFormat(self, format_id='txt', extension='.txt')

@registercomponent
class Preprocess(StandardWorkflowComponent):

    tokconfig = Parameter(default=False)
    frogconfig = Parameter(default=False) # both should be set to False to be able to pick one of them
    strip_punctuation = BoolParameter()

    def accepts(self):
        return InputFormat(self, format_id='txt', extension='.txt'), InputFormat(self, format_id='txtdir', extension='txtdir',directory=True)

    def setup(self, workflow, input_feeds):

        if 'txt' in input_feeds.keys():
            # could either be frogged or tokenized according to the config that is given as argument
            if self.tokconfig:
                preprocess = workflow.new_task('tokenize_instances', Tokenize_instances, autopass=True, tokconfig=self.tokconfig, strip_punctuation=self.strip_punctuation)
                preprocess.in_txt = input_feeds['txt']
            elif self.frogconfig:
                preprocess = workflow.new_task('frog_instances', Frog_instances, autopass=True, frogconfig=self.frogconfig, strip_punctuation=self.strip_punctuation)
                preprocess.in_txt = input_feeds['txt']

        elif 'txtdir' in input_feeds.keys():
            # could either be frogged or tokenized according to the config that is given as argument
            if self.tokconfig:
                preprocess = workflow.new_task('tokenize_dir', Tokenize_txtdir, autopass=True, tokconfig=self.tokconfig, strip_punctuation=self.strip_punctuation)
                preprocess.in_txtdir = input_feeds['txtdir']
            elif self.frogconfig:
                preprocess = workflow.new_task('frog_dir', Frog_txtdir, autopass=True, frogconfig=self.frogconfig, strip_punctuation=self.strip_punctuation)
                preprocess.in_txtdir = input_feeds['txtdir']

        return preprocess
