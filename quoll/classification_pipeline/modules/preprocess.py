
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter

import frog
import ucto

import glob
import json

class Tokenize_document(Task):
    """"
    Tokenizes a document
    """

    in_txt = InputSlot()

    tokconfig = Parameter()
    strip_punctuation = BoolParameter()

    def out_preprocessed(self):
        return self.outputfrominput(inputformat='txt', stripextension='.txt', addextension='.preprocessed.json')

    def run(self):

        # open file
        with open(self.in_txt().path, 'r', encoding = 'utf-8') as file_in:
            lines = file_in.read().strip().split('\n')

        # initialize tokenizer
        tokenizer = ucto.Tokenizer(self.tokconfig)

        # for each line
        documents = []
        for line in lines:
            # tokenize
            tokenizer.process(line)
            # initialize sentences
            sentences = []
            sentence = []
            for token in tokenizer:
                # add each token to the sentence...
                if not (self.strip_punctuation and token.tokentype == 'PUNCTUATION'):
                    sentence.append({'text':token.text})
                # ...until the sentence ends
                if token.isendofsentence():
                    sentences.append(sentence)
                    # initialize a new sentence
                    sentence = []
            # scrape last bits of info
            if len(sentence) > 0:
                sentences.append(sentence)
            documents.append({'raw':line,'processed':sentences})

        with open(self.out_preprocessed().path,'w',encoding = 'utf-8') as file_out:
            json.dump(documents,file_out)

class Tokenize_txtdir(Task):
    """"
    Tokenizes a directory with files
    """

    in_txtdir = InputSlot()

    tokconfig = Parameter()
    strip_punctuation = BoolParameter()

    def out_preprocessdir(self):
        return self.outputfrominput(inputformat='txtdir', stripextension='.txtdir', addextension='.preprocessdir')

    def run(self):
        #Set up the output directory, will create it and tear it down on failure automatically
        self.setup_output_dir(self.out_toktxtdir().path)

        #gather input files
        inputfiles = [ filename for filename in glob.glob(self.in_txtdir().path + '/*.txt') ]

        print('Running Tokenizer...')
        yield [ Tokenize_doc(inputfile=inputfile,outputdir=self.out_preprocessdir().path,tokconfig=self.tokconfig,strip_punctuation=self.strip_punctuation) for inputfile in inputfiles ]


class Frog_document(Task):
    """"
    Frogs a file one document per line
    """

    in_txt = InputSlot()

    frogconfig = Parameter()
    strip_punctuation = BoolParameter()

    def out_preprocessed(self):
        return self.outputfrominput(inputformat='txt', stripextension='.txt', addextension='.preprocessed.json')

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
            print(line)
            frogged = frogger.process(line)
            # initialize sentences
            sentences = []
            sentence = []
            # add each token to the sentence...
            for token in frogged:
                if not (self.strip_punctuation and token['pos'] == 'LET()'):
                    sentence.append({'text':token['text'], 'lemma':token['lemma'], 'pos':token['pos'].split('(')[0]})
                # ...until the sentence ends
                if 'eos' in token:
                    sentences.append(sentence)
                    # initialize a new sentence
                    sentence = []
            # scrape last bits of info
            if len(sentence) > 0:
                sentences.append(sentence)
            # when finished add sentences as new document, along with the original raw line
            documents.append({'raw':line,'processed':sentences})

        # write output
        with open(self.out_preprocessed().path,'w',encoding = 'utf-8') as file_out:
            json.dump(documents,file_out)

class Frog_txtdir(Task):
    """"
    Tokenizes a directory with files
    """

    in_txtdir = InputSlot()

    frogconfig = Parameter()
    strip_punctuation = BoolParameter()

    def out_preprocessdir(self):
        return self.outputfrominput(inputformat='txtdir', stripextension='.txtdir', addextension='.preprocessdir')

    def run(self):
        #Set up the output directory, will create it and tear it down on failure automatically
        self.setup_output_dir(self.out_frogjsondir().path)

        #gather input files
        inputfiles = [ filename for filename in glob.glob(self.in_txtdir().path + '/*.txt') ]

        print('Running Frogger...')
        yield [ Frog_doc(inputfile=inputfile,outputdir=self.out_preprocessdir().path,frogconfig=self.frogconfig,strip_punctuation=self.strip_punctuation) for inputfile in inputfiles ]

#################################################################
### Components
#################################################################

@registercomponent
class Tokenize_doc(StandardWorkflowComponent): 
    """
    connection between Tokenize_txtdir and Tokenize_document
    """

    tokconfig = Parameter()
    strip_punctuation = BoolParameter()

    def autosetup(self):
        return Tokenize_document

    def accepts(self):
        return InputFormat(self, format_id='txt', extension='.txt')

@registercomponent
class Frog_doc(StandardWorkflowComponent): 
    """
    connection between Frog_txtdir and Frog_instances
    """

    frogconfig = Parameter()
    strip_punctuation = BoolParameter()

    def autosetup(self):
        return Frog_instances

    def accepts(self):
        return InputFormat(self, format_id='txt', extension='.txt')

@registercomponent
class Preprocess(StandardWorkflowComponent):
    """
    The preprocess module
    ==================================

    The preprocess module is the first module in the pipeline, taking care of basic text preprocessing tasks: tokenization (Ucto_ / Frog_), stemming and part-of-speech tagging (Frog_). Its output provides the input to the Featurize_ module. 
    
    Input
    --------
    
    --inputfile             + The preprocess module takes text documents as input. They can come in two formats:
    1. Extension **.txt** - File with text documents on each line. Note that text documents with a linebreak will be seen as two separate documents
    2. Extension **.txtdir** - Directory with text documents (files ending with **.txt**).
    
    
    Options
    --------

--tokconfig             + Give path to Ucto configuration file to tokenize the text documents applying Ucto
                        + Using LaMachine, the path can be found in path-to-lamachine-directory/share/ucto/
                        + Ucto supports several languages
                        + Will not work if frogconfig is also specified 
                        + Output: **.preprocessed.json** or **.preprocessdir**

--frogconfig            + Give path to frog configuration file to tokenize, stem and part-of-speech tag the textdocuments applying Frog.
                        + Using Lamachine, the path can be found as path-to-lamachine-directory/share/frog/nld/frog.cfg
                        + Frog currently only supports the Dutch language. 
                        + Will not work if tokconfig is also specified.
                        + Output: **.preprocessed.json** or **.preprocessdir**
                        
--strip-punctuation     + Boolean option; default is False
                        + Choose to strip punctuation from text.

Output
-------
The output comes in one of the following two extensions, depending on the input (repectively **.txt** and **.txtdir** (see `Overview`_). 

:.preprocessed.json:
  Json-file where each document is comes in the form of the raw text (key 'raw') and the processed text (key 'processed'), as a list of tokens, where each token is a dictionary with keys 'text' (and 'lemma' and 'pos' if frog is used)
:.frog.jsondir:
  Directory with .preprocessed.json files per document


Overview
--------

+---------+------------+--------------------+
| Input   | Option     | Output             |
+=========+============+====================+
| .txt    | tokconfig  | .preprocessed.json |
+---------+------------+--------------------+
| .txt    | frogconfig | .preprocessed.json |
+---------+------------+--------------------+
| .txtdir | tokconfig  | .preprocessdir     |
+---------+------------+--------------------+
| .txtdir | frogconfig | .preprocessdir     |
+---------+------------+--------------------+

Command line examples 
--------

**Tokenize text document and strip punctuation**

$ luiginlp Preprocess --module quoll.classification_pipeline.modules.preprocess --inputfile docs.txt --tokconfig /mylamachinedir/share/ucto/tokconfig-nld --strip-punctuation

**Tokenize directory with text documents**

$ luiginlp Preprocess --module quoll.classification_pipeline.modules.preprocess --inputfile docs.txtdir --tokconfig /mylamachinedir/share/ucto/tokconfig-nld

**Frog text document**

$ luiginlp Preprocess --module quoll.classification_pipeline.modules.preprocess --inputfile docs.txt --frogconfig /mylamachinedir/share/frog/nld/frog.cfg

**Frog directory with text documents and strip punctuation**

$ luiginlp Preprocess --module quoll.classification_pipeline.modules.preprocess --inputfile docs.txt --frogconfig /mylamachinedir/share/frog/nld/frog.cfg --strip-punctuation

.. _Ucto: https://languagemachines.github.io/ucto/
.. _Frog: https://languagemachines.github.io/frog/
.. _Featurize: featurize.rst

    """

    tokconfig = Parameter(default=False)
    frogconfig = Parameter(default=False) # both are set to False to be able to pick one of them
    strip_punctuation = BoolParameter()

    def accepts(self):
        return InputFormat(self, format_id='txt', extension='.txt'), InputFormat(self, format_id='txtdir', extension='txtdir',directory=True)

    def setup(self, workflow, input_feeds):

        assert self.tokconfig != False or self.frogconfig != False, 'No config file included, specify a ucto config using \'--tokconfig\' or a frog config using \'frogconfig\''
        assert not (self.tokconfig and self.frogconfig), 'Both a tokconfig and a frogconfig included, only one of the two required'
        
        if 'txt' in input_feeds.keys():
            # could either be frogged or tokenized according to the config that is given as argument
            if self.tokconfig:
                preprocess = workflow.new_task('tokenize_document', Tokenize_document, autopass=True, tokconfig=self.tokconfig, strip_punctuation=self.strip_punctuation)
                preprocess.in_txt = input_feeds['txt']
            elif self.frogconfig:
                preprocess = workflow.new_task('frog_document', Frog_document, autopass=True, frogconfig=self.frogconfig, strip_punctuation=self.strip_punctuation)
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
