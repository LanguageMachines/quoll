
from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter

import ucto
import glob

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
#                print(len(' '.join(tokens).split('\n')))
#                    print('Too long!! Inputline',line,'\nTokens',tokens)
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



@registercomponent
class Tokenize(StandardWorkflowComponent):

    tokconfig = Parameter()
    strip_punctuation = BoolParameter()

    def autosetup(self):
        return Tokenize_instances

    def accepts(self):
        return InputFormat(self, format_id='txt', extension='.txt')

@registercomponent
class Tokenize_doc(StandardWorkflowComponent):

    tokconfig = Parameter()
    strip_punctuation = BoolParameter()

    def autosetup(self):
        return Tokenize_document

    def accepts(self):
        return InputFormat(self, format_id='txt', extension='.txt')


@registercomponent
class Tokenize_dir(StandardWorkflowComponent):

    tokconfig = Parameter()
    strip_punctuation = BoolParameter()

    def autosetup(self):
        return Tokenize_txtdir

    def accepts(self):
        return InputFormat(self, format_id='txtdir', extension='.txtdir')
