

from luiginlp.engine import Task, StandardWorkflowComponent, InputFormat, InputComponent, registercomponent, InputSlot, Parameter, BoolParameter
import ucto

class Tokenize_instances(Task):
    """"Tokenizes a file one document per line"""
    in_txt = InputSlot()

    tokconfig = Parameter()
    strip_punctuation = BoolParameter()

    def out_tokenized(self):
        return self.outputfrominput(inputformat='txt', stripextension='.txt', addextension='.tok.txt')

    def run(self):

        print('Running Tokenizer...')

        with open(self.in_txt().path, 'r', encoding = 'utf-8') as file_in:
            lines = file_in.readlines()

        with open(self.out_tokenized().path,'w',encoding = 'utf-8') as file_out:
            tokenizer = ucto.Tokenizer(self.tokconfig)
            for line in lines:
                tokenizer.process(line)
                tokens = []
                for token in tokenizer:
                    if not (self.strip_punctuation and token.tokentype == 'PUNCTUATION'):
                        tokens.append(token.text)
                file_out.write(' '.join(tokens) + '\n')

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
        yield [ Tokenize(inputfile=inputfile,outputdir=self.out_toktxtdir().path,tokconfig=self.tokconfig,strip_punctuation=self.strip_punctuation) for inputfile in inputfiles ]



@registercomponent
class Tokenize(StandardWorkflowComponent):

    tokconfig = Parameter()
    strip_punctuation = BoolParameter()

    def autosetup(self):
        return Tokenize_instances

    def accepts(self):
        return InputFormat(self, format_id='txt', extension='.txt')

class Tokenize_dir(StandardWorkflowComponent):

    tokconfig = Parameter()
    strip_punctuation = BoolParameter()

    def autosetup(self):
        return Tokenize_dir

    def accepts(self):
        return InputFormat(self, format_id='txtdir', extension='.txtdir')
