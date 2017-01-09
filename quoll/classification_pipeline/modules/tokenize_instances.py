

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

@registercomponent
class Tokenize(StandardWorkflowComponent):
    
    tokconfig = Parameter()
    strip_punctuation = BoolParameter()

    def autosetup(self):
        return Tokenize_instances

    def accepts(self):
        return InputFormat(self, format_id='txt', extension='.txt')
