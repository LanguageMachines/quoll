import os
import glob
import sys
import shutil
from luiginlp.util import getlog
from luiginlp.engine import Task, StandardWorkflowComponent, registercomponent, InputComponent, Parallel, run, InputFormat, InputSlot, Parameter, BoolParameter
import luiginlp.modules.frog
import luiginlp.modules.ucto

log = getlog()

@registercomponent
class AnnotateDocumentComponent(StandardWorkflowComponent):
    language = Parameter()

    sentenceperline = BoolParameter()

    lemma = BoolParameter()
    pos = BoolParameter()
    ner = BoolParameter()
    morph = BoolParameter()
    shallowparse = BoolParameter()
    dep = BoolParameter()

    def accepts(self):
        if self.language == 'nld':
            return ( InputFormat(self, format_id='txt',extension='txt'), InputFormat(self, format_id='tok',extension='tok'), InputFormat(self, format_id='folia',extension='folia.xml'))
        else:
            raise NotImplementedError("Language " + self.language + " is not supported yet")

    def setup(self, workflow, input_feeds):
        if self.language == 'nld':
            doFrog = self.lemma or self.pos or self.ner or self.morph or self.dep or self.shallowparse
            frog_skip = ''
            if not self.dep:
                frog_skip += 'mp'
            if not self.shallowparse:
                frog_skip += 'c'
            if not self.ner:
                frog_skip += 'n'
            if not self.lemma and not self.ner and not self.dep and not self.shallowparse:
                frog_skip += 'l'
            if not self.morph:
                frog_skip += 'a'

            if 'txt' in input_feeds:
                #untokenized input
                if doFrog:
                    frog = workflow.new_task('frog', luiginlp.modules.frog.Frog_txt2folia, language=self.language, tok_input_sentenceperline=self.sentenceperline, skip=frog_skip)
                    frog.in_txt = input_feeds['txt']
                    return frog
                else:
                    ucto = workflow.new_task('ucto', luiginlp.modules.ucto.Ucto_txt2folia, language=self.language, tok_input_sentenceperline=self.sentenceperline)
                    ucto.in_txt = input_feeds['txt']
                    return ucto
            elif 'tok' in input_feeds:
                #pre-tokenized input
                if doFrog:
                    frog = workflow.new_task('frog', luiginlp.modules.frog.Frog_txt2folia, language=self.language, tok_input_sentenceperline=self.sentenceperline, skip='t' +frog_skip)
                    frog.in_txt = input_feeds['tok']
                    return frog
                else:
                    ucto = workflow.new_task('ucto', luiginlp.modules.ucto.Ucto_tok2folia, language=self.language, tok_input_sentenceperline=self.sentenceperline)
                    ucto.in_txt = input_feeds['tok']
                    return ucto
            elif 'folia' in input_feeds:
                if doFrog:
                    frog = workflow.new_task('frog', luiginlp.modules.frog.Frog_folia2folia, language=self.language, skip=frog_skip)
                    frog.in_folia = input_feeds['folia']
                    return frog
                else:
                    ucto = workflow.new_task('ucto', luiginlp.modules.ucto.Ucto_folia2folia, language=self.language)
                    ucto.in_folia = input_feeds['folia']
                    return ucto
        else:
            raise NotImplementedError("Language " + self.language + " is not supported yet")

