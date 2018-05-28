
The preprocess module
==================================

The preprocess module is the first module in the pipeline, taking care of basic text preprocessing tasks: tokenization (Ucto_ / Frog_), stemming and part-of-speech tagging (Frog_). Its output provides the input to the Featurize module. 

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
                        + Output: **.tok.txt** or **.tok.txt**

--frogconfig            + Give path to frog configuration file to tokenize, stem and part-of-speech tag the textdocuments applying Frog.
                        + Using Lamachine, the path can be found as path-to-lamachine-directory/share/frog/nld/frog.cfg
                        + Frog currently only supports the Dutch language. 
                        + Will not work if tokconfig is also specified.
                        + Output: **.frog.json** or **.frog.jsondir**
                        
--strip-punctuation     + Boolean option; default is False
                        + Choose to strip punctuation from text.

Output
-------
*The output comes in one of the following four extensions, depending on the setting (see Overview_)* 

:.tok.txt:
  File with each token divided by a space
:.tok.txtdir:
  Directory with .tok.txt files
:.frog.json:
  Json-file where each document is a list of tokens, and each token is a dictionary with keys 'text', 'lemma' and 'pos'
:.frog.jsondir:
  Directory with .frog.json files


Overview
--------

+---------+------------+---------------+
| Input   | Option     | Output        |
+=========+============+===============+
| .txt    | tokconfig  | .tok.txt      |
+---------+------------+---------------+
| .txt    | frogconfig | .frog.json    |
+---------+------------+---------------+
| .txtdir | tokconfig  | .tok.txtdir   |
+---------+------------+---------------+
| .txtdir | frogconfig | .frog.jsondir |
+---------+------------+---------------+

Examples of command line usage
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

