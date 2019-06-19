The Featurize module
==================================

The Featurize module is the second module in the pipeline, taking care of feature extraction from the output of the Preprocess_ module. It makes use of ColibriCore_ to count features, and its output forms the input to the Vectorize_ module. 

Input
--------

If the input to Preprocess_ (.txt or .txtdir) is given as inputfile, this module is ran prior to the Featurize module.

--inputfile                 + The featurize module takes preprocessed documents as input. They can come in two formats:
                            1. Extension **.preprocessed.frog** - File with tokenized text documents on each line. 
                            2. Extension **.preprocessdir** - Directory with tokenized text documents (files ending with **.tok.txt**).                

Options
--------

--featuretypes              + Specify the types of features to extract
                            + Options: **tokens**, **lemmas**, **pos**, **chars**
                            + lemmas and pos only apply when frog is used during preprocessing
                            + multiple options can be given with quotes, divided by a space (for example: *\'chars tokens pos\'*)
                            + String parameter; default: **tokens**

--ngrams                    + Specify the length of the N-grams that you want to include
                            + Ngram values should be given within quotes, divided by a space (for example: *\'1 2\'*)
                            + String parameter; default: **\'1 2 3\'**
                            
--blackfeats                + In order to exclude words from the feature space, specify them here
                            + Each feature to be excluded should be given within quotes, divided by a space (for example: *\'do re mi\'*)
                            + Each ngram within the feature space that includes any of the given blackfeats will be removed
                            + String parameter; default: **False**

--lowercase                 + Choose to lowercase all text and lemma features
                            + Boolean parameter; default: **False**
                        
--minimum-token-frequency   + Option to delete all features that occur below the given threshold
                            + Recommended to set to 5 or 10 when applying tfidf or infogain weighting in the Vectorize module
                            + Integer parameter; default: **1**

Output
-------
:.features.npz:
Binary file in Numpy format, storing the extracted features per document in sparse form 
:.vocabulary.txt:
File that stores the index of each feature

Overview
--------

+------------------------+-----------------------+-----------+--------------------+--------------+----------------------------+
| --inputfile            | --featuretypes        | --ngrams  | --blackfeats       | --lowercase  | --minimum-token-frequency  |                                                                              
+========================+=======================+===========+====================+==============+============================+
| docs.preprocessed.json | \'tokens chars\'      | \'1 2 3\' | False              | True         | 2                          | 
+------------------------+-----------------------+-----------+--------------------+--------------+----------------------------+
| docs.preprocessdir     | \'tokens lemmas pos\' | 1         | \'koala kangaroo\' | False        | 10                         |
+------------------------+-----------------------+-----------+--------------------+--------------+----------------------------+

Examples of command line usage
--------

**Extract word Ngrams from preprocessed text document, lowercasing them and stripping away Ngrams that occur less than 5 times**

$ luiginlp Featurize --module quoll.classification_pipeline.modules.featurize --inputfile docs.preprocessed.json --lowercase --minimum-token-frequency 5

**Extract lemma and pos Ngrams from directory with frogged texts**

$ luiginlp Featurize --module quoll.classification_pipeline.modules.featurize --inputfile docs.preprocessdir --featuretypes \'lemmas pos\'

**Frog text document, extract character, text and pos features and strip away any feature with the word \'snake\'**

$ luiginlp Featurize --module quoll.classification_pipeline.modules.featurize --inputfile docs.txt --frogconfig /mylamachinedir/share/frog/nld/frog.cfg --featuretypes \'tokens pos chars\' --blackfeats snake

.. _ColibriCore: https://proycon.github.io/colibri-core/
.. _Preprocess: preprocess.rst
.. _Vectorize: vectorize.rst
