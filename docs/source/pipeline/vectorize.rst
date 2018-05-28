The Vectorize module
==================================

The Vectorize module is the third module in the pipeline, taking care of weighting and pruning features based on characteristics in the training data. In contrast to the preliminary Featurize_ and Preprocess_ modules, this module requires separate train and test input.

Input
--------

*If the input to Preprocess_ (.txt or .txtdir) or Featurize (.tok.txt, .tok.txtdir, .frog.json or .frog.jsondir) is given as input to traininstances or testinstances, this module is ran prior to the Vectorize module.* 

--traininstances                            + Traininstances takes featurized or vectorized documents as input. They can come in the following formats:
                                            1. Extension **.features.npz** - Output of the Featurize module. 
                                            2. Extension **.vectors.npz** - Output of the Vectorize module (can be used for vectorizing test documents).
                                            3. Extension **.csv** - File with feature vectors formatted as comma-separated-values (useful when applying feature extraction that is not accomodated by quoll). When working with \'.csv\'-input, a file with featurenames should be created that has the same path and name as the \'.csv\'-file, replacing .csv with .featureselection.txt.

--trainlabels                               + Extension **.labels** - File with a label per line, should be as many instances as traininstances, where the position of the label corresponds to the instance on the same position.  

--testinstances                             + Like traininstances, test instances takes featurized or vectorized documents as input. They can come in the following formats:
                                            1. Extension **.features.npz** - Output of the Featurize module. 
                                            2. Extension **.csv** - File with feature vectors formatted as comma-separated-values (useful when applying feature extraction that is not accomodated by quoll). Can only be used when the input to traininstances is of the same format; the number of columns should be as many as for the traininstances input. Like 


Options
--------
*Options for Preprocess_ and Featurize_ also apply and are effective in combination with the inputfiles for these modules.* 

--weight                    + Specify the feature weighting
                            + Does not work in combination with a \'.csv\'-file
                            + Options: **frequency**, **binary**, **tfidf**, **infogain
                            + For tfidf or infogain, it is recommended to set minimum feature frequency to 5 or 10 in the Featurize module
                            + String parameter; default: **frequency**

--prune                     + Specify the number of features to maintain after pruning
                            + Does not work in combination with a \'.csv\'-file
                            + Pruning is done by ranking features based on their feature weight (total count of a feature is taken in case of \'frequency\' and \'binary\' weighting  
                            + Integer parameter; default: 5000
                            
--balance                   + Choose to balance the number of train instances
                            + The number of instances for each class label is decreased to the instance count of the least frequent class 
                            + Can help in case of strong class skewness
                            + Boolean parameter; default: **False**

--delimiter                 + Specify the delimiter by which columns in the \'csv\'-file are separated
                            + Only applies to \'.csv\'-files
                            + String parameter; default: **,**
                        
--scale                     + Option to normalize feature values to the same scale
                            + Only applies to \'.csv\'-file
                            + Useful in combination with some classifiers, if the features in the \'.csv\'-file are from different sources and have a wide range of values 
                            + Boolean parameter; default: **False**

Output
-------
:.balanced.features.npz:
  Balanced instances 
  Only applied when \'balance\' is chosen
:.balanced.labels:
  Labels related to balanced instances 
  Only applies when \'balance\' is chosen
:.balanced.vocabulary:
  Vocabulary related to balanced instances

Overview
--------

+------------------+-----------------------+---------------+--------------------+------------------+--------------------------------+---------------------------------------------------------------------------------------+
| --inputfile      | --featuretypes        | --ngrams      | --blackfeats       | --lowercase      | --minimum-token-frequency      | Output                                                                                |
+==================+=======================+===============+====================+==================+================================+=======================================================================================+
| docs.tok.txt     | tokens                | \'1 2 3\'     | False              | True             | 2                              | + docs.tokens.n_1_2_3.min2.lower_True.black_False.features.npz                        |
|                  |                       |               |                    |                  |                                | + docs.tokens.n_1_2_3.min2.lower_True.black_False.vocabulary.txt                      |                     
+------------------+-----------------------+---------------+--------------------+------------------+--------------------------------+---------------------------------------------------------------------------------------+
| dos.frog.jsondir | \'tokens lemmas pos\' | 1             | \'koala kangaroo\' | False            | 10                             | + docs.tokens.lemmas.pos.n_1.min10.lower_False.black_koala_kangaroo.features.npz      |
|                  |                       |               |                    |                  |                                | + docs.tokens.lemmas.pos.n_1.min10.lower_False.black_koala_kangaroo.vocabulary.txt    |
+------------------+-----------------------+---------------+--------------------+------------------+--------------------------------+---------------------------------------------------------------------------------------+

Examples of command line usage
--------

**Extract word Ngrams from tokenized text document, lowercasing them and stripping away token Ngrams that occur less than 5 times**

$ luiginlp Featurize --module quoll.classification_pipeline.modules.featurize --inputfile docs.tok.txt --lowercase --minimum-token-frequency 5

**Extract lemma and pos Ngrams from directory with frogged texts**

$ luiginlp Featurize --module quoll.classification_pipeline.modules.featurize --inputfile docs.frog.jsondir --featuretypes \'lemmas pos\'

**Frog text document, extract text and pos features and strip away any feature with the word \'snake\'**

$ luiginlp Featurize --module quoll.classification_pipeline.modules.featurize --inputfile docs.txt --frogconfig /mylamachinedir/share/frog/nld/frog.cfg --featuretypes \'tokens pos\' --blackfeats snake

.. _ColibriCore: https://proycon.github.io/colibri-core/
.. _Preprocess: preprocess.rst
.. _Vectorize: vectorize.rst
