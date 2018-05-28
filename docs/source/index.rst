.. Quoll documentation master file, created by
   sphinx-quickstart on Mon May 28 10:30:13 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Quoll's documentation!
==================================

Quoll is the name of carnivorous marsupials living in Australia, New Guinea and Tazmania. It is also a python library for running NLP pipelines, based on the LuigiNLP_ workflow system. Quoll takes care of the sequence of tasks that are common to basic Machine Learning experiments with textual input: preprocessing, feature extraction, vectorizing, classification and evaluation. Provided that you prepare instances and labels, all these tasks can be ran as a pipeline, in one go. Quoll is built on top of several applications in the LaMachine_ Software distribution (Ucto_, Frog_, Colibri-core_, PyNLPl_), as well as popular python packages (Numpy_, Scipy_ and Scikit-learn_).     

.. _LuigiNLP: https://github.com/LanguageMachines/LuigiNLP
.. _LaMachine: https://github.com/proycon/LaMachine
.. _Ucto: https://languagemachines.github.io/ucto/
.. _Frog: https://languagemachines.github.io/frog/
.. _Colibri-core: https://proycon.github.io/colibri-core/
.. _PyNLPl: http://pynlpl.readthedocs.io/en/latest/
.. _Numpy: http://www.numpy.org/
.. _Scipy: https://www.scipy.org/
.. _Scikit-learn: http://scikit-learn.org/stable/

Quoll has the following advantages:

- Can run full supervised machine learning pipeline with one command.
- Stores intermediate output of the pipeline. 
- Maintains a full log of your experiments.
- Offers various options at each stage of the pipeline. 
- If part of the pipeline is already completed, will continue from that point. 
- Experiments with different settings can be distinguished based on filenames. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   The Quoll pipeline
      Preprocess
      Featurize
      



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
