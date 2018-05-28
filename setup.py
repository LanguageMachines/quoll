#! /usr/bin/env python3
# -*- coding: utf8 -*-


import os
import sys
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name = "quoll",
    version = "0.1",
    author = "Florian Kunneman, Maarten van Gompel, Wessel Stoop, Louis Onrust, Suzan Verberne, Iris Hendrikx, Ali Hürriyetoğlu",
    author_email = "f.kunneman@let.ru.nl",
    description = ("Quoll NLP Pipeline"),
    license = "GPL",
    keywords = "nlp computational_linguistics",
    url = "https://github.com/LanguageMachines/quoll",
    packages=['quoll','quoll.classification_pipeline', 'quoll.classification_pipeline.functions', 'quoll.classification_pipeline.modules'],
    long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Text Processing :: Linguistic",
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    zip_safe=False,
    #include_package_data=True,
    #package_data = {'': ['*.wsgi','*.js','*.xsl','*.gif','*.png','*.xml','*.html','*.jpg','*.svg','*.rng'] },
    install_requires=['LuigiNLP','pynlpl','gensim','scikit-learn','numpy','scipy','colibricore'],
    #entry_points = {    'console_scripts': [
    #        'luiginlp = luiginlp.luiginlp:main',
    #]
    #}
)
