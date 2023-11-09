from setuptools import setup, find_packages
from rblur import __version__

setup(
    name='rblur',
    version=__version__,

    url='https://github.com/ahmedshah1494/mllib',
    author='Muhammad Ahmed Shah',
    author_email='mshah1@cmu.edu',

    py_modules=find_packages(),

    install_requires=[
        'einops==0.4.1',
        'fastai==2.7.9',
        'transformers==4.21.1',
        'imageio==2.21.2',
        'Cython==3.0.5',
        'pysaliency==0.2.21',
        'lmdb==1.3.0',
        'IPython==8.4.0',
        'pysaliency=0.2.21'
    ],
)