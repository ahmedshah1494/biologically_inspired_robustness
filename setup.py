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
    ],
)