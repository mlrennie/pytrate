#!/usr/bin/env python3

import sys
import numpy

if sys.version_info[0] < 3:
    sys.exit('Sorry, Python < 3.x is not supported')

# Try using setuptools first, if it's installed
from setuptools import setup, find_packages

# Need to add all dependencies to setup as we go!
setup(name='pytrate',
      packages=find_packages(),
      version='0.1',
      description="Python software package for analyzing multi-titration data",
      author='Martin L. Rennie',
      author_email='martinlrennie@gmail.com',
      url='https://github.com/mlrennie/pytrate',
      download_url='https://github.com/mlrennie/pytrate',
      zip_safe=False,
      install_requires=["matplotlib","scipy","numpy","emcee","corner"],
      classifiers=['Programming Language :: Python'])