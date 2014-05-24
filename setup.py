#!/usr/bin/env python
# encoding: utf-8
"""
This file contains the setup for setuptools to distribute as a package.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

extensions = [Extension('madmom.ml.fast_rnn',
                        ['madmom/ml/fast_rnn.pyx'],
                        include_dirs=[np.get_include()]),
              Extension('madmom.ml.rnn',
                        ['madmom/ml/rnn.py', 'madmom/ml/rnn.pxd'],
                        include_dirs=[np.get_include()])]

setup(name='madmom',
      version='0.01',
      description='Python package used at cp.jku.at and ofai.at',
      long_description=open('README').read(),
      author='Department of Computational Perception, Johannes Kepler '
             'University, Linz, Austria and Austrian Research Institute for '
             'Artificial Intelligence (OFAI), Vienna, Austria',
      author_email='python-sigk@jku.at',
      url='https://jobim.ofai.at/gitlab/madmom/madmom',
      license='BSD, with some exclusions',
      ext_modules=extensions,
      cmdclass={'build_ext': build_ext})
