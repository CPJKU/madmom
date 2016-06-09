#!/usr/bin/env python
# encoding: utf-8
"""
This file contains the setup for setuptools to distribute everything as a
(PyPI) package.

"""

from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext

import glob
import numpy as np

# define version
version = '0.13.2'

# define which extensions need to be compiled
extensions = [Extension('madmom.ml.rnn',
                        ['madmom/ml/rnn.py', 'madmom/ml/rnn.pxd'],
                        include_dirs=[np.get_include()]),
              Extension('madmom.audio.comb_filters',
                        ['madmom/audio/comb_filters.pyx'],
                        include_dirs=[np.get_include()]),
              Extension('madmom.features.beats_crf',
                        ['madmom/features/beats_crf.pyx'],
                        include_dirs=[np.get_include()]),
              Extension('madmom.ml.hmm',
                        ['madmom/ml/hmm.pyx'],
                        include_dirs=[np.get_include()])]

# define scripts to be installed by the PyPI package
scripts = glob.glob('bin/*')

# define the models to be included in the PyPI package
package_data = ['models/LICENSE',
                'models/README.rst',
                'models/beats/*/*',
                'models/notes/*/*',
                'models/onsets/*/*',
                'models/patterns/*/*',
                ]

# some PyPI metadata
classifiers = ['Development Status :: 3 - Alpha',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.3',
               'Programming Language :: Python :: 3.4',
               'Programming Language :: Python :: 3.5',
               'Environment :: Console',
               'License :: OSI Approved :: BSD License',
               'License :: Free for non-commercial use',
               'Topic :: Multimedia :: Sound/Audio :: Analysis',
               'Topic :: Scientific/Engineering :: Artificial Intelligence']

# docs to be included
long_description = open('README.rst').read()
long_description += '\n' + open('CHANGES.rst').read()

# the actual setup routine
setup(name='madmom',
      version=version,
      description='Python audio signal processing library',
      long_description=long_description,
      author='Department of Computational Perception, Johannes Kepler '
             'University, Linz, Austria and Austrian Research Institute for '
             'Artificial Intelligence (OFAI), Vienna, Austria',
      author_email='madmom-users@googlegroups.com',
      url='https://github.com/CPJKU/madmom',
      license='BSD, CC BY-NC-SA',
      packages=find_packages(exclude=['tests', 'docs']),
      ext_modules=extensions,
      package_data={'madmom': package_data},
      exclude_package_data={'': ['tests', 'docs']},
      scripts=scripts,
      cmdclass={'build_ext': build_ext},
      test_suite='nose.collector',
      classifiers=classifiers)
