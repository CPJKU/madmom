#!/usr/bin/env python
# encoding: utf-8
"""
This file contains the setup for setuptools to distribute everything as a
(PyPI) package.

"""

from setuptools import setup, find_packages
from distutils.extension import Extension

from Cython.Build import cythonize, build_ext

import glob
import numpy as np

# define version
version = '0.16.dev0'

# define which extensions to compile
include_dirs = [np.get_include()]

extensions = [
    Extension('madmom.audio.comb_filters', ['madmom/audio/comb_filters.pyx'],
              include_dirs=include_dirs),
    Extension('madmom.features.beats_crf', ['madmom/features/beats_crf.pyx'],
              include_dirs=include_dirs),
    Extension('madmom.ml.hmm', ['madmom/ml/hmm.pyx'],
              include_dirs=include_dirs),
    Extension('madmom.ml.nn.layers', ['madmom/ml/nn/layers.py'],
              include_dirs=include_dirs),
]

# define scripts to be installed by the PyPI package
scripts = glob.glob('bin/*')

# define the models to be included in the PyPI package
package_data = ['models/LICENSE',
                'models/README.rst',
                'models/beats/201[56]/*',
                'models/chords/*/*',
                'models/chroma/*/*',
                'models/downbeats/*/*',
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

# requirements
requirements = ['numpy>=1.8.1',
                'scipy>=0.14',
                'cython>=0.22.1',
                ]

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
      ext_modules=cythonize(extensions),
      package_data={'madmom': package_data},
      exclude_package_data={'': ['tests', 'docs']},
      scripts=scripts,
      install_requires=requirements,
      cmdclass={'build_ext': build_ext},
      test_suite='nose.collector',
      classifiers=classifiers)
