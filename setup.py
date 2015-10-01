#!/usr/bin/env python
# encoding: utf-8
"""
This file contains the setup for setuptools to distribute everything as a
(PyPI) package.

"""

from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

# define the modules to be included in the PyPI package
modules = ['madmom.audio',
           'madmom.audio.ffmpeg',
           'madmom.audio.signal',
           'madmom.audio.filters',
           'madmom.audio.comb_filters',
           'madmom.audio.stft',
           'madmom.audio.spectrogram',
           'madmom.features',
           'madmom.features.onsets',
           'madmom.features.beats',
           'madmom.features.notes',
           'madmom.features.tempo',
           'madmom.ml',
           'madmom.ml.gmm',
           'madmom.ml.rnn',
           'madmom.ml.hmm',
           'madmom.utils',
           'madmom.utils.midi',
           'madmom.utils.stats',
           'madmom.evaluation.onsets',
           'madmom.evaluation.beats',
           'madmom.evaluation.notes',
           'madmom.evaluation.tempo',
           'madmom.evaluation.alignment']

# define the models to be included in the PyPI package
# Note: we need to explicitly define which data to include in the package since
#       pip install does not properly process non-module entries in MANIFEST.in
package_data = ['models/LICENSE',
                'models/README.rst',
                'models/beats/*/*',
                'models/downbeats/*/*',
                'models/notes/*/*',
                'models/onsets/*/*']

# define to be compiled extensions in the PyPI package
extensions = [Extension('madmom.ml.rnn',
                        ['madmom/ml/rnn.py', 'madmom/ml/rnn.pxd'],
                        include_dirs=[np.get_include()]),
              Extension('madmom.audio.comb_filters',
                        ['madmom/audio/comb_filters.pyx'],
                        include_dirs=[np.get_include()]),
              Extension('madmom.features.beats_crf',
                        ['madmom/features/beats_crf.pyx'],
                        include_dirs=[np.get_include()]),
              Extension('madmom.features.beats_hmm',
                        ['madmom/features/beats_hmm.pyx'],
                        include_dirs=[np.get_include()]),
              Extension('madmom.ml.hmm',
                        ['madmom/ml/hmm.pyx'],
                        include_dirs=[np.get_include()])]

# some PyPI metadata
classifiers = ['Development Status :: 3 - Alpha',
               'Programming Language :: Python :: 2.7',
               'Environment :: Console',
               'License :: OSI Approved :: BSD License',
               'License :: Free for non-commercial use',
               'Topic :: Multimedia :: Sound/Audio :: Analysis',
               'Topic :: Scientific/Engineering :: Artificial Intelligence']

# installation requirements
install_requires = ['numpy>=1.8.1',
                    'scipy>=0.14',
                    'cython>=0.22.1']

# the actual setup routine
setup(name='madmom',
      version='0.11',
      description='Python audio signal processing library',
      long_description=open('README.rst').read(),
      author='Department of Computational Perception, Johannes Kepler '
             'University, Linz, Austria and Austrian Research Institute for '
             'Artificial Intelligence (OFAI), Vienna, Austria',
      author_email='madmom-users@googlegroups.com',
      url='https://github.com/CPJKU/madmom',
      license='BSD, CC BY-NC-SA',
      py_modules=modules,
      ext_modules=extensions,
      packages=['madmom'],
      package_data={'madmom': package_data},
      cmdclass={'build_ext': build_ext},
      install_requires=install_requires,
      classifiers=classifiers)
