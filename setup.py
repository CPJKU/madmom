#!/usr/bin/env python
# encoding: utf-8
"""
This file contains the setup for setuptools to distribute everything as a
package.

"""

from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

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
           'madmom.evaluation.onsets',
           'madmom.evaluation.beats',
           'madmom.evaluation.notes',
           'madmom.evaluation.tempo',
           'madmom.evaluation.alignment']

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

classifiers = ['Development Status :: 3 - Alpha',
               'Programming Language :: Python :: 2.7',
               'Environment :: Console',
               'License :: OSI Approved :: BSD License',
               'License :: Free for non-commercial use',
               'Topic :: Multimedia :: Sound/Audio :: Analysis',
               'Topic :: Scientific/Engineering :: Artificial Intelligence']

setup(name='madmom',
      version='0.10.2',
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
      package_data={'madmom': ['models/*']},
      cmdclass={'build_ext': build_ext},
      classifiers=classifiers)
