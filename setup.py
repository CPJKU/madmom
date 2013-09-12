#!/usr/bin/env python
# encoding: utf-8
"""
This file contains the setup for setuptools to distribute as a package.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from distutils.core import setup

setup(name='cpjku',
      version='0.01',
      description='Python package used at cp.jku.at',
      long_description=open('README').read(),
      author='Sebastian Böck, Department of Computational Perception, Johannes Kepler University, Linz, Austria',
      author_email='sebastian.boeck@jku.at',
      url='http://www.cp.jku.at',
      license='BSD, with some exclusions'
      )
