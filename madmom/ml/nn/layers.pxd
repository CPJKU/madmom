# encoding: utf-8
"""
This file contains the speed critical parts of ml.nn.layers module.

Note: right now, this file is just an empty augmenting file. However, it
increases performance when cython is used to compile layers.py.

"""

from __future__ import absolute_import, division, print_function

import cython

import numpy as np
cimport numpy as np
