# encoding: utf-8
"""
This file contains the speed critical parts of neural network related
functionality.

Note: right now, this file is just an empty augmenting file. However, it
increases performance when cython is used to compile the normal layers.py file.

"""

from __future__ import absolute_import, division, print_function

import cython

import numpy as np
cimport numpy as np
