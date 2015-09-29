# encoding: utf-8
"""
This file contains the speed critical parts of recurrent neural network (RNN)
related functionality.

Note: right now, this file is just an empty augmenting file. However, it
increases performance when cython is used to compile the normal rnn.py file.

"""
import cython

import numpy as np
cimport numpy as np
