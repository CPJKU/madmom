#!/usr/bin/env python
# encoding: utf-8
"""
This file is deprecated. All filter related functionality should go into
madmom.audio.filters

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from .filters import *

import warnings
warnings.warn("madmom.audio.filterbank has been renamed to "
              "madmom.audio.filters, please update your code. "
              "This wrapper will be removed soon.")
