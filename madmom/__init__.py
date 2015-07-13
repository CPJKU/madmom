# encoding: utf-8
"""
This package is used internally by the Department of Computational Perception,
Johannes Kepler University, Linz, Austria (http://www.cp.jku.at) and the
Austrian Research Institute for Artificial Intelligence (OFAI), Vienna, Austria
(http://www.ofai.at).

Please see the README for further details of this package.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import os

MODELS_PATH = '%s/models' % (os.path.dirname(__file__))

# keep namespace clean
del os

# finally import all submodules
from . import audio, features, evaluation, ml, utils, processors
