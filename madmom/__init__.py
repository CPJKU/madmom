# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=wrong-import-position
"""
Madmom is an audio and music signal processing library.

This library is used internally by the Department of Computational Perception,
Johannes Kepler University, Linz, Austria (http://www.cp.jku.at) and the
Austrian Research Institute for Artificial Intelligence (OFAI), Vienna, Austria
(http://www.ofai.at).

Please see the README for further details of this package.

"""

from __future__ import absolute_import, division, print_function

import pkg_resources

# import all packages
from . import audio, evaluation, features, ml, models, processors, utils

# define a version variable
__version__ = pkg_resources.get_distribution("madmom").version

# keep namespace clean
del pkg_resources
