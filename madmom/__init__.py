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

import numpy as np
import pkg_resources

# import all packages
from . import audio, evaluation, features, ml, models, processors, utils

# define a version variable
__version__ = pkg_resources.get_distribution("madmom").version

# keep namespace clean
del pkg_resources


# set and restore numpy's print options for doctests
_NP_PRINT_OPTIONS = np.get_printoptions()


def setup():
    # pylint: disable=missing-docstring
    # sets up the environment for doctests (when run through nose)
    np.set_printoptions(precision=5, edgeitems=2, suppress=True)


def teardown():
    # pylint: disable=missing-docstring
    # restore the environment after doctests (when run through nose)
    np.set_printoptions(**_NP_PRINT_OPTIONS)
