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

import doctest
import numpy as np
import pkg_resources

# import all packages
from . import audio, evaluation, features, ml, models, processors, utils

# define a version variable
__version__ = pkg_resources.get_distribution("madmom").version

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


# Create a doctest output checker that optionally ignores the unicode string
# literal.

# declare the new doctest directive IGNORE_UNICODE
IGNORE_UNICODE = doctest.register_optionflag("IGNORE_UNICODE")
doctest.IGNORE_UNICODE = IGNORE_UNICODE
doctest.__all__.append("IGNORE_UNICODE")
doctest.COMPARISON_FLAGS = doctest.COMPARISON_FLAGS | IGNORE_UNICODE

_doctest_OutputChecker = doctest.OutputChecker


class MadmomOutputChecker(_doctest_OutputChecker):
    def check_output(self, want, got, optionflags):
        super_check_output = _doctest_OutputChecker.check_output
        if optionflags & IGNORE_UNICODE:
            import sys
            import re
            if sys.version_info[0] > 2:
                want = re.sub("u'(.*?)'", "'\\1'", want)
                want = re.sub('u"(.*?)"', '"\\1"', want)
            return super_check_output(self, want, got, optionflags)
        else:
            return super_check_output(self, want, got, optionflags)

    def output_difference(self, example, got, optionflags):
        super_output_difference = _doctest_OutputChecker.output_difference
        return super_output_difference(self, example, got, optionflags)

# monkey-patching
doctest.OutputChecker = MadmomOutputChecker

# keep namespace clean
del pkg_resources, doctest
