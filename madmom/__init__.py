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
from . import audio, evaluation, features, io, ml, models, processors, utils

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

# declare the new doctest directives
IGNORE_UNICODE = doctest.register_optionflag("IGNORE_UNICODE")
doctest.IGNORE_UNICODE = IGNORE_UNICODE
doctest.__all__.append("IGNORE_UNICODE")
doctest.COMPARISON_FLAGS = doctest.COMPARISON_FLAGS | IGNORE_UNICODE

NORMALIZE_ARRAYS = doctest.register_optionflag("NORMALIZE_ARRAYS")
doctest.NORMALIZE_ARRAYS = NORMALIZE_ARRAYS
doctest.__all__.append("NORMALIZE_ARRAYS")
doctest.COMPARISON_FLAGS = doctest.COMPARISON_FLAGS | NORMALIZE_ARRAYS

_doctest_OutputChecker = doctest.OutputChecker


class MadmomOutputChecker(_doctest_OutputChecker):
    """
    Output checker which enhances `doctest.OutputChecker` to compare doctests
    and computed output with additional flags.

    """

    def check_output(self, want, got, optionflags):
        """
        Return 'True' if the actual output from an example matches the
        expected.

        Parameters
        ----------
        want : str
            Expected output.
        got : str
            Actual output.
        optionflags : int
            Comparison flags.

        Returns
        -------
        bool
            'True' if the output maches the expectation.

        """
        import re
        import sys
        if optionflags & IGNORE_UNICODE and sys.version_info[0] > 2:
            # remove unicode indicators
            want = re.sub("u'(.*?)'", "'\\1'", want)
            want = re.sub('u"(.*?)"', '"\\1"', want)
        if optionflags & NORMALIZE_ARRAYS:
            # in different versions of numpy arrays sometimes are displayed as
            # 'array([ 0. ,' or 'array([0.0,', thus correct both whitespace
            # after parenthesis and before commas as well as .0 decimals
            got = re.sub("\\( ", '(', got)
            got = re.sub("\\[ ", '[', got)
            got = re.sub("0\\.0", '0.', got)
            got = re.sub("\s*,", ',', got)
            want = re.sub("\\( ", '(', want)
            want = re.sub("\\[ ", '[', want)
            want = re.sub("0\\.0", '0.', want)
            want = re.sub("\s*,", ',', want)
        super_check_output = _doctest_OutputChecker.check_output
        return super_check_output(self, want, got, optionflags)

# monkey-patching
doctest.OutputChecker = MadmomOutputChecker

# keep namespace clean
del pkg_resources, doctest
