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

# Create a doctest output checker that optionally ignores the unicode string
# literal.

# declare the new doctest directives
_IGNORE_UNICODE = doctest.register_optionflag("IGNORE_UNICODE")
doctest.IGNORE_UNICODE = _IGNORE_UNICODE
doctest.__all__.append("IGNORE_UNICODE")
doctest.COMPARISON_FLAGS = doctest.COMPARISON_FLAGS | _IGNORE_UNICODE

_NORMALIZE_ARRAYS = doctest.register_optionflag("NORMALIZE_ARRAYS")
doctest.NORMALIZE_ARRAYS = _NORMALIZE_ARRAYS
doctest.__all__.append("NORMALIZE_ARRAYS")
doctest.COMPARISON_FLAGS = doctest.COMPARISON_FLAGS | _NORMALIZE_ARRAYS

_doctest_OutputChecker = doctest.OutputChecker


class _OutputChecker(_doctest_OutputChecker):
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
        if optionflags & _NORMALIZE_ARRAYS:
            # in different versions of numpy arrays sometimes are displayed as
            # 'array([ 0. ,' or 'array([0.0,', thus correct both whitespace
            # after parenthesis and before commas as well as .0 decimals
            got = re.sub(r'\( ', '(', got)
            got = re.sub(r'\[ ', '[', got)
            got = re.sub(r'0\.0', '0.', got)
            got = re.sub(r'\s*,', ',', got)
            want = re.sub(r'\( ', '(', want)
            want = re.sub(r'\[ ', '[', want)
            want = re.sub(r'0\.0', '0.', want)
            want = re.sub(r'\s*,', ',', want)
        super_check_output = _doctest_OutputChecker.check_output
        return super_check_output(self, want, got, optionflags)


# monkey-patching
doctest.OutputChecker = _OutputChecker

# keep namespace clean
del pkg_resources, doctest
