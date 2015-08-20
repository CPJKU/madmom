# encoding: utf-8
"""
This package includes audio handling functionality and lower level features.
The definition of "lower" may vary, but all "higher" level features
(e.g. beats, onsets, etc.) can be found in the `features` package.

"""

# import the submodules
from . import signal, ffmpeg, filters, spectrogram
