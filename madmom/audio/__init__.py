# encoding: utf-8
"""
This package includes audio handling functionality and low-level features.
The definition of "low" may vary, but all "high"-level features (e.g. beats,
onsets, etc. -- basically everything you want to evaluate) should be in the
`madmom.features` package.

Notes
-----
Almost all functionality blocks are split into two classes:

1) A data class: instances are signal dependent, i.e. they operate directly on
   the signal and show different values for different signals.
2) A processor class: for every data class there should be a processor class
   with the exact same name and a "Processor" suffix. This class must inherit
   from madmom.Processor and define a process() method which returns a data
   class or inherit from madmom.SequentialProcessor or ParallelProcessor.

The data classes should be either sub-classed from numpy arrays or be indexable
and iterable. This way they can be used identically to numpy arrays.

"""

from __future__ import absolute_import, division, print_function

# import the submodules
from . import comb_filters, filters, signal, spectrogram, stft
# import classes used often
from .chroma import DeepChromaProcessor
from .signal import (FramedSignal, FramedSignalProcessor, Signal,
                     SignalProcessor, )
from .spectrogram import (FilteredSpectrogram, FilteredSpectrogramProcessor,
                          LogarithmicFilteredSpectrogram,
                          LogarithmicFilteredSpectrogramProcessor,
                          LogarithmicSpectrogram,
                          LogarithmicSpectrogramProcessor,
                          MultiBandSpectrogramProcessor, Spectrogram,
                          SpectrogramDifference,
                          SpectrogramDifferenceProcessor,
                          SpectrogramProcessor, )
from .stft import ShortTimeFourierTransform, ShortTimeFourierTransformProcessor
