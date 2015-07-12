#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all cepstrogram related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from scipy.fftpack import dct

from madmom.processors import Processor, SequentialProcessor
from madmom.audio.filters import MelFilterbank
from madmom.audio.spectrogram import (Spectrogram,
                                      LogarithmicFilteredSpectrogramProcessor)


class Cepstrogram(Processor):
    """
    Cepstrogram is a generic class which applies some transformation (usually
    a DCT) on a spectrogram.

    """

    def __init__(self, transform=dct, **kwargs):
        """
        Creates a new Cepstrogram instance for the given spectrogram.

        :param transform: transform

        """
        self.transform = transform

    def process(self, spectrogram):
        """
        :param spectrogram: Spectrogram instance or numpy array
        :return:            cepstrogram

        """
        if isinstance(spectrogram, Spectrogram):
            return self.transform(spectrogram.spec)
        else:
            return self.transform(spectrogram)


class MFCC(SequentialProcessor):
    """
    MFCC is a subclass of Cepstrogram which filters the magnitude spectrogram
    of the spectrogram with a Mel filterbank, takes the logarithm and performs
    a discrete cosine transform afterwards.

    """

    def __init__(self, num_bands=30, fmin=40, fmax=15000, norm_filters=True,
                 mul=1, add=0, transform=dct, **kwargs):
        """
        Creates a new MFCC processor.

        A Mel filterbank with these parameters:

        :param num_bands:    number of Mel filter bands
        :param fmin:         the minimum frequency [Hz]
        :param fmax:         the maximum frequency [Hz]
        :param norm_filters: normalize filter area to 1

        is used to filter a logarithmic spectrogram, which can be controlled by
        the following parameters:

        :param mul:          multiply the spectrogram with this factor before
                             taking the logarithm of the magnitudes [float]
        :param add:          add this value before taking the logarithm of
                             the magnitudes [float]

        Finally the chosen transform is applied.

        :param transform:    transformation to be applied on the spectrogram

        """
        # from https://en.wikipedia.org/wiki/Mel-frequency_cepstrum:
        #
        # MFCCs are commonly derived as follows:
        #
        # 1) Take the Fourier transform of (a windowed excerpt of) a signal.
        # 2) Map the powers of the spectrum obtained above onto the mel scale,
        #    using triangular overlapping windows.
        # 3) Take the logs of the powers at each of the mel frequencies.
        # 4) Take the discrete cosine transform of the list of mel log powers,
        #    as if it were a signal.
        # 5) The MFCCs are the amplitudes of the resulting spectrum
        # Note: 1) to 4) is handled by LogarithmicFilteredSpectrogramProcessor
        spec = LogarithmicFilteredSpectrogramProcessor(
            filterbank=MelFilterbank, num_bands=num_bands, fmin=fmin,
            fmax=fmax, norm_filters=norm_filters, mul=mul, add=add, **kwargs)
        # make it a SequentialProcessor([1-4, 5])
        super(MFCC, self).__init__([spec, Cepstrogram(transform=transform)])
