#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all MFCC related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

# TODO: move this to cp.audio.spectrogram? It's closely related

import scipy.fftpack as fft
from .spectrogram import Spectrogram

# from https://en.wikipedia.org/wiki/Mel-frequency_cepstrum:
#
# MFCCs are commonly derived as follows:
#
# • Take the Fourier transform of (a windowed excerpt of) a signal.
# • Map the powers of the spectrum obtained above onto the mel scale, using triangular overlapping windows.
# • Take the logs of the powers at each of the mel frequencies.
# • Take the discrete cosine transform of the list of mel log powers, as if it were a signal.
# • The MFCCs are the amplitudes of the resulting spectrum


# TODO: set other defaults than those in cp.audio.filterbank for MFCCs?
class MFCC(Spectrogram):
    """
    MFCC is a subclass of Spectrogram which filters the magnitude spectrogram
    with a Mel filterbank, takes the logarithm and performs a discrete cosine
    transform afterwards.

    """
    def __init__(self, *args, **kwargs):
        """
        Creates a new FilteredSpectrogram object instance.

        :param filterbank: filterbank for dimensionality reduction

        If no filterbank is given, one with the following parameters is created
        automatically.

        :param mel_bands:   number of filter bands per octave [default=40]
        :param fmin:        the minimum frequency [Hz, default=30]
        :param fmax:        the maximum frequency [Hz, default=17000]
        :param norm_filter: normalize the area of the filter to 1 [default=True]

        """
        from .filterbank import MelFilter, MEL_BANDS, FMIN, FMAX, NORM_FILTER
        from .spectrogram import MUL, ADD

        # fetch the arguments special to the filterbank creation (or set defaults)
        filterbank = kwargs.pop('filterbank', None)
        mel_bands = kwargs.pop('mel_bands', MEL_BANDS)
        fmin = kwargs.pop('fmin', FMIN)
        fmax = kwargs.pop('fmax', FMAX)
        norm_filter = kwargs.pop('norm_filter', NORM_FILTER)

        # fetch the arguments special to the logarithmic magnitude (or set defaults)
        mul = kwargs.pop('mul', MUL)
        add = kwargs.pop('add', ADD)

        # create Spectrogram object
        super(MFCC, self).__init__(*args, **kwargs)
        # if no filterbank was given, create one
        if filterbank is None:
            filterbank = MelFilter(fft_bins=self.fft_bins, sample_rate=self.audio.sample_rate, mel_bands=mel_bands, fmin=fmin, fmax=fmax, norm=norm_filter)

        # set the parameters, so they get used when the magnitude spectrogram gets computed
        self.filterbank = filterbank
        self.log = True
        self.mul = mul
        self.add = add

        # MFCC matrix
        self.__mfcc = None

    @property
    def mfcc(self):
        if self.__mfcc is None:
            # take the DCT of the LogMelSpec
            # FIXME: include all bins or skip the first?
            self.__mfcc = fft.dct(self.spec)
        return self.__mfcc
