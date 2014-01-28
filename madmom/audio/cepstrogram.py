#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all Cepstrogram related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import scipy.fftpack as fft


class Cepstrogram(object):

    """
    Cepstrogram is a generic class which applies some transformation (usually
    a DFT or DCT) on a spectrogram.

    """
    def __init__(self, spectrogram, *args, **kwargs):
        """
        Creates a new Cepstrogram object instance for the given spectrogram

        :param spectrogram: spectrogram to operate on

        """
        from .spectrogram import Spectrogram
        # audio signal stuff
        if isinstance(spectrogram, Cepstrogram):
            # already a Cepstrogram object, copy the attributes (which can be
            # overwritten by passing other values to the constructor)
            self._spectrogram = spectrogram.spectrogram
        else:
            # try to instantiate a Spectrogram object
            self._spectrogram = Spectrogram(spectrogram, *args, **kwargs)

    @property
    def spectrogram(self):
        """Spectrogram."""
        return self._spectrogram

    # alias
    s = spectrogram


# TODO: set other defaults than those in cp.audio.filterbank for MFCCs?
class MFCC(Cepstrogram):
    """
    MFCC is a subclass of Cepstrogram which filters the magnitude spectrogram
    of the spectrogram with a Mel filterbank, takes the logarithm and performs
    a discrete cosine transform afterwards.

    """
    def __init__(self, *args, **kwargs):
        """
        Creates a new MFCC object instance.

        :param filterbank: filterbank for dimensionality reduction

        If no filterbank is given, one with the following parameters is created
        automatically.

        :param mel_bands:    number of filter bands per octave
        :param fmin:         the minimum frequency [Hz]
        :param fmax:         the maximum frequency [Hz]
        :param norm_filters: normalize filter area to 1

        """
        # from https://en.wikipedia.org/wiki/Mel-frequency_cepstrum:
        #
        # MFCCs are commonly derived as follows:
        #
        # • Take the Fourier transform of (a windowed excerpt of) a signal.
        # • Map the powers of the spectrum obtained above onto the mel scale,
        #   using triangular overlapping windows.
        # • Take the logs of the powers at each of the mel frequencies.
        # • Take the discrete cosine transform of the list of mel log powers,
        #   as if it were a signal.
        # • The MFCCs are the amplitudes of the resulting spectrum
        from .filterbank import (MelFilterBank, MEL_BANDS, FMIN, FMAX,
                                 NORM_FILTERS)
        from .spectrogram import MUL, ADD

        # fetch the arguments for filterbank creation (or set defaults)
        fb = kwargs.pop('filterbank', None)
        mel_bands = kwargs.pop('mel_bands', MEL_BANDS)
        fmin = kwargs.pop('fmin', FMIN)
        fmax = kwargs.pop('fmax', FMAX)
        norm_filters = kwargs.pop('norm_filters', NORM_FILTERS)

        # fetch the arguments for the logarithmic magnitude (or set defaults)
        mul = kwargs.pop('mul', MUL)
        add = kwargs.pop('add', ADD)

        # create Cepstrogram object
        super(MFCC, self).__init__(*args, **kwargs)
        # if no filterbank was given, create one
        if fb is None:
            sample_rate = self.spectrogram.frames.signal.sample_rate
            fb = MelFilterBank(fft_bins=self.spectrogram.num_fft_bins,
                               sample_rate=sample_rate,
                               bands=mel_bands, fmin=fmin, fmax=fmax,
                               norm=norm_filters)

        # set the parameters, so they get used for computation
        self._spectrogram._filterbank = fb
        self._spectrogram._log = True
        self._spectrogram._mul = mul
        self._spectrogram._add = add

        # MFCC matrix
        self._mfcc = None

    @property
    def mfcc(self):
        """Mel-frequency cepstral coefficients."""
        if self._mfcc is None:
            # take the DCT of the LogMelSpec (including the first bin)
            self._mfcc = fft.dct(self._spectrogram.spec)
        return self._mfcc
