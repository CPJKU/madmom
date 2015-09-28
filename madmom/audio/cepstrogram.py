#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all cepstrogram related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np
from scipy.fftpack import dct

from ..processors import Processor
from .stft import PropertyMixin
from .filters import MelFilterbank
from .spectrogram import Spectrogram


class Cepstrogram(PropertyMixin, np.ndarray):
    """
    Cepstrogram is a generic class which applies some transformation (usually
    a DCT) on a spectrogram.

    """

    def __new__(cls, spectrogram, transform=dct, **kwargs):
        """
        Creates a new Cepstrogram instance from the given Spectrogram.

        :param spectrogram:       Spectrogram instance (or anything a
                                  Spectrogram can be instantiated from)

        If no Spectrogram instance was given, one is instantiated and
        these arguments are passed:

        :param kwargs:            keyword arguments passed to Spectrogram

        """
        # instantiate a Spectrogram if needed
        if not isinstance(spectrogram, Spectrogram):
            # try to instantiate a Spectrogram object
            spectrogram = Spectrogram(spectrogram, **kwargs)

        # apply the transformation to the spectrogram
        data = transform(spectrogram)
        # cast as Cepstrogram
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.transform = transform
        obj.spectrogram = spectrogram
        # and those from the given spectrogram
        obj.stft = spectrogram.stft
        obj.frames = spectrogram.stft.frames
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.transform = getattr(obj, 'transform', None)
        self.spectrogram = getattr(obj, 'spectrogram', None)

    def __reduce__(self):
        # get the parent's __reduce__ tuple
        pickled_state = super(Cepstrogram, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.transform,)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # set the attributes
        self.transform = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(Cepstrogram, self).__setstate__(state[0:-1])

    @property
    def bin_frequencies(self):
        """Frequencies of the bins."""
        # TODO: what are the frequencies of the bins?
        raise NotImplementedError('please implement!')


class CepstrogramProcessor(Processor):
    """
    Cepstrogram processor class.

    """

    def __init__(self, transform=dct, **kwargs):
        """
        Creates a new CepstrogramProcessor.

        :param transform: transform

        """
        self.transform = transform

    def process(self, data):
        """
        :param data: data to be processed
        :return:     Cepstrogram instance

        """
        return Cepstrogram(data)


MFCC_BANDS = 30
MFCC_FMIN = 40.
MFCC_FMAX = 15000.
MFCC_NORM_FILTERS = True
MFCC_MUL = 1
MFCC_ADD = 0


class MFCC(Cepstrogram):
    """
    MFCC class.

    From https://en.wikipedia.org/wiki/Mel-frequency_cepstrum:

    MFCCs are commonly derived as follows:

    1) Take the Fourier transform of (a windowed excerpt of) a signal.
    2) Map the powers of the spectrum obtained above onto the mel scale,
       using triangular overlapping windows.
    3) Take the logs of the powers at each of the mel frequencies.
    4) Take the discrete cosine transform of the list of mel log powers,
       as if it were a signal.
    5) The MFCCs are the amplitudes of the resulting spectrum

    """

    def __new__(cls, spectrogram, transform=dct, filterbank=MelFilterbank,
                num_bands=MFCC_BANDS, fmin=MFCC_FMIN, fmax=MFCC_FMAX,
                norm_filters=MFCC_NORM_FILTERS, mul=MFCC_MUL, add=MFCC_ADD,
                **kwargs):
        """
        Creates a new MFCC instance from the given Spectrogram.

        :param spectrogram:       Spectrogram instance (or anything a
                                  Spectrogram can be instantiated from)
        :param transform:         transformation to be applied to the
                                  spectrogram to obtain a cepstrogram

        Filterbank parameters:

        :param filterbank:        Filterbank type or instance [Filterbank]

        If a Filterbank type is given rather than a Filterbank instance, one
        will be created with the given type and these parameters:

        :param num_bands:         number of filter bands (per octave, depending
                                  on the type of the filterbank) [int]
        :param fmin:              the minimum frequency [Hz, float]
        :param fmax:              the maximum frequency [Hz, float]

        Logarithmic magnitude parameters:

        :param mul:               multiply the magnitude spectrogram with this
                                  factor before taking the logarithm [float]
        :param add:               add this value before taking the logarithm
                                  of the magnitudes [float]

        If no Spectrogram instance was given, one is instantiated and
        these arguments are passed:

        :param kwargs:            keyword arguments passed to Spectrogram

        """
        from .filters import Filterbank
        # instantiate a Spectrogram if needed
        if not isinstance(spectrogram, Spectrogram):
            # try to instantiate a Spectrogram object
            spectrogram = Spectrogram(spectrogram, **kwargs)

        # recalculate the spec if it is filtered or scaled already
        if (spectrogram.filterbank is not None or
                spectrogram.mul is not None or
                spectrogram.add is not None):
            import warnings
            warnings.warn('Spectrogram was filtered or scaled already, redo '
                          'calculation!')
            spectrogram = Spectrogram(spectrogram.stft)

        # instantiate a Filterbank if needed
        if issubclass(filterbank, Filterbank):
            # create a filterbank of the given type
            filterbank = filterbank(spectrogram.bin_frequencies,
                                    num_bands=num_bands, fmin=fmin, fmax=fmax,
                                    norm_filters=norm_filters,
                                    duplicate_filters=False)
        if not isinstance(filterbank, Filterbank):
            raise ValueError('not a Filterbank type or instance: %s' %
                             filterbank)
        # filter the spectrogram
        data = np.dot(spectrogram, filterbank)
        # logarithmically scale the magnitudes
        np.log10(mul * data + add, out=data)
        # apply the transformation
        data = transform(data)
        # cast as MFCC
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.transform = transform
        obj.spectrogram = spectrogram
        obj.filterbank = filterbank
        obj.mul = mul
        obj.add = add
        # and those from the given spectrogram
        obj.stft = spectrogram.stft
        obj.frames = spectrogram.stft.frames
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.transform = getattr(obj, 'transform', None)
        self.spectrogram = getattr(obj, 'spectrogram', None)
        self.filterbank = getattr(obj, 'filterbank', None)
        super(MFCC, self).__array_finalize__(obj)

    def __reduce__(self):
        # get the parent's __reduce__ tuple
        pickled_state = super(MFCC, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.transform, self.filterbank,
                                        self.mul, self.add)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # set the attributes
        self.transform = state[-4]
        self.filterbank = state[-3]
        self.mul = state[-2]
        self.add = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(MFCC, self).__setstate__(state[0:-4])


class MFCCProcessor(Processor):
    """
    MFCC is a subclass of Cepstrogram which filters the magnitude spectrogram
    of the spectrogram with a Mel filterbank, takes the logarithm and performs
    a discrete cosine transform afterwards.

    """

    def __init__(self, num_bands=MFCC_BANDS, fmin=MFCC_FMIN, fmax=MFCC_FMAX,
                 norm_filters=MFCC_NORM_FILTERS, mul=MFCC_MUL, add=MFCC_ADD,
                 transform=dct, **kwargs):
        """
        Creates a new MFCC processor.

        A Mel filterbank with these parameters:

        :param num_bands:    number of Mel filter bands
        :param fmin:         the minimum frequency [Hz]
        :param fmax:         the maximum frequency [Hz]
        :param norm_filters: normalize filter area to 1

        is used to filter the spectrogram, and then scaled logarithmically
        with these parameters:

        :param mul:          multiply the spectrogram with this factor before
                             taking the logarithm of the magnitudes [float]
        :param add:          add this value before taking the logarithm of
                             the magnitudes [float]

        Finally the chosen transform is applied.

        :param transform:    transformation to be applied on the spectrogram

        """
        self.num_bands = num_bands
        self.fmin = fmin
        self.fmax = fmax
        self.norm_filters = norm_filters
        self.mul = mul
        self.add = add
        self.transform = transform

    def process(self, data):
        """
        Process the data and return the MFCCs of it.

        :param data: data to be processed
        :return:     MFCCs of the data

        """
        return MFCC(data, num_bands=self.num_bands, fmin=self.fmin,
                    fmax=self.fmax, norm_filters=self.norm_filters,
                    mul=self.mul, add=self.add)
