# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains all cepstrogram related functionality.

"""

from __future__ import absolute_import, division, print_function

import warnings
import numpy as np
from scipy.fftpack import dct

from ..processors import Processor
from .filters import MelFilterbank
from .spectrogram import Spectrogram


class Cepstrogram(np.ndarray):
    """
    The Cepstrogram class represents a transformed Spectrogram. This generic
    class applies some transformation (usually a DCT) on a spectrogram.

    Parameters
    ----------
    spectrogram : :class:`.audio.spectrogram.Spectrogram` instance
        Spectrogram.
    transform : numpy ufunc
        Transformation applied to the `spectrogram`.
    kwargs : dict
        If no :class:`.audio.spectrogram.Spectrogram` instance was given,
        one is instantiated with these additional keyword arguments.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, spectrogram, transform=dct, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, spectrogram, transform=dct, **kwargs):
        # instantiate a Spectrogram if needed
        if not isinstance(spectrogram, Spectrogram):
            # try to instantiate a Spectrogram object
            spectrogram = Spectrogram(spectrogram, **kwargs)

        # apply the transformation to the spectrogram
        data = transform(spectrogram)
        # cast as Cepstrogram
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.spectrogram = spectrogram
        obj.transform = transform
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.spectrogram = getattr(obj, 'spectrogram', None)
        self.transform = getattr(obj, 'transform', None)

    @property
    def num_frames(self):
        """Number of frames."""
        return len(self)

    @property
    def num_bins(self):
        """Number of bins."""
        return int(self.shape[1])


class CepstrogramProcessor(Processor):
    """
    Cepstrogram processor class.

    Parameters
    ----------
    transform : numpy ufunc
        Transformation applied during processing.

    """

    def __init__(self, transform=dct, **kwargs):
        # pylint: disable=unused-argument
        self.transform = transform

    def process(self, data, **kwargs):
        """
        Return a Cepstrogram of the given data.

        Parameters
        ----------
        data : numpy array
            Data to be processed (usually a spectrogram).
        kwargs : dict
            Keyword arguments passed to :class:`Cepstrogram`.

        Returns
        -------
        :class:`Cepstrogram` instance
            Cepstrogram.

        """
        # update arguments passed to Cepstrogram
        args = dict(transform=self.transform)
        args.update(kwargs)
        # instantiate and return Cepstrogram
        return Cepstrogram(data, **args)


MFCC_BANDS = 30
MFCC_FMIN = 40.
MFCC_FMAX = 15000.
MFCC_NORM_FILTERS = True
MFCC_MUL = 1.
MFCC_ADD = np.spacing(1)


class MFCC(Cepstrogram):
    """
    MFCC class.

    Parameters
    ----------
    spectrogram : :class:`.audio.spectrogram.Spectrogram` instance
        Spectrogram.
    filterbank : :class:`.audio.filters.Filterbank` type or instance, optional
        Filterbank used to filter the `spectrogram`; if a
        :class:`.audio.filters.Filterbank` type (i.e. class) is given
        (rather than an instance), one will be created with the given type
        and following parameters:
    num_bands : int, optional
        Number of filter bands (per octave, depending on the type of the
        filterbank).
    fmin : float, optional
        The minimum frequency of the filterbank [Hz].
    fmax : float, optional
        The maximum frequency of the filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filters to area 1.
    mul : float, optional
        Multiply the magnitude spectrogram with this factor before taking the
        logarithm.
    add : float, optional
        Add this value before taking the logarithm of the magnitudes.
    kwargs : dict
        If no :class:`.audio.spectrogram.Spectrogram` instance was given, one
        is instantiated and these keyword arguments are passed.

    Notes
    -----

    If a filtered or scaled Spectrogram is given, a new unfiltered and unscaled
    Spectrogram will be computed and then the given filter and scaling will be
    applied accordingly.

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
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, spectrogram, filterbank=MelFilterbank,
                 num_bands=MFCC_BANDS, fmin=MFCC_FMIN, fmax=MFCC_FMAX,
                 norm_filters=MFCC_NORM_FILTERS, mul=MFCC_MUL, add=MFCC_ADD,
                 **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, spectrogram, filterbank=MelFilterbank,
                num_bands=MFCC_BANDS, fmin=MFCC_FMIN, fmax=MFCC_FMAX,
                norm_filters=MFCC_NORM_FILTERS, mul=MFCC_MUL, add=MFCC_ADD,
                **kwargs):
        # for signature documentation see __init__()
        from .filters import Filterbank
        # instantiate a Spectrogram if needed
        if not isinstance(spectrogram, Spectrogram):
            # try to instantiate a Spectrogram object
            spectrogram = Spectrogram(spectrogram, **kwargs)

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
        # apply DCT
        data = dct(data)
        # cast as MFCC
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.spectrogram = spectrogram
        obj.filterbank = filterbank
        obj.mul = mul
        obj.add = add
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.filterbank = getattr(obj, 'filterbank', None)
        self.mul = getattr(obj, 'mul', MFCC_MUL)
        self.add = getattr(obj, 'add', MFCC_ADD)
        super(MFCC, self).__array_finalize__(obj)


class MFCCProcessor(Processor):
    """
    MFCCProcessor is CepstrogramProcessor which filters the magnitude
    spectrogram of the spectrogram with a Mel filterbank, takes the logarithm
    and performs a discrete cosine transform afterwards.

    Parameters
    ----------
    num_bands : int, optional
        Number of Mel filter bands.
    fmin : float, optional
        Minimum frequency of the Mel filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the Mel filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filters to area 1.
    mul : float, optional
        Multiply the magnitude spectrogram with this factor before taking the
        logarithm.
    add : float, optional
        Add this value before taking the logarithm of the magnitudes.
    transform : numpy ufunc
        Transformation applied to the Mel filtered spectrogram.

    """

    def __init__(self, num_bands=MFCC_BANDS, fmin=MFCC_FMIN, fmax=MFCC_FMAX,
                 norm_filters=MFCC_NORM_FILTERS, mul=MFCC_MUL, add=MFCC_ADD,
                 **kwargs):
        # pylint: disable=unused-argument
        self.num_bands = num_bands
        self.fmin = fmin
        self.fmax = fmax
        self.norm_filters = norm_filters
        self.mul = mul
        self.add = add
        # TODO: add filterbank argument to the processor?
        self.filterbank = None  # needed for caching

    def process(self, data, **kwargs):
        """
        Process the data and return the MFCCs of it.

        Parameters
        ----------
        data : numpy array
            Data to be processed (a spectrogram).
        kwargs : dict
            Keyword arguments passed to :class:`MFCC`.

        Returns
        -------
        :class:`MFCC` instance
            MFCCs of the data.

        """
        # update arguments passed to MFCCs
        # TODO: if these arguments change, the filterbank needs to be discarded
        args = dict(num_bands=self.num_bands, fmin=self.fmin, fmax=self.fmax,
                    norm_filters=self.norm_filters, mul=self.mul, add=self.add,
                    filterbank=self.filterbank)
        args.update(kwargs)
        # instantiate MFCCs
        data = MFCC(data, **args)
        # cache the filterbank
        self.filterbank = data.filterbank
        # return MFCCs
        return data
