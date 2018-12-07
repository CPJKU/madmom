# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains all cepstrogram related functionality.

"""

from __future__ import absolute_import, division, print_function

import inspect

import numpy as np
from scipy.fftpack import dct

from .filters import MelFilterbank
from .spectrogram import Spectrogram
from ..processors import Processor
from ..utils import lazyprop


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
MFCC_DCT_NORM = 'ortho'
MFCC_DELTA_FILTER = np.linspace(4, -4, 9) / 60
MFCC_DELTA_DELTA_FILTER = np.linspace(1, -1, 3) / 2


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
    dct_norm : {'ortho', None}, optional
        Normalization mode (see scipy.fftpack.dct). Default is 'ortho'.
    kwargs : dict, optional
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
                 dct_norm=MFCC_DCT_NORM, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, spectrogram, filterbank=MelFilterbank,
                num_bands=MFCC_BANDS, fmin=MFCC_FMIN, fmax=MFCC_FMAX,
                norm_filters=MFCC_NORM_FILTERS, mul=MFCC_MUL, add=MFCC_ADD,
                dct_norm=MFCC_DCT_NORM, **kwargs):
        # for signature documentation see __init__()
        from .filters import Filterbank
        # instantiate a Spectrogram if needed
        if not isinstance(spectrogram, Spectrogram):
            # try to instantiate a Spectrogram object
            spectrogram = Spectrogram(spectrogram, **kwargs)

        # instantiate a Filterbank if needed
        if inspect.isclass(filterbank) and issubclass(filterbank, Filterbank):
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
        np.log(mul * data + add, out=data)
        # apply type 2 DCT
        data = dct(data, norm=dct_norm)
        # cast as MFCC
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.spectrogram = spectrogram
        obj.filterbank = filterbank
        obj.mul = mul
        obj.add = add
        # return the object
        return obj

    @staticmethod
    def calc_deltas(data, delta_filter):
        """
        Apply the given filter to the data after automatically padding by
        replicating the first and last frame. The length of the padding is
        calculated via ceil(len(delta_filter)).

        Applying a filter means passing the matrix column after column to
        ``np.convolve()``. Afterwards the array is truncated to the same
        shape as the input array.

        Parameters
        ----------
        data: numpy array
            Data to process, i.e. MFCCs or deltas thereof.
        delta_filter: numpy array
            Filter used for convolution.

        Returns
        -------
        deltas: numpy array
             Deltas of `data`, same shape as `data`.

        """
        # pad data by replicating the first and the last frame
        k = int(np.ceil(len(delta_filter) / 2))
        padded = np.vstack((np.array([data[0], ] * k), data,
                            np.array([data[-1], ] * k)))
        # calculate the deltas for each coefficient
        deltas = []
        for band in padded.T:
            deltas.append(np.convolve(band, delta_filter, 'same'))
        # return deltas (first/last k frames truncated)
        return np.vstack(deltas).T[k:-k]

    @lazyprop
    def deltas(self, delta_filter=MFCC_DELTA_FILTER):
        """
        First order derivative of the MFCCs.

        Parameters
        ----------
        delta_filter: numpy array, optional
            Filter to calculate the derivative of the MFCCs.

        Returns
        -------
        deltas: numpy array
             Deltas of the MFCCs, same shape as MFCCs.

        Notes
        -----
        Accessing this property corresponds to the function call
        ``MFCC.calc_deltas(mfccs, delta_filter)``, with results being cached.

        """
        return MFCC.calc_deltas(self, delta_filter)

    @lazyprop
    def delta_deltas(self, delta_delta_filter=MFCC_DELTA_DELTA_FILTER):
        """
        Second order derivatives of the MFCCs.

        Parameters
        ----------
        delta_delta_filter: numpy array, optional
            Filter to calculate the derivative of the derivative.

        Returns
        -------
        deltas: numpy array
             Delta deltas of the MFCCs, same shape as MFCCs.

        Notes
        -----
        Accessing this property corresponds to the function call
        ``MFCC.calc_deltas(deltas, delta_delta_filter)``, with results being
        cached.

        """
        return MFCC.calc_deltas(self.deltas, delta_delta_filter)

    def calc_voicebox_deltas(self, delta_filter=MFCC_DELTA_FILTER,
                             delta_delta_filter=MFCC_DELTA_DELTA_FILTER):
        """
        Calculates deltas and delta deltas the way it is done in the voicebox
        MatLab toolbox [1]_.

        Parameters
        ----------
        delta_filter : numpy array
            Filter to calculate the derivative of the MFCCs.
        delta_delta_filter : numpy array
            Filter to calculate the derivative of the derivative.

        Returns
        -------
        [mfcc, delta, delta_delta] : numpy array, shape (num_frames, bands * 3)
            Horizontally stacked array consisting of the MFCC coefficients,
            their first and second order derivatives.

        References
        ----------
        .. [1] http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html

        """
        padded_input = np.vstack(
            (np.array([self[0], ] * 5), self, np.array([self[-1], ] * 5)))
        deltashape = tuple(reversed(padded_input.shape))
        flat_input = padded_input.transpose().flatten()
        deltas = np.convolve(flat_input, delta_filter, mode='same')
        deltas = deltas.reshape(deltashape).T[4:-4, ]
        deltadeltashape = tuple(reversed(deltas.shape))
        flat_deltas = deltas.transpose().flatten()
        deltas = deltas[1:-1, ]
        delta_deltas = np.convolve(flat_deltas, delta_delta_filter,
                                   mode='same')
        delta_deltas = delta_deltas.reshape(deltadeltashape).T[1:-1, ]
        return np.hstack((self, deltas, delta_deltas))

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
    MFCCProcessor filters the magnitude spectrogram with a Mel filterbank,
    takes the logarithm and performs a discrete cosine transform afterwards.

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
        kwargs : dict, optional
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
