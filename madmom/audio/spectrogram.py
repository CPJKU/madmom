# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains spectrogram related functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from ..processors import Processor, SequentialProcessor
from .filters import (LogarithmicFilterbank, NUM_BANDS, FMIN, FMAX, A4,
                      NORM_FILTERS, UNIQUE_FILTERS)


def spec(stft):
    """
    Computes the magnitudes of the complex Short Time Fourier Transform of a
    signal.

    Parameters
    ----------
    stft : numpy array
        Complex STFT of a signal.

    Returns
    -------
    spec : numpy array
        Magnitude spectrogram.

    """
    return np.abs(stft)


# magnitude spectrogram of STFT
class Spectrogram(np.ndarray):
    """
    A :class:`Spectrogram` represents the magnitude spectrogram of a
    :class:`.audio.stft.ShortTimeFourierTransform`.

    Parameters
    ----------
    stft : :class:`.audio.stft.ShortTimeFourierTransform` instance
        Short Time Fourier Transform.
    kwargs : dict, optional
        If no :class:`.audio.stft.ShortTimeFourierTransform` instance was
        given, one is instantiated with these additional keyword arguments.

    Examples
    --------
    Create a :class:`Spectrogram` from a
    :class:`.audio.stft.ShortTimeFourierTransform` (or anything it can be
    instantiated from:

    >>> spec = Spectrogram('tests/data/audio/sample.wav')
    >>> spec  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Spectrogram([[ 3.15249,  4.00272, ...,  0.03634,  0.03671],
                 [ 4.28429,  2.85158, ...,  0.0219 ,  0.02227],
                 ...,
                 [ 4.92274, 10.27775, ...,  0.00607,  0.00593],
                 [ 9.22709,  9.6387 , ...,  0.00981,  0.00984]], dtype=float32)

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, stft, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, stft, **kwargs):
        from .stft import ShortTimeFourierTransform
        # check stft type
        if isinstance(stft, Spectrogram):
            # already a Spectrogram
            data = stft
        elif isinstance(stft, ShortTimeFourierTransform):
            # take the abs of the STFT
            data = np.abs(stft)
        else:
            # try to instantiate a ShortTimeFourierTransform
            stft = ShortTimeFourierTransform(stft, **kwargs)
            # take the abs of the STFT
            data = np.abs(stft)
        # cast as Spectrogram
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.stft = stft
        obj.bin_frequencies = stft.bin_frequencies
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.stft = getattr(obj, 'stft', None)
        self.bin_frequencies = getattr(obj, 'bin_frequencies', None)
        # Note: these attributes are added for compatibility, if they are
        #       present any spectrogram sub-class behaves exactly the same
        self.filterbank = getattr(obj, 'filterbank', None)
        self.mul = getattr(obj, 'mul', None)
        self.add = getattr(obj, 'add', None)

    @property
    def num_frames(self):
        """Number of frames."""
        return len(self)

    @property
    def num_bins(self):
        """Number of bins."""
        return int(self.shape[1])

    def diff(self, **kwargs):
        """
        Return the difference of the magnitude spectrogram.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to :class:`SpectrogramDifference`.

        Returns
        -------
        diff : :class:`SpectrogramDifference` instance
            The differences of the magnitude spectrogram.

        """
        return SpectrogramDifference(self, **kwargs)

    def filter(self, **kwargs):
        """
        Return a filtered version of the magnitude spectrogram.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to :class:`FilteredSpectrogram`.

        Returns
        -------
        filt_spec : :class:`FilteredSpectrogram` instance
            Filtered version of the magnitude spectrogram.

        """
        return FilteredSpectrogram(self, **kwargs)

    def log(self, **kwargs):
        """
        Return a logarithmically scaled version of the magnitude spectrogram.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to :class:`LogarithmicSpectrogram`.

        Returns
        -------
        log_spec : :class:`LogarithmicSpectrogram` instance
            Logarithmically scaled version of the magnitude spectrogram.

        """
        return LogarithmicSpectrogram(self, **kwargs)


class SpectrogramProcessor(Processor):
    """
    SpectrogramProcessor class.

    """
    def __init__(self, **kwargs):
        pass

    def process(self, data, **kwargs):
        """
        Create a Spectrogram from the given data.

        Parameters
        ----------
        data : numpy array
            Data to be processed.
        kwargs : dict
            Keyword arguments passed to :class:`Spectrogram`.

        Returns
        -------
        spec : :class:`Spectrogram` instance
            Spectrogram.

        """
        return Spectrogram(data, **kwargs)


# filtered spectrogram stuff
FILTERBANK = LogarithmicFilterbank


class FilteredSpectrogram(Spectrogram):
    """
    FilteredSpectrogram class.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        Spectrogram.
    filterbank : :class:`.audio.filters.Filterbank`, optional
        Filterbank class or instance; if a class is given (rather than an
        instance), one will be created with the given type and parameters.
    num_bands : int, optional
        Number of filter bands (per octave, depending on the type of the
        `filterbank`).
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    fref : float, optional
        Tuning frequency of the filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filter bands of the filterbank to area 1.
    unique_filters : bool, optional
        Indicate if the filterbank should contain only unique filters, i.e.
        remove duplicate filters resulting from insufficient resolution at
        low frequencies.
    kwargs : dict, optional
        If no :class:`Spectrogram` instance was given, one is instantiated
        with these additional keyword arguments.

    Examples
    --------
    Create a :class:`FilteredSpectrogram` from a :class:`Spectrogram` (or
    anything it can be instantiated from. Per default a
    :class:`.madmom.audio.filters.LogarithmicFilterbank` with 12 bands per
    octave is used.

    >>> spec = FilteredSpectrogram('tests/data/audio/sample.wav')
    >>> spec  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    FilteredSpectrogram([[  5.66156, 6.30141, ..., 0.05426, 0.06461],
                         [  8.44266, 8.69582, ..., 0.07703, 0.0902 ],
                         ...,
                         [ 10.04626, 1.12018, ..., 0.0487 , 0.04282],
                         [  8.60186, 6.81195, ..., 0.03721, 0.03371]],
                        dtype=float32)

    The resulting spectrogram has fewer frequency bins, with the centers of
    the bins aligned logarithmically (lower frequency bins still have a linear
    spacing due to the coarse resolution of the DFT at low frequencies):

    >>> spec.shape
    (281, 81)
    >>> spec.num_bins
    81
    >>> spec.bin_frequencies  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([    43.06641,    64.59961,    86.13281,   107.66602,
              129.19922,   150.73242,   172.26562,   193.79883, ...,
            10551.26953, 11175.73242, 11843.26172, 12553.85742,
            13285.98633, 14082.71484, 14922.50977, 15805.37109])

    The filterbank used to filter the spectrogram is saved as an attribute:

    >>> spec.filterbank  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    LogarithmicFilterbank([[ 0., 0., ..., 0., 0.],
                           [ 0., 0., ..., 0., 0.],
                           ...,
                           [ 0., 0., ..., 0., 0.],
                           [ 0., 0., ..., 0., 0.]], dtype=float32)
    >>> spec.filterbank.num_bands
    81

    The filterbank can be chosen at instantiation time:

    >>> from madmom.audio.filters import MelFilterbank
    >>> spec = FilteredSpectrogram('tests/data/audio/sample.wav', \
    filterbank=MelFilterbank, num_bands=40)
    >>> type(spec.filterbank)
    <class 'madmom.audio.filters.MelFilterbank'>
    >>> spec.shape
    (281, 40)

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, spectrogram, filterbank=FILTERBANK, num_bands=NUM_BANDS,
                 fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, spectrogram, filterbank=FILTERBANK, num_bands=NUM_BANDS,
                fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                unique_filters=UNIQUE_FILTERS, **kwargs):
        # pylint: disable=unused-argument
        import inspect
        from .filters import Filterbank
        # instantiate a Spectrogram if needed
        if not isinstance(spectrogram, Spectrogram):
            # try to instantiate a Spectrogram object
            spectrogram = Spectrogram(spectrogram, **kwargs)
        # instantiate a Filterbank if needed
        if inspect.isclass(filterbank) and issubclass(filterbank, Filterbank):
            # a Filterbank class is given, create a filterbank of this type
            filterbank = filterbank(spectrogram.bin_frequencies,
                                    num_bands=num_bands, fmin=fmin, fmax=fmax,
                                    fref=fref, norm_filters=norm_filters,
                                    unique_filters=unique_filters)
        if not isinstance(filterbank, Filterbank):
            raise TypeError('not a Filterbank type or instance: %s' %
                            filterbank)
        # filter the spectrogram
        data = np.dot(spectrogram, filterbank)
        # cast as FilteredSpectrogram
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.filterbank = filterbank
        # use the center frequencies of the filterbank as bin_frequencies
        obj.bin_frequencies = filterbank.center_frequencies
        # and those from the given spectrogram
        obj.stft = spectrogram.stft
        obj.mul = spectrogram.mul
        obj.add = spectrogram.add
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.filterbank = getattr(obj, 'filterbank', None)
        super(FilteredSpectrogram, self).__array_finalize__(obj)


class FilteredSpectrogramProcessor(Processor):
    """
    FilteredSpectrogramProcessor class.

    Parameters
    ----------
    filterbank : :class:`.audio.filters.Filterbank`
        Filterbank used to filter a spectrogram.
    num_bands : int
        Number of bands (per octave).
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    fref : float, optional
        Tuning frequency of the filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filter of the filterbank to area 1.
    unique_filters : bool, optional
        Indicate if the filterbank should contain only unique filters, i.e.
        remove duplicate filters resulting from insufficient resolution at
        low frequencies.

    """

    def __init__(self, filterbank=FILTERBANK, num_bands=NUM_BANDS, fmin=FMIN,
                 fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS, **kwargs):
        # pylint: disable=unused-argument

        self.filterbank = filterbank
        self.num_bands = num_bands
        self.fmin = fmin
        self.fmax = fmax
        self.fref = fref
        self.norm_filters = norm_filters
        self.unique_filters = unique_filters

    def process(self, data, **kwargs):
        """
        Create a FilteredSpectrogram from the given data.

        Parameters
        ----------
        data : numpy array
            Data to be processed.
        kwargs : dict
            Keyword arguments passed to :class:`FilteredSpectrogram`.

        Returns
        -------
        filt_spec : :class:`FilteredSpectrogram` instance
            Filtered spectrogram.

        """
        # instantiate a FilteredSpectrogram and return it
        return FilteredSpectrogram(data, filterbank=self.filterbank,
                                   num_bands=self.num_bands, fmin=self.fmin,
                                   fmax=self.fmax, fref=self.fref,
                                   norm_filters=self.norm_filters,
                                   unique_filters=self.unique_filters,
                                   **kwargs)


# logarithmic spectrogram stuff
LOG = np.log10
MUL = 1.
ADD = 1.


class LogarithmicSpectrogram(Spectrogram):
    """
    LogarithmicSpectrogram class.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        Spectrogram.
    log : numpy ufunc, optional
        Logarithmic scaling function to apply.
    mul : float, optional
        Multiply the magnitude spectrogram with this factor before taking
        the logarithm.
    add : float, optional
        Add this value before taking the logarithm of the magnitudes.
    kwargs : dict, optional
        If no :class:`Spectrogram` instance was given, one is instantiated
        with these additional keyword arguments.

    Examples
    --------
    Create a :class:`LogarithmicSpectrogram` from a :class:`Spectrogram` (or
    anything it can be instantiated from. Per default `np.log10` is used as
    the scaling function and a value of 1 is added to avoid negative values.

    >>> spec = LogarithmicSpectrogram('tests/data/audio/sample.wav')
    >>> spec  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    LogarithmicSpectrogram([[...]], dtype=float32)
    >>> spec.min()
    LogarithmicSpectrogram(1.604927092557773e-06, dtype=float32)

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, spectrogram, log=LOG, mul=MUL, add=ADD, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, spectrogram, log=LOG, mul=MUL, add=ADD, **kwargs):
        # instantiate a Spectrogram if needed
        if not isinstance(spectrogram, Spectrogram):
            # try to instantiate a Spectrogram object
            spectrogram = Spectrogram(spectrogram, **kwargs)
            data = spectrogram
        else:
            # make a copy of the spectrogram
            data = spectrogram.copy()
        # scale the spectrogram
        if mul is not None:
            data *= mul
        if add is not None:
            data += add
        if log is not None:
            log(data, data)
        # cast as FilteredSpectrogram
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.mul = mul
        obj.add = add
        # and those from the given spectrogram
        obj.stft = spectrogram.stft
        obj.bin_frequencies = spectrogram.bin_frequencies
        obj.filterbank = spectrogram.filterbank
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.mul = getattr(obj, 'mul', MUL)
        self.add = getattr(obj, 'add', ADD)
        super(LogarithmicSpectrogram, self).__array_finalize__(obj)


class LogarithmicSpectrogramProcessor(Processor):
    """
    Logarithmic Spectrogram Processor class.

    Parameters
    ----------
    log : numpy ufunc, optional
        Loagrithmic scaling function to apply.
    mul : float, optional
        Multiply the magnitude spectrogram with this factor before taking the
        logarithm.
    add : float, optional
        Add this value before taking the logarithm of the magnitudes.

    """

    def __init__(self, log=LOG, mul=MUL, add=ADD, **kwargs):
        # pylint: disable=unused-argument
        self.log = log
        self.mul = mul
        self.add = add

    def process(self, data, **kwargs):
        """
        Perform logarithmic scaling of a spectrogram.

        Parameters
        ----------
        data : numpy array
            Data to be processed.
        kwargs : dict
            Keyword arguments passed to :class:`LogarithmicSpectrogram`.

        Returns
        -------
        log_spec : :class:`LogarithmicSpectrogram` instance
            Logarithmically scaled spectrogram.

        """
        # instantiate a LogarithmicSpectrogram
        return LogarithmicSpectrogram(data, log=self.log, mul=self.mul,
                                      add=self.add, **kwargs)

    @staticmethod
    def add_arguments(parser, log=None, mul=None, add=None):
        """
        Add spectrogram scaling related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        log : bool, optional
            Take the logarithm of the spectrogram.
        mul : float, optional
            Multiply the magnitude spectrogram with this factor before taking
            the logarithm.
        add : float, optional
            Add this value before taking the logarithm of the magnitudes.

        Returns
        -------
        argparse argument group
            Spectrogram scaling argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.

        """
        # add log related options to the existing parser
        g = parser.add_argument_group('magnitude scaling arguments')
        # log
        if log is True:
            g.add_argument('--linear', dest='log', action='store_const',
                           const=None, default=LOG,
                           help='linear magnitudes [default=logarithmic]')
        elif log is False:
            g.add_argument('--log', action='store_const',
                           const=LOG, default=None,
                           help='logarithmic magnitudes [default=linear]')
        # mul
        if mul is not None:
            g.add_argument('--mul', action='store', type=float,
                           default=mul, help='multiplier (before taking '
                           'the log) [default=%(default)i]')
        # add
        if add is not None:
            g.add_argument('--add', action='store', type=float,
                           default=add, help='value added (before taking '
                           'the log) [default=%(default)i]')
        # return the group
        return g


# logarithmic filtered spectrogram class
class LogarithmicFilteredSpectrogram(LogarithmicSpectrogram,
                                     FilteredSpectrogram):
    """
    LogarithmicFilteredSpectrogram class.

    Parameters
    ----------
    spectrogram : :class:`FilteredSpectrogram` instance
        Filtered spectrogram.
    kwargs : dict, optional
        If no :class:`FilteredSpectrogram` instance was given, one is
        instantiated with these additional keyword arguments and
        logarithmically scaled afterwards, i.e. passed to
        :class:`LogarithmicSpectrogram`.

    Notes
    -----
    For the filtering and scaling parameters, please refer to
    :class:`FilteredSpectrogram` and :class:`LogarithmicSpectrogram`.

    See Also
    --------
    :class:`FilteredSpectrogram`
    :class:`LogarithmicSpectrogram`

    Examples
    --------
    Create a :class:`LogarithmicFilteredSpectrogram` from a
    :class:`Spectrogram` (or anything it can be instantiated from. This is
    mainly a convenience class which first filters the spectrogram and then
    scales it logarithmically.

    >>> spec = LogarithmicFilteredSpectrogram('tests/data/audio/sample.wav')
    >>> spec  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    LogarithmicFilteredSpectrogram([[ 0.82358, 0.86341, ...,
                                      0.02295, 0.02719],
                                    [ 0.97509, 0.98658, ...,
                                      0.03223, 0.0375 ],
                                    ...,
                                    [ 1.04322, 0.32637, ...,
                                      0.02065, 0.01821],
                                    [ 0.98236, 0.89276, ...,
                                      0.01587, 0.0144 ]], dtype=float32)
    >>> spec.shape
    (281, 81)
    >>> spec.filterbank  # doctest: +ELLIPSIS
    LogarithmicFilterbank([[...]], dtype=float32)
    >>> spec.min()  # doctest: +ELLIPSIS
    LogarithmicFilteredSpectrogram(0.00830..., dtype=float32)

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, spectrogram, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, spectrogram, **kwargs):
        # get the log args
        mul = kwargs.pop('mul', MUL)
        add = kwargs.pop('add', ADD)
        # instantiate a FilteredSpectrogram if needed
        if not isinstance(spectrogram, FilteredSpectrogram):
            spectrogram = FilteredSpectrogram(spectrogram, **kwargs)
        # take the logarithm
        data = LogarithmicSpectrogram(spectrogram, mul=mul, add=add, **kwargs)
        # cast as LogarithmicFilteredSpectrogram
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.mul = data.mul
        obj.add = data.add
        # and those from the given spectrogram
        obj.stft = spectrogram.stft
        obj.filterbank = spectrogram.filterbank
        obj.bin_frequencies = spectrogram.bin_frequencies
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.filterbank = getattr(obj, 'filterbank', None)
        self.mul = getattr(obj, 'mul', MUL)
        self.add = getattr(obj, 'add', ADD)
        super(LogarithmicFilteredSpectrogram, self).__array_finalize__(obj)


class LogarithmicFilteredSpectrogramProcessor(Processor):
    """
    Logarithmic Filtered Spectrogram Processor class.

    Parameters
    ----------
    filterbank : :class:`.audio.filters.Filterbank`
        Filterbank used to filter a spectrogram.
    num_bands : int
        Number of bands (per octave).
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    fref : float, optional
        Tuning frequency of the filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filter of the filterbank to area 1.
    unique_filters : bool, optional
        Indicate if the filterbank should contain only unique filters, i.e.
        remove duplicate filters resulting from insufficient resolution at
        low frequencies.
    mul : float, optional
        Multiply the magnitude spectrogram with this factor before taking the
        logarithm.
    add : float, optional
        Add this value before taking the logarithm of the magnitudes.

    """

    def __init__(self, filterbank=FILTERBANK, num_bands=NUM_BANDS, fmin=FMIN,
                 fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS, mul=MUL, add=ADD, **kwargs):
        # pylint: disable=unused-argument
        self.filterbank = filterbank
        self.num_bands = num_bands
        self.fmin = fmin
        self.fmax = fmax
        self.fref = fref
        self.norm_filters = norm_filters
        self.unique_filters = unique_filters
        self.mul = mul
        self.add = add

    def process(self, data, **kwargs):
        """
        Perform filtering and logarithmic scaling of a spectrogram.

        Parameters
        ----------
        data : numpy array
            Data to be processed.
        kwargs : dict
            Keyword arguments passed to
            :class:`LogarithmicFilteredSpectrogram`.

        Returns
        -------
        log_filt_spec : :class:`LogarithmicFilteredSpectrogram` instance
            Logarithmically scaled filtered spectrogram.

        """
        # instantiate a LogarithmicFilteredSpectrogram
        return LogarithmicFilteredSpectrogram(
            data, filterbank=self.filterbank, num_bands=self.num_bands,
            fmin=self.fmin, fmax=self.fmax, fref=self.fref,
            norm_filters=self.norm_filters, unique_filters=self.unique_filters,
            mul=self.mul, add=self.add, **kwargs)


# spectrogram difference stuff
DIFF_RATIO = 0.5
DIFF_FRAMES = None
DIFF_MAX_BINS = None
POSITIVE_DIFFS = False


class SpectrogramDifference(Spectrogram):
    """
    SpectrogramDifference class.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        Spectrogram.
    diff_ratio : float, optional
        Calculate the difference to the frame at which the window used for the
        STFT yields this ratio of the maximum height.
    diff_frames : int, optional
        Calculate the difference to the `diff_frames`-th previous frame (if
        set, this overrides the value calculated from the `diff_ratio`)
    diff_max_bins : int, optional
        Apply a maximum filter with this width (in bins in frequency dimension)
        to the spectrogram the difference is calculated to.
    positive_diffs : bool, optional
        Keep only the positive differences, i.e. set all diff values < 0 to 0.
    kwargs : dict, optional
        If no :class:`Spectrogram` instance was given, one is instantiated with
        these additional keyword arguments.

    Notes
    -----
    The SuperFlux algorithm [1]_ uses a maximum filtered spectrogram with 3
    `diff_max_bins` together with a 24 band logarithmic filterbank to calculate
    the difference spectrogram with a `diff_ratio` of 0.5.

    The effect of this maximum filter applied to the spectrogram is that the
    magnitudes are "widened" in frequency direction, i.e. the following
    difference calculation is less sensitive against frequency fluctuations.
    This effect is exploited to suppress false positive energy fragments
    originating from vibrato.

    References
    ----------
    .. [1] Sebastian Böck and Gerhard Widmer
           "Maximum Filter Vibrato Suppression for Onset Detection"
           Proceedings of the 16th International Conference on Digital Audio
           Effects (DAFx), 2013.

    Examples
    --------
    To obtain the SuperFlux feature as described above first create a filtered
    and logarithmically spaced spectrogram:

    >>> spec = LogarithmicFilteredSpectrogram('tests/data/audio/sample.wav', \
                                              num_bands=24, fps=200)
    >>> spec  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    LogarithmicFilteredSpectrogram([[ 0.82358, 0.86341, ...,
                                      0.02809, 0.02672],
                                    [ 0.92514, 0.93211, ...,
                                      0.03607, 0.0317 ],
                                    ...,
                                    [ 1.03826, 0.767  , ...,
                                      0.01814, 0.01138],
                                    [ 0.98236, 0.89276, ...,
                                      0.01669, 0.00919]], dtype=float32)
    >>> spec.shape
    (561, 140)

    Then use the temporal first order difference and apply a maximum filter
    with 3 bands, keeping only the positive differences (i.e. rise in energy):

    >>> superflux = SpectrogramDifference(spec, diff_max_bins=3, \
                                          positive_diffs=True)
    >>> superflux  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    SpectrogramDifference([[ 0.     , 0. , ...,  0. ,  0. ],
                           [ 0.     , 0. , ...,  0. ,  0. ],
                           ...,
                           [ 0.01941, 0. , ...,  0. ,  0. ],
                           [ 0.     , 0. , ...,  0. ,  0. ]], dtype=float32)

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, spectrogram, diff_ratio=DIFF_RATIO,
                 diff_frames=DIFF_FRAMES, diff_max_bins=DIFF_MAX_BINS,
                 positive_diffs=POSITIVE_DIFFS, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, spectrogram, diff_ratio=DIFF_RATIO,
                diff_frames=DIFF_FRAMES, diff_max_bins=DIFF_MAX_BINS,
                positive_diffs=POSITIVE_DIFFS, **kwargs):
        # instantiate a Spectrogram if needed
        if not isinstance(spectrogram, Spectrogram):
            # try to instantiate a Spectrogram object
            spectrogram = Spectrogram(spectrogram, **kwargs)

        # calculate the number of diff frames to use
        if diff_frames is None:
            # calculate the number of diff_frames on basis of the diff_ratio
            # get the first sample with a higher magnitude than given ratio
            window = spectrogram.stft.window
            sample = np.argmax(window > float(diff_ratio) * max(window))
            diff_samples = len(spectrogram.stft.window) / 2 - sample
            # convert to frames
            hop_size = spectrogram.stft.frames.hop_size
            diff_frames = round(diff_samples / hop_size)

        # use at least 1 frame
        diff_frames = max(1, int(diff_frames))

        # init matrix
        diff = np.zeros_like(spectrogram)

        # apply a maximum filter to diff_spec if needed
        if diff_max_bins is not None and diff_max_bins > 1:
            from scipy.ndimage.filters import maximum_filter
            # widen the spectrogram in frequency dimension
            size = [1, int(diff_max_bins)]
            diff_spec = maximum_filter(spectrogram, size=size)
        else:
            diff_spec = spectrogram
        # calculate the diff
        diff[diff_frames:] = (spectrogram[diff_frames:] -
                              diff_spec[: -diff_frames])
        # positive differences only?
        if positive_diffs:
            np.maximum(diff, 0, out=diff)

        # cast as FilteredSpectrogram
        obj = np.asarray(diff).view(cls)
        # save additional attributes
        obj.spectrogram = spectrogram
        obj.diff_ratio = diff_ratio
        obj.diff_frames = diff_frames
        obj.diff_max_bins = diff_max_bins
        obj.positive_diffs = positive_diffs
        # and those from the given spectrogram
        obj.filterbank = spectrogram.filterbank
        obj.bin_frequencies = spectrogram.bin_frequencies
        obj.mul = spectrogram.mul
        obj.add = spectrogram.add
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.diff_ratio = getattr(obj, 'diff_ratio', 0.5)
        self.diff_frames = getattr(obj, 'diff_frames', None)
        self.diff_max_bins = getattr(obj, 'diff_max_bins', None)
        self.positive_diffs = getattr(obj, 'positive_diffs', False)
        super(SpectrogramDifference, self).__array_finalize__(obj)

    def positive_diff(self):
        """Positive diff."""
        return np.maximum(self, 0)


class SpectrogramDifferenceProcessor(Processor):
    """
    Difference Spectrogram Processor class.

    Parameters
    ----------
    diff_ratio : float, optional
        Calculate the difference to the frame at which the window used for the
        STFT yields this ratio of the maximum height.
    diff_frames : int, optional
        Calculate the difference to the `diff_frames`-th previous frame (if
        set, this overrides the value calculated from the `diff_ratio`)
    diff_max_bins : int, optional
        Apply a maximum filter with this width (in bins in frequency dimension)
        to the spectrogram the difference is calculated to.
    positive_diffs : bool, optional
        Keep only the positive differences, i.e. set all diff values < 0 to 0.
    stack_diffs : numpy stacking function, optional
        If 'None', only the differences are returned. If set, the diffs are
        stacked with the underlying spectrogram data according to the `stack`
        function:

        - ``np.vstack``
          the differences and spectrogram are stacked vertically, i.e. in time
          direction,
        - ``np.hstack``
          the differences and spectrogram are stacked horizontally, i.e. in
          frequency direction,
        - ``np.dstack``
          the differences and spectrogram are stacked in depth, i.e. return
          them as a 3D representation with depth as the third dimension.

    """

    def __init__(self, diff_ratio=DIFF_RATIO, diff_frames=DIFF_FRAMES,
                 diff_max_bins=DIFF_MAX_BINS, positive_diffs=POSITIVE_DIFFS,
                 stack_diffs=None, **kwargs):
        # pylint: disable=unused-argument
        self.diff_ratio = diff_ratio
        self.diff_frames = diff_frames
        self.diff_max_bins = diff_max_bins
        self.positive_diffs = positive_diffs
        self.stack_diffs = stack_diffs

    def process(self, data, **kwargs):
        """
        Perform a temporal difference calculation on the given data.

        Parameters
        ----------
        data : numpy array
            Data to be processed.
        kwargs : dict
            Keyword arguments passed to :class:`SpectrogramDifference`.

        Returns
        -------
        diff : :class:`SpectrogramDifference` instance
            Spectrogram difference.

        """
        # instantiate a SpectrogramDifference
        diff = SpectrogramDifference(data, diff_ratio=self.diff_ratio,
                                     diff_frames=self.diff_frames,
                                     diff_max_bins=self.diff_max_bins,
                                     positive_diffs=self.positive_diffs,
                                     **kwargs)
        # decide if we need to stack the diff on the data or just return it
        if self.stack_diffs is None:
            return diff
        else:
            # we can't use `data` directly, because it could be a str
            # we ave to access diff.spectrogram (i.e. converted data)
            return self.stack_diffs((diff.spectrogram, diff))

    @staticmethod
    def add_arguments(parser, diff=None, diff_ratio=None, diff_frames=None,
                      diff_max_bins=None, positive_diffs=None):
        """
        Add spectrogram difference related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        diff : bool, optional
            Take the difference of the spectrogram.
        diff_ratio : float, optional
            Calculate the difference to the frame at which the window used for
            the STFT yields this ratio of the maximum height.
        diff_frames : int, optional
            Calculate the difference to the `diff_frames`-th previous frame (if
            set, this overrides the value calculated from the `diff_ratio`)
        diff_max_bins : int, optional
            Apply a maximum filter with this width (in bins in frequency
            dimension) to the spectrogram the difference is calculated to.
        positive_diffs : bool, optional
            Keep only the positive differences, i.e. set all diff values < 0
            to 0.

        Returns
        -------
        argparse argument group
            Spectrogram difference argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.

        Only the `diff_frames` parameter behaves differently, it is included
        if either the `diff_ratio` is set or a value != 'None' is given.

        """
        # add diff related options to the existing parser
        g = parser.add_argument_group('spectrogram difference arguments')
        # diff
        if diff is True:
            g.add_argument('--no_diff', dest='diff', action='store_false',
                           help='use the spectrogram [default=differences '
                                'of the spectrogram]')
        elif diff is False:
            g.add_argument('--diff', action='store_true',
                           help='use the differences of the spectrogram '
                                '[default=spectrogram]')
        # diff ratio
        if diff_ratio is not None:
            g.add_argument('--diff_ratio', action='store', type=float,
                           default=diff_ratio,
                           help='calculate the difference to the frame at '
                                'which the window of the STFT have this ratio '
                                'of the maximum height '
                                '[default=%(default).1f]')
        # diff frames
        if diff_ratio is not None or diff_frames:
            g.add_argument('--diff_frames', action='store', type=int,
                           default=diff_frames,
                           help='calculate the difference to the N-th previous'
                                ' frame (this overrides the value calculated '
                                'with `diff_ratio`) [default=%(default)s]')
        # positive diffs
        if positive_diffs is True:
            g.add_argument('--all_diffs', dest='positive_diffs',
                           action='store_false',
                           help='keep both positive and negative diffs '
                                '[default=only the positive diffs]')
        elif positive_diffs is False:
            g.add_argument('--positive_diffs', action='store_true',
                           help='keep only positive diffs '
                                '[default=positive and negative diffs]')
        # add maximum filter related options to the existing parser
        if diff_max_bins is not None:
            g.add_argument('--max_bins', action='store', type=int,
                           dest='diff_max_bins', default=diff_max_bins,
                           help='apply a maximum filter with this width (in '
                                'frequency bins) [default=%(default)d]')
        # return the group
        return g


class SuperFluxProcessor(SequentialProcessor):
    """
    Spectrogram processor which sets the default values suitable for the
    SuperFlux algorithm.

    """
    # pylint: disable=too-many-ancestors

    def __init__(self, **kwargs):
        from .stft import ShortTimeFourierTransformProcessor
        # set the default values (can be overwritten if set)
        # we need an un-normalized LogarithmicFilterbank with 24 bands
        filterbank = kwargs.pop('filterbank', FILTERBANK)
        num_bands = kwargs.pop('num_bands', 24)
        norm_filters = kwargs.pop('norm_filters', False)
        # we want max filtered diffs
        diff_ratio = kwargs.pop('diff_ratio', 0.5)
        diff_max_bins = kwargs.pop('diff_max_bins', 3)
        positive_diffs = kwargs.pop('positive_diffs', True)
        # processing chain
        stft = ShortTimeFourierTransformProcessor(**kwargs)
        spec = SpectrogramProcessor(**kwargs)
        filt = FilteredSpectrogramProcessor(filterbank=filterbank,
                                            num_bands=num_bands,
                                            norm_filters=norm_filters,
                                            **kwargs)
        log = LogarithmicSpectrogramProcessor(**kwargs)
        diff = SpectrogramDifferenceProcessor(diff_ratio=diff_ratio,
                                              diff_max_bins=diff_max_bins,
                                              positive_diffs=positive_diffs,
                                              **kwargs)
        # sequentially process everything
        super(SuperFluxProcessor, self).__init__([stft, spec, filt, log, diff])


class MultiBandSpectrogram(FilteredSpectrogram):
    """
    MultiBandSpectrogram class.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        Spectrogram.
    crossover_frequencies : list or numpy array
        List of crossover frequencies at which the `spectrogram` is split
        into multiple bands.
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filter bands of the filterbank to area 1.
    unique_filters : bool, optional
        Indicate if the filterbank should contain only unique filters, i.e.
        remove duplicate filters resulting from insufficient resolution at
        low frequencies.
    kwargs : dict, optional
        If no :class:`Spectrogram` instance was given, one is instantiated
        with these additional keyword arguments.

    Notes
    -----
    The MultiBandSpectrogram is implemented as a :class:`Spectrogram` which
    uses a :class:`.audio.filters.RectangularFilterbank` to combine multiple
    frequency bins.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, spectrogram, crossover_frequencies, fmin=FMIN,
                 fmax=FMAX, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, spectrogram, crossover_frequencies, fmin=FMIN, fmax=FMAX,
                norm_filters=NORM_FILTERS, unique_filters=UNIQUE_FILTERS,
                **kwargs):
        from .filters import RectangularFilterbank
        # instantiate a Spectrogram if needed
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)
        # create a rectangular filterbank
        filterbank = RectangularFilterbank(spectrogram.bin_frequencies,
                                           crossover_frequencies,
                                           fmin=fmin, fmax=fmax,
                                           norm_filters=norm_filters,
                                           unique_filters=unique_filters)
        # filter the spectrogram
        data = np.dot(spectrogram, filterbank)
        # cast as FilteredSpectrogram
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.spectrogram = spectrogram
        obj.filterbank = filterbank
        obj.bin_frequencies = filterbank.center_frequencies
        obj.crossover_frequencies = crossover_frequencies
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.spectrogram = getattr(obj, 'spectrogram', None)
        self.filterbank = getattr(obj, 'filterbank', None)
        self.bin_frequencies = getattr(obj, 'bin_frequencies', None)
        self.crossover_frequencies = getattr(obj, 'crossover_frequencies',
                                             None)


class MultiBandSpectrogramProcessor(Processor):
    """
    Spectrogram processor which combines the spectrogram magnitudes into
    multiple bands.

    Parameters
    ----------
    crossover_frequencies : list or numpy array
        List of crossover frequencies at which a spectrogram is split into
        the individual bands.
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filter bands of the filterbank to area 1.
    unique_filters : bool, optional
        Indicate if the filterbank should contain only unique filters, i.e.
        remove duplicate filters resulting from insufficient resolution at
        low frequencies.

    """

    def __init__(self, crossover_frequencies, fmin=FMIN, fmax=FMAX,
                 norm_filters=NORM_FILTERS, unique_filters=UNIQUE_FILTERS,
                 **kwargs):
        # pylint: disable=unused-argument
        self.crossover_frequencies = np.array(crossover_frequencies)
        self.fmin = fmin
        self.fmax = fmax
        self.norm_filters = norm_filters
        self.unique_filters = unique_filters

    def process(self, data, **kwargs):
        """
        Return the a multi-band representation of the given data.

        Parameters
        ----------
        data : numpy array
            Data to be processed.
        kwargs : dict
            Keyword arguments passed to :class:`MultiBandSpectrogram`.

        Returns
        -------
        multi_band_spec : :class:`MultiBandSpectrogram` instance
            Spectrogram split into multiple bands.

        """
        # instantiate a MultiBandSpectrogram
        return MultiBandSpectrogram(
            data, crossover_frequencies=self.crossover_frequencies,
            fmin=self.fmin, fmax=self.fmax, norm_filters=self.norm_filters,
            unique_filters=self.unique_filters, **kwargs)
