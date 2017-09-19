# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains spectrogram related functionality.

"""

from __future__ import absolute_import, division, print_function

import inspect
import numpy as np

from ..processors import Processor, SequentialProcessor, BufferProcessor
from .filters import (Filterbank, LogarithmicFilterbank, NUM_BANDS, FMIN, FMAX,
                      A4, NORM_FILTERS, UNIQUE_FILTERS)


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


# some functions working on magnitude spectra
def adaptive_whitening(spec, floor=0.5, relaxation=10):
    """
    Return an adaptively whitened version of the magnitude spectrogram.

    Parameters
    ----------
    spec : numpy array
        Magnitude spectrogram.
    floor : float, optional
        Floor coefficient.
    relaxation : int, optional
        Relaxation time [frames].

    Returns
    -------
    whitened_spec : numpy array
        The whitened magnitude spectrogram.

    References
    ----------

    .. [1] Dan Stowell and Mark Plumbley,
           "Adaptive Whitening For Improved Real-time Audio Onset Detection",
           Proceedings of the International Computer Music Conference (ICMC),
           2007

    """
    raise NotImplementedError("check if adaptive_whitening returns meaningful "
                              "results")
    relaxation = 10.0 ** (-6. * relaxation)
    p = np.zeros_like(spec)
    # iterate over all frames
    for f, frame in enumerate(spec):
        if f > 0:
            p[f] = np.maximum(frame, floor, relaxation * p[f - 1])
        else:
            p[f] = np.maximum(frame, floor)
    # return the whitened spectrogram
    return spec / p


def statistical_spectrum_descriptors(spectrogram):
    """
    Statistical Spectrum Descriptors of the STFT.

    Parameters
    ----------
    spectrogram : numpy array
        Magnitude spectrogram.

    Returns
    -------
    statistical_spectrum_descriptors : dict
        Statistical spectrum descriptors of the spectrogram.

    References
    ----------
    .. [1] Thomas Lidy and Andreas Rauber,
           "Evaluation of Feature Extractors and Psycho-acoustic
           Transformations for Music Genre Classification",
           Proceedings of the 6th International Conference on Music Information
           Retrieval (ISMIR), 2005.

    """
    from scipy.stats import skew, kurtosis
    return {'mean': np.mean(spectrogram, axis=0),
            'median': np.median(spectrogram, axis=0),
            'variance': np.var(spectrogram, axis=0),
            'skewness': skew(spectrogram, axis=0),
            'kurtosis': kurtosis(spectrogram, axis=0),
            'min': np.min(spectrogram, axis=0),
            'max': np.max(spectrogram, axis=0)}


def tuning_frequency(spectrogram, bin_frequencies, num_hist_bins=15, fref=A4):
    """
    Determines the tuning frequency of the audio signal based on the given
    magnitude spectrogram.

    To determine the tuning frequency, a weighted histogram of relative
    deviations of the spectrogram bins towards the closest semitones is built.

    Parameters
    ----------
    spectrogram : numpy array
        Magnitude spectrogram.
    bin_frequencies : numpy array
        Frequencies of the spectrogram bins [Hz].
    num_hist_bins : int, optional
        Number of histogram bins.
    fref : float, optional
        Reference tuning frequency [Hz].

    Returns
    -------
    tuning_frequency : float
        Tuning frequency [Hz].

    """
    from .filters import hz2midi
    # interval of spectral bins from the reference frequency in semitones
    semitone_int = hz2midi(bin_frequencies, fref=fref)
    # deviation from the next semitone
    semitone_dev = semitone_int - np.round(semitone_int)
    # np.histogram accepts bin edges, so we need to apply an offset and use 1
    # more bin than given to build a histogram
    offset = 0.5 / num_hist_bins
    hist_bins = np.linspace(-0.5 - offset, 0.5 + offset, num_hist_bins + 1)
    histogram = np.histogram(semitone_dev, weights=np.sum(spectrogram, axis=0),
                             bins=hist_bins)
    # deviation of the bins (centre of the bins)
    dev_bins = (histogram[1][:-1] + histogram[1][1:]) / 2.
    # dominant deviation
    dev = dev_bins[np.argmax(histogram[0])]
    # calculate the tuning frequency
    return fref * 2. ** (dev / 12.)


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
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.stft = getattr(obj, 'stft', None)

    @property
    def num_frames(self):
        """Number of frames."""
        return len(self)

    @property
    def num_bins(self):
        """Number of bins."""
        return int(self.shape[1])

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return self.stft.bin_frequencies

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

    def tuning_frequency(self, **kwargs):
        """
        Return the tuning frequency of the audio signal based on peaks of the
        spectrogram.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to :func:`tuning_frequency`.

        Returns
        -------
        tuning_frequency : float
            Tuning frequency of the spectrogram.

        """
        from scipy.ndimage.filters import maximum_filter
        # widen the spectrogram in frequency dimension
        max_spec = maximum_filter(self, size=[1, 3])
        # get the peaks of the spectrogram
        max_spec = self * (self == max_spec)
        # determine the tuning frequency
        return tuning_frequency(max_spec, self.bin_frequencies, **kwargs)


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
        # and those from the given spectrogram
        obj.stft = spectrogram.stft
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.stft = getattr(obj, 'stft', None)
        self.filterbank = getattr(obj, 'filterbank', None)

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        # use the center frequencies of the filterbank as bin_frequencies
        return self.filterbank.center_frequencies


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
        # update arguments passed to FilteredSpectrogram
        args = dict(filterbank=self.filterbank, num_bands=self.num_bands,
                    fmin=self.fmin, fmax=self.fmax, fref=self.fref,
                    norm_filters=self.norm_filters,
                    unique_filters=self.unique_filters)
        args.update(kwargs)
        # instantiate a FilteredSpectrogram and return it
        data = FilteredSpectrogram(data, **args)
        # cache the filterbank
        self.filterbank = data.filterbank
        return data


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
        obj.spectrogram = spectrogram
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.stft = getattr(obj, 'stft', None)
        self.spectrogram = getattr(obj, 'spectrogram', None)
        self.mul = getattr(obj, 'mul', MUL)
        self.add = getattr(obj, 'add', ADD)

    @property
    def filterbank(self):
        """Filterbank."""
        return self.spectrogram.filterbank

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return self.spectrogram.bin_frequencies


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
        # update arguments passed to LogarithmicSpectrogram
        args = dict(log=self.log, mul=self.mul, add=self.add)
        args.update(kwargs)
        # instantiate a LogarithmicSpectrogram
        return LogarithmicSpectrogram(data, **args)

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
        obj.spectrogram = spectrogram
        # return the object
        return obj

    @property
    def filterbank(self):
        """Filterbank."""
        return self.spectrogram.filterbank

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return self.filterbank.center_frequencies


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
        # update arguments passed to LogarithmicFilteredSpectrogram
        args = dict(filterbank=self.filterbank, num_bands=self.num_bands,
                    fmin=self.fmin, fmax=self.fmax, fref=self.fref,
                    norm_filters=self.norm_filters,
                    unique_filters=self.unique_filters, mul=self.mul,
                    add=self.add)
        args.update(kwargs)
        # instantiate a LogarithmicFilteredSpectrogram
        data = LogarithmicFilteredSpectrogram(data, **args)
        # cache the filterbank
        self.filterbank = data.filterbank
        return data


# spectrogram difference stuff
DIFF_RATIO = 0.5
DIFF_FRAMES = None
DIFF_MAX_BINS = None
POSITIVE_DIFFS = False


def _diff_frames(diff_ratio, hop_size, frame_size, window=np.hanning):
    """
    Compute the number of `diff_frames` for the given ratio of overlap.

    Parameters
    ----------
    diff_ratio : float
        Ratio of overlap of windows of two consecutive STFT frames.
    hop_size : int
        Samples between two adjacent frames.
    frame_size : int
        Size of one frames in samples.
    window : numpy ufunc or array
        Window funtion.

    Returns
    -------
    diff_frames : int
        Number of frames to calculate the difference to.

    """
    # calculate the number of diff frames on basis of the diff_ratio
    # first sample of the window with a higher magnitude than given ratio
    if hasattr(window, '__call__'):
        # Note: if only a window function is given (default in audio.stft),
        #       generate a window of size `frame_size` with the given shape
        window = window(frame_size)
    sample = np.argmax(window > float(diff_ratio) * max(window))
    diff_samples = len(window) / 2 - sample
    # convert to frames, must be at least 1
    return int(max(1, round(diff_samples / hop_size)))


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
    keep_dims : bool, optional
        Indicate if the dimensions (i.e. shape) of the spectrogram should be
        kept.
    kwargs : dict, optional
        If no :class:`Spectrogram` instance was given, one is instantiated with
        these additional keyword arguments.

    Notes
    -----
    The first `diff_frames` frames will have a value of 0.

    If `keep_dims` is 'True' the returned difference has the same shape as the
    spectrogram. This is needed if the diffs should be stacked on top of it.
    If set to 'False', the length will be `diff_frames` frames shorter (mostly
    used by the SpectrogramDifferenceProcessor which first buffers that many
    frames.

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
                 positive_diffs=POSITIVE_DIFFS, keep_dims=True, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, spectrogram, diff_ratio=DIFF_RATIO,
                diff_frames=DIFF_FRAMES, diff_max_bins=DIFF_MAX_BINS,
                positive_diffs=POSITIVE_DIFFS, keep_dims=True, **kwargs):
        # instantiate a Spectrogram if needed
        if not isinstance(spectrogram, Spectrogram):
            # try to instantiate a Spectrogram object
            spectrogram = Spectrogram(spectrogram, **kwargs)

        # calculate the number of diff frames to use
        if diff_frames is None:
            diff_frames = _diff_frames(
                diff_ratio, hop_size=spectrogram.stft.frames.hop_size,
                frame_size=spectrogram.stft.frames.frame_size,
                window=spectrogram.stft.window)

        # apply a maximum filter to diff_spec if needed
        if diff_max_bins is not None and diff_max_bins > 1:
            from scipy.ndimage.filters import maximum_filter
            # widen the spectrogram in frequency dimension
            size = [1, int(diff_max_bins)]
            diff_spec = maximum_filter(spectrogram, size=size)
        else:
            diff_spec = spectrogram

        # calculate the diff
        if keep_dims:
            diff = np.zeros_like(spectrogram)
            diff[diff_frames:] = (spectrogram[diff_frames:] -
                                  diff_spec[:-diff_frames])
        else:
            diff = spectrogram[diff_frames:] - diff_spec[:-diff_frames]

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

    @property
    def bin_frequencies(self):
        """Bin frequencies."""
        return self.spectrogram.bin_frequencies

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
        # attributes needed for stateful processing
        # Note: do not init the buffer here, since it depends on the data
        self._buffer = None

    def __getstate__(self):
        # copy everything to a pickleable object
        state = self.__dict__.copy()
        # do not pickle attributes needed for stateful processing
        state.pop('_buffer', None)
        return state

    def __setstate__(self, state):
        # restore pickled instance attributes
        self.__dict__.update(state)
        # add non-pickled attributes needed for stateful processing
        self._buffer = None

    def process(self, data, reset=True, **kwargs):
        """
        Perform a temporal difference calculation on the given data.

        Parameters
        ----------
        data : numpy array
            Data to be processed.
        reset : bool, optional
            Reset the spectrogram buffer before computing the difference.
        kwargs : dict
            Keyword arguments passed to :class:`SpectrogramDifference`.

        Returns
        -------
        diff : :class:`SpectrogramDifference` instance
            Spectrogram difference.

        Notes
        -----
        If `reset` is 'True', the first `diff_frames` differences will be 0.

        """
        # update arguments passed to SpectrogramDifference
        args = dict(diff_ratio=self.diff_ratio, diff_frames=self.diff_frames,
                    diff_max_bins=self.diff_max_bins,
                    positive_diffs=self.positive_diffs)
        args.update(kwargs)
        # calculate the number of diff frames
        if self.diff_frames is None:
            # Note: use diff_ration from args, not self.diff_ratio
            self.diff_frames = _diff_frames(
                args['diff_ratio'], frame_size=data.stft.frames.frame_size,
                hop_size=data.stft.frames.hop_size, window=data.stft.window)
        # init buffer or shift it
        if self._buffer is None or reset:
            # put diff_frames NaNs before the data (will be replaced by 0s)
            init = np.empty((self.diff_frames, data.shape[1]))
            init[:] = np.nan
            data = np.insert(data, 0, init, axis=0)
            # use the data for the buffer
            self._buffer = BufferProcessor(init=data)
        else:
            # shift buffer by length of data and put new data at end of buffer
            data = self._buffer(data)
        # compute difference based on this data (reduce 1st dimension)
        diff = SpectrogramDifference(data, keep_dims=False, **args)
        # set all NaN-diffs to 0
        diff[np.isnan(diff)] = 0
        # stack the diff and the data if needed
        if self.stack_diffs is None:
            return diff
        else:
            # Note: don't use `data` directly, because it could be a str
            #       we ave to access diff.spectrogram (i.e. converted data)
            return self.stack_diffs((diff.spectrogram[self.diff_frames:],
                                     diff))

    def reset(self):
        """Reset the SpectrogramDifferenceProcessor."""
        # reset cached spectrogram data
        self._buffer = None

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
        obj.crossover_frequencies = crossover_frequencies
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.spectrogram = getattr(obj, 'spectrogram', None)
        self.filterbank = getattr(obj, 'filterbank', None)
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
        # update arguments passed to MultiBandSpectrogram
        args = dict(crossover_frequencies=self.crossover_frequencies,
                    fmin=self.fmin, fmax=self.fmax,
                    norm_filters=self.norm_filters,
                    unique_filters=self.unique_filters)
        args.update(kwargs)
        # instantiate a MultiBandSpectrogram
        return MultiBandSpectrogram(data, **args)


class SemitoneBandpassSpectrogram(FilteredSpectrogram):
    """
    Construct a semitone spectrogram by using a time domain filterbank of
    bandpass filters as described in [1]_.

    Parameters
    ----------
    signal : Signal
        Signal instance.
    fps : float, optional
        Frame rate of the spectrogram [Hz].
    fmin : float, optional
        Lowest frequency of the spectrogram [Hz].
    fmax : float, optional
        Highest frequency of the spectrogram [Hz].

    References
    ----------
    .. [1] Meinard Müller,
           "Information retrieval for music and motion", Springer, 2007.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, signal, fps=50., fmin=27.5, fmax=4200.):
        # this method is for documentation purposes only
        pass

    def __new__(cls, signal, fps=50., fmin=27.5, fmax=4200.):
        from scipy.signal import filtfilt
        from .filters import SemitoneBandpassFilterbank
        from .signal import FramedSignal, Signal, energy, resample
        # check if we got a mono Signal
        if not isinstance(signal, Signal) or signal.num_channels != 1:
            signal = Signal(signal, num_channels=1)
        sample_rate = float(signal.sample_rate)
        # keep a reference to the original signal
        signal_ = signal
        # determine how many frames the filtered signal will have
        num_frames = np.round(len(signal) * fps / sample_rate) + 1
        # compute the energy of the frames of the bandpass filtered signal
        filterbank = SemitoneBandpassFilterbank(fmin=fmin, fmax=fmax)
        bands = []
        for filt, band_sample_rate in zip(filterbank.filters,
                                          filterbank.band_sample_rates):
            # frames should overlap 50%
            frame_size = np.round(2 * band_sample_rate / float(fps))
            # down-sample audio if needed
            if band_sample_rate != signal.sample_rate:
                signal = resample(signal_, band_sample_rate)
            # filter the signal
            b, a = filt
            filtered_signal = filtfilt(b, a, signal)
            # normalise the signal if it has an integer dtype
            try:
                filtered_signal /= np.iinfo(signal.dtype).max
            except ValueError:
                pass
            # compute the energy of the filtered signal
            # Note: 1) the energy of the signal is computed with respect to the
            #          reference sampling rate as in the MATLAB chroma toolbox
            #       2) we do not sum here, but rather after splitting the
            #          signal into overlapping frames to avoid doubled
            #          computation due to the overlapping frames
            filtered_signal = filtered_signal ** 2 / band_sample_rate * 22050.
            # split into overlapping frames
            frames = FramedSignal(filtered_signal, frame_size=frame_size,
                                  fps=fps, sample_rate=band_sample_rate,
                                  num_frames=num_frames)
            # finally sum the energy of all frames
            bands.append(np.sum(frames, axis=1))
        # cast as SemitoneBandpassSpectrogram
        obj = np.vstack(bands).T.view(cls)
        # save additional attributes
        obj.filterbank = filterbank
        obj.fps = fps
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.filterbank = getattr(obj, 'filterbank', None)
        self.fps = getattr(obj, 'fps', None)
