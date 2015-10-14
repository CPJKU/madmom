# encoding: utf-8
"""
This file contains spectrogram related functionality.

"""

import numpy as np

from madmom.processors import Processor, SequentialProcessor, ParallelProcessor
from .stft import (PropertyMixin, ShortTimeFourierTransform,
                   ShortTimeFourierTransformProcessor)
from .filters import (LogarithmicFilterbank, NUM_BANDS, FMIN, FMAX, A4,
                      NORM_FILTERS, UNIQUE_FILTERS)


def spec(stft):
    """
    Returns the magnitudes of the complex Short Time Fourier Transform of a
    signal.

    :param stft: complex STFT of a signal
    :return:     magnitude spectrogram

    """
    return np.abs(stft)


# some functions working on magnitude spectra
def adaptive_whitening(spec, floor=0.5, relaxation=10):
    """
    Return an adaptively whitened version of the magnitude spectrogram.

    :param spec:       magnitude spectrogram [numpy array]
    :param floor:      floor coefficient [float]
    :param relaxation: relaxation time [frames]
    :return:           the whitened magnitude spectrogram

    "Adaptive Whitening For Improved Real-time Audio Onset Detection"
    Dan Stowell and Mark Plumbley
    Proceedings of the International Computer Music Conference (ICMC), 2007

    """
    raise NotImplementedError("check if adaptive_whitening returns meaningful "
                              "results")
    relaxation = 10.0 ** (-6. * relaxation)
    p = np.zeros_like(spec)
    # iterate over all frames
    for f in range(len(spec)):
        if f > 0:
            p[f] = np.maximum(spec[f], floor, relaxation * p[f - 1])
        else:
            p[f] = np.maximum(spec[f], floor)
    # return the whitened spectrogram
    return spec / p


def statistical_spectrum_descriptors(spectrogram):
    """
    Statistical Spectrum Descriptors of the STFT.

    :param spectrogram: magnitude spectrogram [numpy array]
    :return:            statistical spectrum descriptors of the spectrogram

    "Evaluation of Feature Extractors and Psycho-acoustic Transformations
     for Music Genre Classification."
    Thomas Lidy and Andreas Rauber
    Proceedings of the 6th International Conference on Music Information
    Retrieval (ISMIR), 2005

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

    :param spectrogram:     magnitude spectrogram [numpy array]
    :param bin_frequencies: frequencies of the spectrogram bins [numpy array]
    :param num_hist_bins:   number of histogram bins
    :param fref:            reference tuning frequency [Hz]
    :return:                tuning frequency

    To determine the tuning frequency, a weighted histogram of relative
    deviations of the spectrogram bins towards the closest semitones is built.

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
class Spectrogram(PropertyMixin, np.ndarray):
    """
    Spectrogram class.

    """

    def __init__(self, stft, **kwargs):
        """
        Creates a new Spectrogram instance from the given
        ShortTimeFourierTransform.

        :param stft:   ShortTimeFourierTransform instance (or anything a
                       ShortTimeFourierTransform can be instantiated from)

        If no ShortTimeFourierTransform instance was given, one is instantiated
        and these arguments are passed:

        :param kwargs: keyword arguments passed to ShortTimeFourierTransform

        """
        # this method exists only for argument documentation purposes
        # the initialisation is done in __new__() and __array_finalize__()

    def __new__(cls, stft, **kwargs):
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
        obj.frames = stft.frames
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.stft = getattr(obj, 'stft', None)
        self.frames = getattr(obj, 'frames', None)
        self.filterbank = getattr(obj, 'filterbank', None)
        self.mul = getattr(obj, 'mul', None)
        self.add = getattr(obj, 'add', None)

    def diff(self, **kwargs):
        """
        Compute the difference of the magnitude spectrogram.

        :param kwargs: keyword arguments passed to SpectrogramDifference
        :return:       SpectrogramDifference instance

        """
        return SpectrogramDifference(self, **kwargs)

    def filter(self, **kwargs):
        """
        Compute a filtered version of the magnitude spectrogram.

        :param kwargs: keyword arguments passed to FilteredSpectrogram
        :return:       FilteredSpectrogram instance

        """
        return FilteredSpectrogram(self, **kwargs)

    def log(self, **kwargs):
        """
        Compute a logarithmically scaled version of the magnitude spectrogram.

        :param kwargs: keyword arguments passed to LogarithmicSpectrogram
        :return:       LogarithmicSpectrogram instance

        """
        return LogarithmicSpectrogram(self, **kwargs)

    def tuning_frequency(self, **kwargs):
        """
        Determines the tuning frequency of the audio signal based on peaks
        of the spectrogram.

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
        """
        Creates a new SpectrogramProcessor instance.

        """
        pass

    def process(self, data, **kwargs):
        """
        Create a Spectrogram from the given data.

        :param data:   data to be processed
        :param kwargs: keyword arguments passed to Spectrogram
        :return:       Spectrogram instance

        """
        return Spectrogram(data, **kwargs)

    add_arguments = ShortTimeFourierTransformProcessor.add_arguments


# filtered spectrogram stuff
FILTERBANK = LogarithmicFilterbank


class FilteredSpectrogram(Spectrogram):
    """
    FilteredSpectrogram class.

    """
    # we just want to inherit some properties from Spectrogram

    def __init__(self, spectrogram, filterbank=FILTERBANK, num_bands=NUM_BANDS,
                 fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS, block_size=2048, **kwargs):
        """
        Creates a new FilteredSpectrogram instance from the given Spectrogram.

        :param spectrogram:    Spectrogram instance (or anything a Spectrogram
                               can be instantiated from)

        Filterbank parameters:

        :param filterbank:     Filterbank type or instance [Filterbank]

        If a Filterbank type is given rather than a Filterbank instance, one
        will be created with the given type and these parameters:

        :param num_bands:      number of filter bands (per octave, depending
                               on the type of the filterbank)
        :param fmin:           the minimum frequency [Hz, float]
        :param fmax:           the maximum frequency [Hz, float]
        :param fref:           tuning frequency [Hz, float]
        :param norm_filters:   normalize the filter to area 1 [bool]
        :param unique_filters: keep only unique filters, i.e. remove duplicate
                               filters resulting from insufficient resolution
                               at low frequencies [bool]

        Other filtering options:

        :param block_size:     perform filtering in blocks of this size
                               [int, power of 2]

        If no Spectrogram instance was given, one is instantiated and
        these arguments are passed:

        :param kwargs:         keyword arguments passed to Spectrogram

        """
        # this method exists only for argument documentation purposes
        # the initialisation is done in __new__() and __array_finalize__()

    def __new__(cls, spectrogram, filterbank=FILTERBANK, num_bands=NUM_BANDS,
                fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                unique_filters=UNIQUE_FILTERS, block_size=2048, **kwargs):
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

        # TODO: reactivate this or move this whole block/batch processing to
        #       the processors?
        # # init the return matrix
        # num_frames = spectrogram.num_frames
        # data = np.empty((num_frames, filterbank.num_bands), np.float32)
        # # process in blocks of this size
        # if block_size is None:
        #     block_size = spectrogram.num_frames
        # # iterate over the STFT in blocks of the given size
        # for b, start in enumerate(range(0, num_frames, block_size)):
        #     # determine stop index of the block
        #     stop = min(start + block_size, num_frames)
        #     # get the block
        #     block = spectrogram[start: stop]
        #     # determine the position inside the data to be returned
        #     start = b * block_size
        #     stop = start + len(block)
        #     # filter it and put it in the return spectrogram
        #     data[start: stop] = np.dot(block, filterbank)

        # filter the spectrogram
        data = np.dot(spectrogram, filterbank)
        # cast as FilteredSpectrogram
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.filterbank = filterbank
        # and those from the given spectrogram
        obj.stft = spectrogram.stft
        obj.frames = spectrogram.stft.frames
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

    def __reduce__(self):
        # needed for correct pickling
        # source: http://stackoverflow.com/questions/26598109/
        # get the parent's __reduce__ tuple
        pickled_state = super(FilteredSpectrogram, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.filterbank,)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for correct un-pickling
        # set the attributes
        self.filterbank = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(FilteredSpectrogram, self).__setstate__(state[0:-1])


class FilteredSpectrogramProcessor(Processor):
    """
    FilteredSpectrogramProcessor class.

    """

    def __init__(self, filterbank=FILTERBANK, num_bands=NUM_BANDS, fmin=FMIN,
                 fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS, **kwargs):
        """
        Creates a new FilteredSpectrogramProcessor instance.

        Magnitude spectrogram filtering parameters:

        :param filterbank:     filter the magnitude spectrogram with a
                               filterbank of this type [None or Filterbank]
        :param num_bands:      number of filter bands (per octave) [int]
        :param fmin:           minimum frequency of the filterbank [Hz, float]
        :param fmax:           maximum frequency of the filterbank [Hz, float]
        :param fref:           tuning frequency [Hz, float]
        :param norm_filters:   normalize the filter to area 1 [bool]
        :param unique_filters: keep only unique filters, i.e. remove duplicate
                               filters resulting from insufficient resolution
                               at low frequencies [bool]

        """
        self.filterbank = filterbank
        self.num_bands = num_bands
        self.fmin = fmin
        self.fmax = fmax
        self.fref = fref
        self.norm_filters = norm_filters
        self.unique_filters = unique_filters

    def process(self, data, **kwargs):
        """
        Perform filtering of a spectrogram.

        :param data:   data to be processed
        :param kwargs: keyword arguments passed to FilteredSpectrogram
        :return:       FilteredSpectrogram instance

        """
        # instantiate a FilteredSpectrogram and return it
        return FilteredSpectrogram(data, filterbank=self.filterbank,
                                   num_bands=self.num_bands, fmin=self.fmin,
                                   fmax=self.fmax, fref=self.fref,
                                   norm_filters=self.norm_filters,
                                   unique_filters=self.unique_filters,
                                   **kwargs)

    @classmethod
    def add_arguments(cls, parser, filterbank=FILTERBANK, num_bands=NUM_BANDS,
                      fmin=FMIN, fmax=FMAX, norm_filters=NORM_FILTERS,
                      unique_filters=UNIQUE_FILTERS):
        """
        Add spectrogram filtering related arguments to an existing parser.

        :param parser:         existing argparse parser
        :param filterbank:     filter the magnitude spectrogram with a
                               filterbank of that type [Filterbank]
        :param num_bands:      number of filter bands (per octave) [int]
        :param fmin:           minimum frequency of the filterbank [Hz, float]
        :param fmax:           maximum frequency of the filterbank [Hz, float]
        :param norm_filters:   normalize the filters to area 1 [bool]
        :param unique_filters: keep only unique filters, i.e. remove duplicate
                               filters resulting from insufficient resolution
                               at low frequencies [bool]
        :return:               spectrogram filtering argument parser group

        Parameters are included in the group only if they are not 'None'.

        """
        from .filters import Filterbank
        # add filterbank related options to the existing parser
        g = parser.add_argument_group('spectrogram filtering arguments')
        # filterbank
        if issubclass(filterbank, Filterbank):
            g.add_argument('--no_filter', dest='filterbank',
                           action='store_false',
                           default=filterbank,
                           help='do not filter the spectrogram with a '
                                'filterbank [default=%(default)s]')
        elif filterbank is not None:
            # TODO: add filterbank option list?
            g.add_argument('--filter', action='store_true', default=None,
                           help='filter the spectrogram with a filterbank '
                                'of this type')
        # number of bands
        if num_bands is not None:
            g.add_argument('--num_bands', action='store', type=int,
                           default=num_bands,
                           help='number of filter bands (per octave) '
                                '[default=%(default)i]')
        # minimum frequency
        if fmin is not None:
            g.add_argument('--fmin', action='store', type=float,
                           default=fmin,
                           help='minimum frequency of the filterbank '
                                '[Hz, default=%(default).1f]')
        # maximum frequency
        if fmax is not None:
            g.add_argument('--fmax', action='store', type=float,
                           default=fmax,
                           help='maximum frequency of the filterbank '
                                '[Hz, default=%(default).1f]')
        # normalize filters
        if norm_filters is True:
            g.add_argument('--no_norm_filters', dest='norm_filters',
                           action='store_false', default=norm_filters,
                           help='do not normalize the filters to area 1 '
                                '[default=True]')
        elif norm_filters is False:
            g.add_argument('--norm_filters', dest='norm_filters',
                           action='store_true', default=norm_filters,
                           help='normalize the filters to area 1 '
                                '[default=False]')
        # unique or duplicate filters
        if unique_filters is True:
            # add option to keep the duplicate filters
            g.add_argument('--duplicate_filters', dest='unique_filters',
                           action='store_false', default=unique_filters,
                           help='keep duplicate filters resulting from '
                                'insufficient resolution at low frequencies '
                                '[default=only unique filters are kept]')
        elif unique_filters is False:
            g.add_argument('--unique_filters', action='store_true',
                           default=unique_filters,
                           help='keep only unique filters, i.e. remove '
                                'duplicate filters resulting from '
                                'insufficient resolution at low frequencies '
                                '[default=duplicate filters are kept]')
        # return the group
        return g


# logarithmic spectrogram stuff
LOG = True
MUL = 1.
ADD = 1.


class LogarithmicSpectrogram(Spectrogram):
    """
    LogarithmicSpectrogram class.

    """
    # we just want to inherit some properties from Spectrogram

    def __init__(self, spectrogram, mul=MUL, add=ADD, **kwargs):
        """
        Creates a new LogarithmicSpectrogram instance from the given
        Spectrogram.

        :param spectrogram: Spectrogram instance (or anything a Spectrogram
                            can be instantiated from)

        Logarithmic magnitude parameters:

        :param mul:         multiply the magnitude spectrogram with this factor
                            before taking the logarithm [float]
        :param add:         add this value before taking the logarithm of the
                            magnitudes [float]

        If no Spectrogram instance was given, one is instantiated and these
        arguments are passed:

        :param kwargs:      keyword arguments passed to Spectrogram

        """
        # this method exists only for argument documentation purposes
        # the initialisation is done in __new__() and __array_finalize__()

    def __new__(cls, spectrogram, mul=MUL, add=ADD, **kwargs):
        # instantiate a Spectrogram if needed
        if not isinstance(spectrogram, Spectrogram):
            # try to instantiate a Spectrogram object
            spectrogram = Spectrogram(spectrogram, **kwargs)

        # filter the spectrogram
        data = np.log10(mul * spectrogram + add)
        # cast as FilteredSpectrogram
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.mul = mul
        obj.add = add
        # and those from the given spectrogram
        obj.stft = spectrogram.stft
        obj.frames = spectrogram.stft.frames
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

    def __reduce__(self):
        # needed for correct pickling
        # source: http://stackoverflow.com/questions/26598109/
        # get the parent's __reduce__ tuple
        pickled_state = super(LogarithmicSpectrogram, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.mul, self.add,)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for correct un-pickling
        # set the attributes
        self.mul = state[-2]
        self.add = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(LogarithmicSpectrogram, self).__setstate__(state[0:-2])


class LogarithmicSpectrogramProcessor(Processor):
    """
    Logarithmic Spectrogram Processor class.

    """

    def __init__(self, mul=MUL, add=ADD, **kwargs):
        """
        Creates a new LogarithmicSpectrogramProcessor instance.

        Magnitude spectrogram scaling parameters:

        :param mul: multiply the spectrogram with this factor before taking
                    the logarithm of the magnitudes [float]
        :param add: add this value before taking the logarithm of the
                    magnitudes [float]

        """
        self.mul = mul
        self.add = add

    def process(self, data, **kwargs):
        """
        Perform logarithmic scaling of a spectrogram.

        :param data:   data to be processed
        :param kwargs: keyword arguments passed to LogarithmicSpectrogram
        :return:       LogarithmicSpectrogram instance

        """
        # instantiate a LogarithmicSpectrogram
        return LogarithmicSpectrogram(data, mul=self.mul, add=self.add,
                                      **kwargs)

    @classmethod
    def add_arguments(cls, parser, log=None, mul=None, add=None):
        """
        Add spectrogram scaling related arguments to an existing parser.

        :param parser: existing argparse parser
        :param log:    take the logarithm of the spectrogram [bool]
        :param mul:    multiply the spectrogram with this factor before
                       taking the logarithm of the magnitudes [float]
        :param add:    add this value before taking the logarithm of the
                       magnitudes [float]
        :return:       spectrogram scaling argument parser group

        Parameters are included in the group only if they are not 'None'.

        """
        # add log related options to the existing parser
        g = parser.add_argument_group('magnitude scaling arguments')
        # log
        if log is True:
            g.add_argument('--linear', dest='log', action='store_false',
                           help='linear magnitudes [default=logarithmic]')
        elif log is False:
            g.add_argument('--log', action='store_true',
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

    """

    def __init__(self, spectrogram, **kwargs):
        """
        Creates a new LogarithmicFilteredSpectrogram instance of the given
        FilteredSpectrogram.

        :param spectrogram: FilteredSpectrogram instance (or anything a
                            FilteredSpectrogram can be instantiated from)

        If no FilteredSpectrogram instance was given, one is instantiated and
        logarithmically scaled afterwards. These arguments are passed:

        :param kwargs:      keyword arguments passed to FilteredSpectrogram and
                            LogarithmicSpectrogram

        """
        # this method exists only for argument documentation purposes
        # the initialisation is done in __new__() and __array_finalize__()

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
        obj.filterbank = spectrogram.filterbank
        obj.mul = data.mul
        obj.add = data.add
        # and those from the given spectrogram
        obj.stft = spectrogram.stft
        obj.frames = spectrogram.stft.frames
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

    """

    def __init__(self, filterbank=FILTERBANK, num_bands=NUM_BANDS, fmin=FMIN,
                 fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS, mul=MUL, add=ADD, **kwargs):
        """
        Creates a new LogarithmicFilteredSpectrogramProcessor instance.

        Magnitude spectrogram filtering parameters:

        :param filterbank:     filter the magnitude spectrogram with a
                               filterbank of this type [None or Filterbank]
        :param num_bands:      number of filter bands (per octave) [int]
        :param fmin:           minimum frequency of the filterbank [Hz, float]
        :param fmax:           maximum frequency of the filterbank [Hz, float]
        :param fref:           tuning frequency [Hz, float]
        :param norm_filters:   normalize the filter to area 1 [bool]
        :param unique_filters: keep only unique filters, i.e. remove duplicate
                               filters resulting from insufficient resolution
                               at low frequencies [bool]

        Magnitude spectrogram scaling parameters:

        :param mul:            multiply the spectrogram with this factor before
                               taking the logarithm of the magnitudes [float]
        :param add:            add this value before taking the logarithm of
                               the magnitudes [float]

        """
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
        Perform logarithmic scaling of a filtered spectrogram.

        :param data:   data to be processed
        :param kwargs: keyword arguments passed to
                       LogarithmicFilteredSpectrogram
        :return:       LogarithmicFilteredSpectrogram instance

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

    """
    # we just want to inherit some properties from Spectrogram

    def __init__(self, spectrogram, diff_ratio=DIFF_RATIO,
                 diff_frames=DIFF_FRAMES, diff_max_bins=DIFF_MAX_BINS,
                 positive_diffs=POSITIVE_DIFFS, **kwargs):
        """
        Creates a new SpectrogramDifference instance from the given
        spectrogram.

        :param spectrogram:    Spectrogram instance (or anything a Spectrogram
                               can be instantiated from)

        Difference parameters:

        :param diff_ratio:     calculate the difference to the frame at which
                               the window used for the STFT yields this ratio
                               of the maximum height [float]
        :param diff_frames:    calculate the difference to the N-th previous
                               frame (if set, this overrides the value
                               calculated from the `diff_ratio`) [int]
        :param diff_max_bins:  apply a maximum filter with this width (in bins
                               in frequency dimension) [int]
        :param positive_diffs: keep only the positive differences, i.e. set
                               all diff values < 0 to 0. [bool]

        If no Spectrogram instance was given, one is instantiated and these
        arguments are passed:

        :param kwargs:         keyword arguments passed to Spectrogram

        Note: The SuperFlux algorithm uses a maximum filtered spectrogram with
              3 `max_bins` together with a 24 band logarithmic filterbank to
              calculate the difference spectrogram.

        """
        # this method exists only for argument documentation purposes
        # the initialisation is done in __new__() and __array_finalize__()

    def __new__(cls, spectrogram, diff_ratio=DIFF_RATIO,
                diff_frames=DIFF_FRAMES, diff_max_bins=DIFF_MAX_BINS,
                positive_diffs=POSITIVE_DIFFS, **kwargs):
        # instantiate a Spectrogram if needed
        if not isinstance(spectrogram, Spectrogram):
            # try to instantiate a Spectrogram object
            spectrogram = Spectrogram(spectrogram, **kwargs)

        # calculate the number of diff frames to use
        if not diff_frames:
            # calculate the number of diff_frames on basis of the diff_ratio
            # get the first sample with a higher magnitude than given ratio
            window = spectrogram.stft.window
            sample = np.argmax(window > diff_ratio * max(window))
            diff_samples = len(spectrogram.stft.window) / 2 - sample
            # convert to frames
            hop_size = spectrogram.stft.frames.hop_size
            diff_frames = int(round(diff_samples / hop_size))
        # always set the minimum to 1
        if diff_frames < 1:
            diff_frames = 1

        # init matrix
        diff = np.zeros_like(spectrogram)

        # apply a maximum filter to diff_spec if needed
        if diff_max_bins > 1:
            from scipy.ndimage.filters import maximum_filter
            # widen the spectrogram in frequency dimension
            diff_spec = maximum_filter(spectrogram, size=[1, diff_max_bins])
        else:
            diff_spec = spectrogram
        # calculate the diff
        diff[diff_frames:] = (spectrogram[diff_frames:] -
                              diff_spec[: -diff_frames])
        # positive differences only?
        if positive_diffs:
            np.maximum(diff, 0, diff)

        # cast as FilteredSpectrogram
        obj = np.asarray(diff).view(cls)
        # save additional attributes
        obj.diff_ratio = diff_ratio
        obj.diff_frames = diff_frames
        obj.diff_max_bins = diff_max_bins
        obj.positive_diffs = positive_diffs
        # and those from the given spectrogram
        obj.stft = spectrogram.stft
        obj.frames = spectrogram.stft.frames
        obj.filterbank = spectrogram.filterbank
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

    def __reduce__(self):
        # get the parent's __reduce__ tuple
        pickled_state = super(SpectrogramDifference, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.diff_ratio, self.diff_frames,
                                        self.diff_max_bins,
                                        self.positive_diffs)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # set the attributes
        self.diff_ratio = state[-4]
        self.diff_frames = state[-3]
        self.diff_max_bins = state[-2]
        self.positive_diffs = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(SpectrogramDifference, self).__setstate__(state[0:-4])

    def positive_diff(self):
        """Positive diff."""
        return np.maximum(self, 0)


class SpectrogramDifferenceProcessor(Processor):
    """
    Difference Spectrogram Processor class.

    """

    def __init__(self, diff_ratio=DIFF_RATIO, diff_frames=DIFF_FRAMES,
                 diff_max_bins=DIFF_MAX_BINS, positive_diffs=POSITIVE_DIFFS,
                 **kwargs):
        """
        Spectrogram difference parameters:

        :param diff_ratio:     calculate the difference to the frame at which
                               the window used for the STFT yields this ratio
                               of the maximum height [float]
        :param diff_frames:    calculate the difference to the N-th previous
                               frame [int] (if set, this overrides the value
                               calculated from the `diff_ratio`)
        :param diff_max_bins:  apply a maximum filter with this width (in bins
                               in frequency dimension) [int]
        :param positive_diffs: keep only the positive differences, i.e. set all
                               diff values < 0 to 0

        """
        self.diff_ratio = diff_ratio
        self.diff_frames = diff_frames
        self.diff_max_bins = diff_max_bins
        self.positive_diffs = positive_diffs

    def process(self, data, **kwargs):
        """
        Perform a temporal difference calculation on the given data.

        :param data:   data to calculate the difference on
        :param kwargs: keyword arguments passed to SpectrogramDifference
        :return:       SpectrogramDifference instance

        """
        # instantiate a SpectrogramDifference and return it
        return SpectrogramDifference(data, diff_ratio=self.diff_ratio,
                                     diff_frames=self.diff_frames,
                                     diff_max_bins=self.diff_max_bins,
                                     positive_diffs=self.positive_diffs,
                                     **kwargs)

    @classmethod
    def add_arguments(cls, parser, diff=None, diff_ratio=None,
                      diff_frames=None, diff_max_bins=None,
                      positive_diffs=None):
        """
        Add spectrogram difference related arguments to an existing parser.

        :param parser:         existing argparse parser
        :param diff:           take the difference of the spectrogram [bool]
        :param diff_ratio:     calculate the difference to the frame at which
                               the window used for the STFT yields this ratio
                               of the maximum height [float]
        :param diff_frames:    calculate the difference to the N-th previous
                               frame [int] (if set, this overrides the value
                               calculated from the `diff_ratio`)
        :param diff_max_bins:  apply a maximum filter with this width (in bins
                               in frequency dimension) [int]
        :param positive_diffs: keep only the positive differences, i.e. set all
                               diff values < 0 to 0
        :return:               spectrogram difference argument parser group

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

    def __init__(self, **kwargs):
        """
        Creates a new SuperFluxProcessor instance.

        """
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

    """

    def __init__(self, spectrogram, crossover_frequencies, norm_bands=False,
                 **kwargs):
        """
        Creates a new MultiBandSpectrogram instance from the given
        Spectrogram.

        :param spectrogram:           Spectrogram instance (or anything a
                                      Spectrogram can be instantiated from)

        Multi-band parameters:

        :param crossover_frequencies: list of crossover frequencies at which
                                      the spectrogram is split into bands
        :param norm_bands:            normalize the bands [bool]

        If no Spectrogram instance was given, a FilteredSpectrogram is
        instantiated and these arguments are passed:

        :param kwargs:                keyword arguments passed to
                                      FilteredSpectrogram

        """
        # this method exists only for argument documentation purposes
        # the initialisation is done in __new__() and __array_finalize__()

    def __new__(cls, spectrogram, crossover_frequencies, norm_bands=False,
                **kwargs):
        from .filters import RectangularFilterbank
        # instantiate a FilteredSpectrogram if needed
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = FilteredSpectrogram(spectrogram, **kwargs)
        # create a rectangular filterbank
        filterbank = RectangularFilterbank(spectrogram.bin_frequencies,
                                           crossover_frequencies,
                                           norm_filters=norm_bands)
        # filter the spectrogram
        data = np.dot(spectrogram, filterbank)
        # cast as FilteredSpectrogram
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.filterbank = filterbank
        obj.crossover_frequencies = crossover_frequencies
        obj.norm_bands = norm_bands
        obj.stft = spectrogram.stft
        obj.frames = spectrogram.stft.frames
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here, also needed for views
        self.crossover_frequencies = getattr(obj, 'crossover_frequencies',
                                             None)
        self.norm_bands = getattr(obj, 'norm_bands', False)
        self.filterbank = getattr(obj, 'norm_bands', None)
        self.stft = getattr(obj, 'stft', None)
        self.frames = getattr(obj, 'frames', None)

    def __reduce__(self):
        # needed for correct pickling
        # source: http://stackoverflow.com/questions/26598109/
        # get the parent's __reduce__ tuple
        pickled_state = super(FilteredSpectrogram, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.filterbank,
                                        self.crossover_frequencies,
                                        self.norm_bands)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for correct un-pickling
        # set the attributes
        self.filterbank = state[-3]
        self.crossover_frequencies = state[-2]
        self.norm_bands = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(FilteredSpectrogram, self).__setstate__(state[0:-3])

    @property
    def bin_frequencies(self):
        """Frequencies of the spectrogram bins."""
        # overwrite with the filterbank center frequencies
        return self.filterbank.center_frequencies


class MultiBandSpectrogramProcessor(Processor):
    """
    Spectrogram processor which combines the differences of a log filtered
    spectrogram into multiple bands.

    """

    def __init__(self, crossover_frequencies, norm_bands=False, **kwargs):
        """

        :param crossover_frequencies: list of crossover frequencies at which
                                      the spectrogram is split into bands
        :param norm_bands:            normalize the bands [bool]

        """
        self.crossover_frequencies = crossover_frequencies
        self.norm_bands = norm_bands

    def process(self, data, **kwargs):
        """
        Return the a multi-band representation of the given spectrogram.

        :param data:   spectrogram to be processed [Spectrogram]
        :param kwargs: keyword arguments passed to MultiBandSpectrogram
        :return:       MultiBandSpectrogram instance

        """
        # instantiate a MultiBandSpectrogram
        return MultiBandSpectrogram(
            data, crossover_frequencies=self.crossover_frequencies,
            norm_bands=self.norm_bands, **kwargs)

    @classmethod
    def add_arguments(cls, parser, crossover_frequencies=None,
                      norm_bands=None):
        """
        Add multi-band spectrogram related arguments to an existing parser.

        :param parser:                existing argparse parser
        :param crossover_frequencies: list with crossover frequencies
        :param norm_bands:            normalize the bands
        :return:                      multi-band argument parser group

        Parameters are included in the group only if they are not 'None'.

        """
        # add filterbank related options to the existing parser
        g = parser.add_argument_group('multi-band spectrogram arguments')
        # crossover frequencies
        if crossover_frequencies is not None:
            from madmom.utils import OverrideDefaultListAction
            g.add_argument('--crossover_frequencies', type=float, sep=',',
                           action=OverrideDefaultListAction,
                           default=crossover_frequencies,
                           help='(comma separated) list with crossover '
                                'frequencies [Hz, default=%(default)s]')
        # normalization of bands
        if norm_bands is not None:
            if norm_bands:
                g.add_argument('--no_norm_bands', dest='norm_bands',
                               action='store_false', default=norm_bands,
                               help='no not normalize the bands')
            else:
                g.add_argument('--norm_bands', action='store_true',
                               default=-norm_bands,
                               help='normalize the bands')
        # return the group
        return g


class StackedSpectrogramProcessor(ParallelProcessor):
    """
    Class to stack multiple spectrograms (and their differences) in a certain
    dimension.

    """
    # Note: `frame_size` is used instead of the more meaningful `frame_sizes`,
    #       this way the existing argument from `FramedSignal` can be reused
    def __init__(self, frame_size, spectrogram, difference=None,
                 stack=np.hstack, **kwargs):
        """
        Creates a new StackedSpectrogramProcessor instance.

        :param frame_size:  list with frame sizes [list of int]
        :param spectrogram: SpectrogramProcessor instance
        :param difference:  SpectrogramDifferenceProcessor instance; if given
                            the differences of the spectrogram(s) are stacked
                            as well
        :param stack:       stacking function to be used
                            - 'np.vstack' stack multiple spectrograms
                            vertically, i.e. stack in time dimension
                            - 'np.hstack' stack multiple spectrograms
                              horizontally, i.e. stack in the frequency
                              dimension
                            - 'np.dstack' stacks them in depth, i.e.
                              returns them as a 3D representation
                            Additionally, the literal values {'time',
                            'freq' | 'frequency', 'depth'} are supported

        Note: To be able to stack spectrograms in depth (i.e. use 'np.dstack'
              as a stacking function), they must have the same frequency
              dimensionality. If filtered spectrograms are used,
              `unique_filters` must be set to 'False'.

        """
        from .signal import FramedSignalProcessor
        # use the same spectrogram processor for all frame sizes, but use
        # different FramedSignal processors
        processors = []
        for frame_size_ in frame_size:
            fs = FramedSignalProcessor(frame_size=frame_size_, **kwargs)
            processors.append([fs, spectrogram])
        # FIXME: works only with a single thread
        super(StackedSpectrogramProcessor, self).__init__(processors,
                                                          num_threads=1)
        # literal stacking directions
        if stack == 'time':
            stack = np.vstack
        elif stack in ('freq', 'frequency'):
            stack = np.hstack
        elif stack == 'depth':
            stack = np.dstack
        self.stack = stack
        # TODO: it is a bit hackish to define another processor here
        self.diff_processor = difference

    def process(self, data):
        """
        Stack the magnitudes spectrograms (and their differences).

        :param data: Signal instance [Signal]
        :return:     stacked specs (and diffs) [numpy array]

        """
        # process everything
        specs = super(StackedSpectrogramProcessor, self).process(data)
        # stack everything (a list of Spectrogram instances was returned)
        stack = []
        for s in specs:
            # always append the spec
            stack.append(s)
            # and the differences only if needed
            if self.diff_processor is not None:
                diffs = self.diff_processor.process(s)
                stack.append(diffs)
        # stack them along given axis and return them
        return self.stack(stack)

    @classmethod
    def add_arguments(cls, parser, stack='freq', stack_diffs=None):
        """
        Add stacking related arguments to an existing parser.

        :param parser:      existing argparse parser
        :param stack:       stacking direction {'time', 'freq', 'depth'}
        :param stack_diffs: also stack the differences [bool]
        :return:            stacking argument parser group


        """
        # add diff related options to the existing parser
        g = parser.add_argument_group('stacking arguments')
        # stacking axis
        if stack is not None:
            g.add_argument('--stack', action='store', type=str,
                           default=stack, choices=['time', 'freq', 'depth'],
                           help="stacking direction [default=%(default)s]")
        # stack diffs?
        if stack_diffs is True:
            g.add_argument('--no_stack_diffs', dest='stack_diffs',
                           action='store_false',
                           help='no not stack the differences of the '
                                'spectrograms')
        elif stack_diffs is False:
            g.add_argument('--stack_diffs', action='store_true',
                           help='in addition to the spectrograms, also stack '
                                'their differences')
        # return the group
        return g
