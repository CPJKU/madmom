# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains filter and filterbank related functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np
from ..processors import Processor

FILTER_DTYPE = np.float32
A4 = 440.


# Mel frequency scale
def hz2mel(f):
    """
    Convert Hz frequencies to Mel.

    Parameters
    ----------
    f : numpy array
        Input frequencies [Hz].

    Returns
    -------
    m : numpy array
        Frequencies in Mel [Mel].

    """
    return 1127.01048 * np.log(np.asarray(f) / 700. + 1.)


def mel2hz(m):
    """
    Convert Mel frequencies to Hz.

    Parameters
    ----------
    m : numpy array
        Input frequencies [Mel].

    Returns
    -------
    f: numpy array
        Frequencies in Hz [Hz].

    """
    return 700. * (np.exp(np.asarray(m) / 1127.01048) - 1.)


def mel_frequencies(num_bands, fmin, fmax):
    """
    Returns frequencies aligned on the Mel scale.

    Parameters
    ----------
    num_bands : int
        Number of bands.
    fmin : float
        Minimum frequency [Hz].
    fmax : float
        Maximum frequency [Hz].

    Returns
    -------
    mel_frequencies: numpy array
        Frequencies with Mel spacing [Hz].

    """
    # convert fmin and fmax to the Mel scale and return an array of frequencies
    return mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), num_bands))


# logarithmic frequency scale
def log_frequencies(bands_per_octave, fmin, fmax, fref=A4):
    """
    Returns frequencies aligned on a logarithmic frequency scale.

    Parameters
    ----------
    bands_per_octave : int
        Number of filter bands per octave.
    fmin : float
        Minimum frequency [Hz].
    fmax : float
        Maximum frequency [Hz].
    fref : float, optional
        Tuning frequency [Hz].

    Returns
    -------
    log_frequencies : numpy array
        Logarithmically spaced frequencies [Hz].

    Notes
    -----
    If `bands_per_octave` = 12 and `fref` = 440 are used, the frequencies are
    equivalent to MIDI notes.

    """
    # get the range
    left = np.floor(np.log2(float(fmin) / fref) * bands_per_octave)
    right = np.ceil(np.log2(float(fmax) / fref) * bands_per_octave)
    # generate frequencies
    frequencies = fref * 2. ** (np.arange(left, right) /
                                float(bands_per_octave))
    # filter frequencies
    # needed, because range might be bigger because of the use of floor/ceil
    frequencies = frequencies[np.searchsorted(frequencies, fmin):]
    frequencies = frequencies[:np.searchsorted(frequencies, fmax, 'right')]
    # return
    return frequencies


def semitone_frequencies(fmin, fmax, fref=A4):
    """
    Returns frequencies separated by semitones.

    Parameters
    ----------
    fmin : float
        Minimum frequency [Hz].
    fmax : float
        Maximum frequency [Hz].
    fref : float, optional
        Tuning frequency of A4 [Hz].

    Returns
    -------
    semitone_frequencies : numpy array
        Semitone frequencies [Hz].

    """
    # return MIDI frequencies
    return log_frequencies(12, fmin, fmax, fref)


# MIDI
def hz2midi(f, fref=A4):
    """
    Convert frequencies to the corresponding MIDI notes.

    Parameters
    ----------
    f : numpy array
        Input frequencies [Hz].
    fref : float, optional
        Tuning frequency of A4 [Hz].

    Returns
    -------
    m : numpy array
        MIDI notes

    Notes
    -----
    For details see: at http://www.phys.unsw.edu.au/jw/notes.html
    This function does not necessarily return a valid MIDI Note, you may need
    to round it to the nearest integer.

    """
    return (12. * np.log2(np.asarray(f, dtype=np.float) / fref)) + 69.


def midi2hz(m, fref=A4):
    """
    Convert MIDI notes to corresponding frequencies.

    Parameters
    ----------
    m : numpy array
        Input MIDI notes.
    fref : float, optional
        Tuning frequency of A4 [Hz].

    Returns
    -------
    f : numpy array
        Corresponding frequencies [Hz].

    """
    return 2. ** ((np.asarray(m, dtype=np.float) - 69.) / 12.) * fref


# provide an alias to semitone_frequencies
midi_frequencies = semitone_frequencies


# ERB frequency scale
def hz2erb(f):
    """
    Convert Hz to ERB.

    Parameters
    ----------
    f : numpy array
        Input frequencies [Hz].

    Returns
    -------
    e : numpy array
        Frequencies in ERB [ERB].

    Notes
    -----
    Information about the ERB scale can be found at:
    https://ccrma.stanford.edu/~jos/bbt/Equivalent_Rectangular_Bandwidth.html

    """
    return 21.4 * np.log10(1 + 4.37 * np.asarray(f) / 1000.)


def erb2hz(e):
    """
    Convert ERB scaled frequencies to Hz.

    Parameters
    ----------
    e : numpy array
        Input frequencies [ERB].

    Returns
    -------
    f : numpy array
        Frequencies in Hz [Hz].

    Notes
    -----
    Information about the ERB scale can be found at:
    https://ccrma.stanford.edu/~jos/bbt/Equivalent_Rectangular_Bandwidth.html

    """
    return (10. ** (np.asarray(e) / 21.4) - 1.) * 1000. / 4.37


# helper functions for filter creation
def frequencies2bins(frequencies, bin_frequencies, unique_bins=False):
    """
    Map frequencies to the closest corresponding bins.

    Parameters
    ----------
    frequencies : numpy array
        Input frequencies [Hz].
    bin_frequencies : numpy array
        Frequencies of the (FFT) bins [Hz].
    unique_bins : bool, optional
        Return only unique bins, i.e. remove all duplicate bins resulting from
        insufficient resolution at low frequencies.

    Returns
    -------
    bins : numpy array
        Corresponding (unique) bins.

    Notes
    -----
    It can be important to return only unique bins, otherwise the lower
    frequency bins can be given too much weight if all bins are simply summed
    up (as in the spectral flux onset detection).

    """
    # cast as numpy arrays
    frequencies = np.asarray(frequencies)
    bin_frequencies = np.asarray(bin_frequencies)
    # map the frequencies to the closest bins
    # solution found at: http://stackoverflow.com/questions/8914491/
    indices = bin_frequencies.searchsorted(frequencies)
    indices = np.clip(indices, 1, len(bin_frequencies) - 1)
    left = bin_frequencies[indices - 1]
    right = bin_frequencies[indices]
    indices -= frequencies - left < right - frequencies
    # only keep unique bins if requested
    if unique_bins:
        indices = np.unique(indices)
    # return the (unique) bin indices of the closest matches
    return indices


def bins2frequencies(bins, bin_frequencies):
    """
    Convert bins to the corresponding frequencies.

    Parameters
    ----------
    bins : numpy array
        Bins (e.g. FFT bins).
    bin_frequencies : numpy array
        Frequencies of the (FFT) bins [Hz].

    Returns
    -------
    f : numpy array
        Corresponding frequencies [Hz].

    """
    # map the frequencies to spectrogram bins
    return np.asarray(bin_frequencies, dtype=np.float)[np.asarray(bins)]


# filter classes
class Filter(np.ndarray):
    """
    Generic Filter class.

    Parameters
    ----------
    data : 1D numpy array
        Filter data.
    start : int, optional
        Start position (see notes).
    norm : bool, optional
        Normalize the filter area to 1.

    Notes
    -----
    The start position is mandatory if a Filter should be used for the creation
    of a Filterbank.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, data, start=0, norm=False):
        # this method is for documentation purposes only
        pass

    def __new__(cls, data, start=0, norm=False):
        # input is an numpy ndarray instance
        if isinstance(data, np.ndarray):
            # cast as Filter
            obj = np.asarray(data, dtype=FILTER_DTYPE).view(cls)
        else:
            raise TypeError('wrong input data for Filter, must be np.ndarray')
        # right now, allow only 1D
        if obj.ndim != 1:
            raise NotImplementedError('please add multi-dimension support')
        # normalize
        if norm:
            obj /= np.sum(obj)
        # set attributes
        obj.start = int(start)
        obj.stop = int(start + len(data))
        # return the object
        return obj

    @classmethod
    def band_bins(cls, bins, **kwargs):
        """
        Must yield the center/crossover bins needed for filter creation.

        Parameters
        ----------
        bins : numpy array
            Center/crossover bins used for the creation of filters.
        kwargs : dict, optional
            Additional parameters for for the creation of filters
            (e.g. if the filters should overlap or not).

        """
        raise NotImplementedError('needs to be implemented by sub-classes')

    @classmethod
    def filters(cls, bins, norm, **kwargs):
        """
        Create a list with filters for the given bins.

        Parameters
        ----------
        bins : list or numpy array
            Center/crossover bins of the filters.
        norm : bool
            Normalize the area of the filter(s) to 1.
        kwargs : dict, optional
            Additional parameters passed to :func:`band_bins`
            (e.g. if the filters should overlap or not).

        Returns
        -------
        filters : list
            Filter(s) for the given bins.

        """
        # generate a list of filters for the given center/crossover bins
        filters = []
        for filter_args in cls.band_bins(bins, **kwargs):
            # create a filter and append it to the list
            filters.append(cls(*filter_args, norm=norm))
        # return the filters
        return filters


class TriangularFilter(Filter):
    """
    Triangular filter class.

    Create a triangular shaped filter with length `stop`, height 1 (unless
    normalized) with indices <= `start` set to 0.

    Parameters
    ----------
    start : int
        Start bin of the filter.
    center : int
        Center bin of the filter.
    stop : int
        Stop bin of the filter.
    norm : bool, optional
        Normalize the area of the filter to 1.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, start, center, stop, norm=False):
        # this method is for documentation purposes only
        pass

    def __new__(cls, start, center, stop, norm=False):
        # pylint: disable=arguments-differ
        # center must be between start & stop
        if not start <= center < stop:
            raise ValueError('`center` must be between `start` and `stop`')
        # cast variables to int
        center = int(center)
        start = int(start)
        stop = int(stop)
        # make center and stop relative
        center -= start
        stop -= start
        # create filter
        data = np.zeros(stop)
        # rising edge (without the center)
        data[:center] = np.linspace(0, 1, center, endpoint=False)
        # falling edge (including the center, but without the last bin)
        data[center:] = np.linspace(1, 0, stop - center, endpoint=False)
        # cast to TriangularFilter
        obj = Filter.__new__(cls, data, start, norm)
        # set the center bin
        obj.center = start + center
        # return the filter
        return obj

    @classmethod
    def band_bins(cls, bins, overlap=True):
        """
        Yields start, center and stop bins for creation of triangular filters.

        Parameters
        ----------
        bins : list or numpy array
            Center bins of filters.
        overlap : bool, optional
            Filters should overlap (see notes).

        Yields
        ------
        start : int
            Start bin of the filter.
        center : int
            Center bin of the filter.
        stop : int
            Stop bin of the filter.

        Notes
        -----
        If `overlap` is 'False', the `start` and `stop` bins of the filters
        are interpolated between the centre bins, normal rounding applies.

        """
        # pylint: disable=arguments-differ
        # make sure enough bins are given
        if len(bins) < 3:
            raise ValueError('not enough bins to create a TriangularFilter')
        # yield the bins
        index = 0
        while index + 3 <= len(bins):
            # get start, center and stop bins
            start, center, stop = bins[index: index + 3]
            # create non-overlapping filters
            if not overlap:
                # re-arrange the start and stop positions
                start = int(np.floor((center + start) / 2.))
                stop = int(np.ceil((center + stop) / 2.))
            # consistently handle too-small filters
            if stop - start < 2:
                center = start
                stop = start + 1
            # yield the bins and continue
            yield start, center, stop
            # increase counter
            index += 1


class RectangularFilter(Filter):
    """
    Rectangular filter class.

    Create a rectangular shaped filter with length `stop`, height 1 (unless
    normalized) with indices < `start` set to 0.

    Parameters
    ----------
    start : int
        Start bin of the filter.
    stop : int
        Stop bin of the filter.
    norm : bool, optional
        Normalize the area of the filter to 1.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, start, stop, norm=False):
        # this method is for documentation purposes only
        pass

    def __new__(cls, start, stop, norm=False):
        # pylint: disable=signature-differs
        # start must be smaller than stop
        if start >= stop:
            raise ValueError('`start` must be smaller than `stop`')
        # length of the filter
        length = stop - start
        # create filter
        data = np.ones(length, dtype=np.float)
        # cast to RectangularFilter and return it
        return Filter.__new__(cls, data, start, norm)

    @classmethod
    def band_bins(cls, bins, overlap=False):
        """
        Yields start and stop bins and normalization info for creation of
        rectangular filters.

        Parameters
        ----------
        bins : list or numpy array
            Crossover bins of filters.
        overlap : bool, optional
            Filters should overlap.

        Yields
        ------
        start : int
            Start bin of the filter.
        stop : int
            Stop bin of the filter.

        """
        # pylint: disable=arguments-differ
        # make sure enough bins are given
        if len(bins) < 2:
            raise ValueError('not enough bins to create a RectangularFilter')
        # overlapping filters?
        if overlap:
            raise NotImplementedError('please implement if needed!')
        # yield the bins
        index = 0
        while index + 2 <= len(bins):
            # get start and stop bins
            start, stop = bins[index: index + 2]
            # yield the bins and continue
            yield start, stop
            # increase counter
            index += 1


# default values for filter banks
FMIN = 30.
FMAX = 17000.
NUM_BANDS = 12
NORM_FILTERS = True
UNIQUE_FILTERS = True


class Filterbank(np.ndarray):
    """
    Generic filterbank class.

    A Filterbank is a simple numpy array enhanced with several additional
    attributes, e.g. number of bands.

    A Filterbank has a shape of (num_bins, num_bands) and can be used to
    filter a spectrogram of shape (num_frames, num_bins) to (num_frames,
    num_bands).

    Parameters
    ----------
    data : numpy array, shape (num_bins, num_bands)
        Data of the filterbank .
    bin_frequencies : numpy array, shape (num_bins, )
        Frequencies of the bins [Hz].

    Notes
    -----
    The length of `bin_frequencies` must be equal to the first dimension
    of the given `data` array.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, data, bin_frequencies):
        # this method is for documentation purposes only
        pass

    def __new__(cls, data, bin_frequencies):
        # input is an numpy ndarray instance
        if isinstance(data, np.ndarray) and data.ndim == 2:
            # cast as Filterbank
            obj = np.asarray(data, dtype=FILTER_DTYPE).view(cls)
        else:
            raise TypeError('wrong input data for Filterbank, must be a 2D '
                            'np.ndarray')
        # set bin frequencies
        if len(bin_frequencies) != obj.shape[0]:
            raise ValueError('`bin_frequencies` must have the same length as '
                             'the first dimension of `data`.')
        obj.bin_frequencies = np.asarray(bin_frequencies, dtype=np.float)
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.bin_frequencies = getattr(obj, 'bin_frequencies', None)

    @classmethod
    def _put_filter(cls, filt, band):
        """
        Puts a filter in the band, internal helper function.

        Parameters
        ----------
        filt : :class:`Filter` instance
            Filter to be put into the band.
        band : numpy array
            Band in which the filter should be put.

        Notes
        -----
        The `band` must be an existing numpy array where the filter `filt` is
        put in, given the position of the filter. Out of range filters are
        truncated. If there are non-zero values in the filter band at the
        respective positions, the maximum value of the `band` and the filter
        `filt` is used.

        """
        if not isinstance(filt, Filter):
            raise ValueError('unable to determine start position of Filter')
        # determine start and stop positions
        start = filt.start
        stop = start + len(filt)
        # truncate the filter if it starts before the 0th band bin
        if start < 0:
            filt = filt[-start:]
            start = 0
        # truncate the filter if it ends after the last band bin
        if stop > len(band):
            filt = filt[:-(stop - len(band))]
            stop = len(band)
        # put the filter in place
        filter_position = band[start:stop]
        # TODO: if needed, allow other handling (like summing values)
        np.maximum(filt, filter_position, out=filter_position)

    @classmethod
    def from_filters(cls, filters, bin_frequencies):
        """
        Create a filterbank with possibly multiple filters per band.

        Parameters
        ----------
        filters : list (of lists) of Filters
            List of Filters (per band); if multiple filters per band are
            desired, they should be also contained in a list, resulting in a
            list of lists of Filters.
        bin_frequencies : numpy array
            Frequencies of the bins (needed to determine the expected size of
            the filterbank).

        Returns
        -------
        filterbank : :class:`Filterbank` instance
            Filterbank with respective filter elements.

        """
        # create filterbank
        fb = np.zeros((len(bin_frequencies), len(filters)))
        # iterate over all filters
        for band_id, band_filter in enumerate(filters):
            # get the band's corresponding slice of the filterbank
            band = fb[:, band_id]
            # if there's a list of filters for the current band, put them all
            # into this band
            if isinstance(band_filter, list):
                for filt in band_filter:
                    cls._put_filter(filt, band)
            # otherwise put this filter into that band
            else:
                cls._put_filter(band_filter, band)
        # create Filterbank and cast as class where this method was called from
        return Filterbank.__new__(cls, fb, bin_frequencies)

    @property
    def num_bins(self):
        """Number of bins."""
        return self.shape[0]

    @property
    def num_bands(self):
        """Number of bands."""
        return self.shape[1]

    @property
    def corner_frequencies(self):
        """Corner frequencies of the filter bands."""
        freqs = []
        for band in range(self.num_bands):
            # get the non-zero bins per band
            bins = np.nonzero(self[:, band])[0]
            # append the lowest and highest bin
            freqs.append([np.min(bins), np.max(bins)])
        # map to frequencies
        return bins2frequencies(freqs, self.bin_frequencies)

    @property
    def center_frequencies(self):
        """Center frequencies of the filter bands."""
        freqs = []
        for band in range(self.num_bands):
            # get the non-zero bins per band
            bins = np.nonzero(self[:, band])[0]
            min_bin = np.min(bins)
            max_bin = np.max(bins)
            # if we have a uniform filter, use the center bin
            if self[min_bin, band] == self[max_bin, band]:
                center = int(min_bin + (max_bin - min_bin) / 2.)
            # if we have a filter with a peak, use the peak bin
            else:
                center = min_bin + np.argmax(self[min_bin: max_bin, band])
            freqs.append(center)
        # map to frequencies
        return bins2frequencies(freqs, self.bin_frequencies)

    @property
    def fmin(self):
        """Minimum frequency of the filterbank."""
        return self.bin_frequencies[np.nonzero(self)[0][0]]

    @property
    def fmax(self):
        """Maximum frequency of the filterbank."""
        return self.bin_frequencies[np.nonzero(self)[0][-1]]


class FilterbankProcessor(Processor, Filterbank):
    """
    Generic filterbank processor class.

    A FilterbankProcessor is a simple wrapper for Filterbank which adds a
    process() method.

    See Also
    --------
    :class:`Filterbank`

    """
    # Note: this class is only for consistency of the naming scheme. Basically
    #       the process()

    def process(self, data):
        """
        Filter the given data with the Filterbank.

        Parameters
        ----------
        data : 2D numpy array
            Data to be filtered.
        Returns
        -------
        filt_data : numpy array
            Filtered data.

        Notes
        -----
        This method makes the :class:`Filterbank` act as a :class:`Processor`.

        """
        # Note: we do not inherit from Processor, since instantiation gets
        #       messed up
        return np.dot(data, self)

    @staticmethod
    def add_arguments(parser, filterbank=None, num_bands=None,
                      crossover_frequencies=None, fmin=None, fmax=None,
                      norm_filters=None, unique_filters=None):
        """
        Add filterbank related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        filterbank : :class:`.audio.filters.Filterbank`, optional
            Use a filterbank of that type.
        num_bands : int or list, optional
            Number of bands (per octave).
        crossover_frequencies : list or numpy array, optional
            List of crossover frequencies at which the `spectrogram` is split
            into bands.
        fmin : float, optional
            Minimum frequency of the filterbank [Hz].
        fmax : float, optional
            Maximum frequency of the filterbank [Hz].
        norm_filters : bool, optional
            Normalize the filters of the filterbank to area 1.
        unique_filters : bool, optional
            Indicate if the filterbank should contain only unique filters,
            i.e. remove duplicate filters resulting from insufficient
            resolution at low frequencies.

        Returns
        -------
        argparse argument group
            Filterbank argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.
        Depending on the type of the `filterbank`, either `num_bands` or
        `crossover_frequencies` should be used.

        """
        from madmom.utils import OverrideDefaultListAction
        # add filterbank related options to the existing parser
        g = parser.add_argument_group('filterbank arguments')
        # filterbank
        # TODO: add a list with filterbank options?
        if filterbank is not None:
            if issubclass(filterbank, Filterbank):
                g.add_argument('--no_filter', dest='filterbank',
                               action='store_false', default=filterbank,
                               help='do not filter the spectrogram with a '
                                    'filterbank [default=%(default)s]')
            else:
                g.add_argument('--filterbank', action='store_true',
                               default=None,
                               help='filter the spectrogram with a filterbank '
                                    'of this type')
        # number of bands
        # TODO: add a second argument with num_bands_per_octave and rename the
        #       option at the relevant filterbanks accordingly?
        # depending on the type of num_bands, use different options
        if isinstance(num_bands, int):
            g.add_argument('--num_bands', action='store', type=int,
                           default=num_bands,
                           help='number of filter bands (per octave) '
                                '[default=%(default)i]')
        elif isinstance(num_bands, list):
            # Note: this option can be used in conjunction with stacked
            #       spectrograms with different frame sizes to have different
            #       number of bands per frame size
            g.add_argument('--num_bands', type=int, default=num_bands,
                           action=OverrideDefaultListAction, sep=',',
                           help='(comma separated list of) number of filter '
                                'bands (per octave) [default=%(default)s]')
        # crossover frequencies
        if crossover_frequencies is not None:
            g.add_argument('--crossover_frequencies', type=float, sep=',',
                           action=OverrideDefaultListAction,
                           default=crossover_frequencies,
                           help='(comma separated) list with crossover '
                                'frequencies [Hz, default=%(default)s]')
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


class MelFilterbank(Filterbank):
    """
    Mel filterbank class.

    Parameters
    ----------
    bin_frequencies : numpy array
        Frequencies of the bins [Hz].
    num_bands : int, optional
        Number of filter bands.
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filters to area 1.
    unique_filters : bool, optional
        Keep only unique filters, i.e. remove duplicate filters resulting
        from insufficient resolution at low frequencies.

    Notes
    -----
    Because of rounding and mapping of frequencies to bins and back to
    frequencies, the actual minimum, maximum and center frequencies do not
    necessarily match the parameters given.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    NUM_BANDS = 40
    FMIN = 20.
    FMAX = 17000.
    NORM_FILTERS = True
    UNIQUE_FILTERS = True

    def __init__(self, bin_frequencies, num_bands=NUM_BANDS, fmin=FMIN,
                 fmax=FMAX, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, bin_frequencies, num_bands=NUM_BANDS, fmin=FMIN,
                fmax=FMAX, norm_filters=NORM_FILTERS,
                unique_filters=UNIQUE_FILTERS, **kwargs):
        # pylint: disable=arguments-differ
        # get a list of frequencies aligned on the Mel scale
        # request 2 more bands, because these are the edge frequencies
        frequencies = mel_frequencies(num_bands + 2, fmin, fmax)
        # convert to bins
        bins = frequencies2bins(frequencies, bin_frequencies,
                                unique_bins=unique_filters)
        # get overlapping triangular filters
        filters = TriangularFilter.filters(bins, norm=norm_filters,
                                           overlap=True)
        # create a MelFilterbank from the filters
        return cls.from_filters(filters, bin_frequencies)


class LogarithmicFilterbank(Filterbank):
    """
    Logarithmic filterbank class.

    Parameters
    ----------
    bin_frequencies : numpy array
        Frequencies of the bins [Hz].
    num_bands : int, optional
        Number of filter bands (per octave).
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    fref : float, optional
        Tuning frequency of the filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filters to area 1.
    unique_filters : bool, optional
        Keep only unique filters, i.e. remove duplicate filters resulting
        from insufficient resolution at low frequencies.
    bands_per_octave : bool, optional
        Indicates whether `num_bands` is given as number of bands per octave
        ('True', default) or as an absolute number of bands ('False').

    Notes
    -----
    `num_bands` sets either the number of bands per octave or the total number
    of bands, depending on the setting of `bands_per_octave`. `num_bands` is
    used to set also the number of bands per octave to keep the argument for
    all classes the same. If 12 bands per octave are used, a filterbank with
    semitone spacing is created.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    NUM_BANDS_PER_OCTAVE = 12

    def __init__(self, bin_frequencies, num_bands=NUM_BANDS_PER_OCTAVE,
                 fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS, bands_per_octave=True):
        # this method is for documentation purposes only
        pass

    def __new__(cls, bin_frequencies, num_bands=NUM_BANDS_PER_OCTAVE,
                fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                unique_filters=UNIQUE_FILTERS, bands_per_octave=True):
        # pylint: disable=arguments-differ
        # decide whether num_bands is bands per octave or total number of bands
        if bands_per_octave:
            num_bands_per_octave = num_bands
            # get a list of frequencies with logarithmic scaling
            frequencies = log_frequencies(num_bands, fmin, fmax, fref)
            # convert to bins
            bins = frequencies2bins(frequencies, bin_frequencies,
                                    unique_bins=unique_filters)
        else:
            # iteratively get the number of bands
            raise NotImplementedError("please implement `num_bands` with "
                                      "`bands_per_octave` set to 'False' for "
                                      "LogarithmicFilterbank")
        # get overlapping triangular filters
        filters = TriangularFilter.filters(bins, norm=norm_filters,
                                           overlap=True)
        # create a LogarithmicFilterbank from the filters
        obj = cls.from_filters(filters, bin_frequencies)
        # set additional attributes
        obj.fref = fref
        obj.num_bands_per_octave = num_bands_per_octave
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.num_bands_per_octave = getattr(obj, 'num_bands_per_octave',
                                            self.NUM_BANDS_PER_OCTAVE)
        self.fref = getattr(obj, 'fref', A4)


# alias
LogFilterbank = LogarithmicFilterbank


class RectangularFilterbank(Filterbank):
    """
    Rectangular filterbank class.

    Parameters
    ----------
    bin_frequencies : numpy array
        Frequencies of the bins [Hz].
    crossover_frequencies : list or numpy array
        Crossover frequencies of the bands [Hz].
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    norm_filters : bool, optional
        Normalize the filters to area 1.
    unique_filters : bool, optional
        Keep only unique filters, i.e. remove duplicate filters resulting
        from insufficient resolution at low frequencies.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, bin_frequencies, crossover_frequencies, fmin=FMIN,
                 fmax=FMAX, norm_filters=NORM_FILTERS,
                 unique_filters=UNIQUE_FILTERS):
        # this method is for documentation purposes only
        pass

    def __new__(cls, bin_frequencies, crossover_frequencies, fmin=FMIN,
                fmax=FMAX, norm_filters=NORM_FILTERS,
                unique_filters=UNIQUE_FILTERS):
        # pylint: disable=arguments-differ
        # create an empty filterbank
        fb = np.zeros((len(bin_frequencies), len(crossover_frequencies) + 1),
                      dtype=FILTER_DTYPE)
        corner_frequencies = np.r_[fmin, crossover_frequencies, fmax]
        # get the corner bins
        corner_bins = frequencies2bins(corner_frequencies, bin_frequencies,
                                       unique_bins=unique_filters)
        # map the bins to the filterbank bands
        for i in range(len(corner_bins) - 1):
            fb[corner_bins[i]:corner_bins[i + 1], i] = 1
        # normalize the filterbank
        if norm_filters:
            # if the sum over a band is zero, do not normalize this band
            band_sum = np.sum(fb, axis=0)
            band_sum[band_sum == 0] = 1
            fb /= band_sum
        # create Filterbank and cast as RectangularFilterbank
        obj = Filterbank.__new__(cls, fb, bin_frequencies)
        # set additional attributes
        obj.crossover_frequencies = bins2frequencies(corner_bins[1:-1],
                                                     bin_frequencies)
        # return the object
        return obj


# chroma / harmonic filterbanks
class SemitoneBandpassFilterbank(object):
    """
    Time domain semitone filterbank of elliptic filters as proposed in [1]_.

    Parameters
    ----------
    order : int, optional
        Order of elliptic filters.
    passband_ripple : float, optional
        Maximum ripple allowed below unity gain in the passband [dB].
    stopband_rejection : float, optional
        Minimum attenuation required in the stop band [dB].
    q_factor : int, optional
        Q-factor of the filters.
    fmin : float, optional
        Minimum frequency of the filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the filterbank [Hz].
    fref : float, optional
        Reference frequency for the first bandpass filter [Hz].

    References
    ----------
    .. [1] Meinard Müller,
           "Information retrieval for music and motion", Springer, 2007.

    Notes
    -----
    This is a time domain filterbank, thus it cannot be used as the other
    time-frequency filterbanks of this module. Instead of ``np.dot()`` use
    ``scipy.signal.filtfilt()`` to filter a signal.

    """

    def __init__(self, order=4, passband_ripple=1, stopband_rejection=50,
                 q_factor=25, fmin=27.5, fmax=4200., fref=A4):
        from scipy.signal import ellip
        self.order = order
        self.passband_ripple = passband_ripple
        self.stopband_rejection = stopband_rejection
        self.q_factor = q_factor
        self.fref = fref
        self.center_frequencies = semitone_frequencies(fmin, fmax, fref=fref)
        # use different sample rates for the individual bands
        self.band_sample_rates = np.ones_like(self.center_frequencies) * 4410
        self.band_sample_rates[self.center_frequencies > 2000] = 22050
        self.band_sample_rates[self.center_frequencies < 250] = 882
        self.filters = []
        for freq, sample_rate in zip(self.center_frequencies,
                                     self.band_sample_rates):
            freqs = [(freq - freq / q_factor / 2.) * 2. / sample_rate,
                     (freq + freq / q_factor / 2.) * 2. / sample_rate]
            self.filters.append(ellip(order, passband_ripple,
                                      stopband_rejection, freqs,
                                      btype='bandpass'))

    @property
    def num_bands(self):
        """Number of bands."""
        return len(self.center_frequencies)

    @property
    def fmin(self):
        """Minimum frequency of the filterbank."""
        f = self.center_frequencies[0]
        return f - f / self.q_factor / 2.

    @property
    def fmax(self):
        """Maximum frequency of the filterbank."""
        f = self.center_frequencies[-1]
        return f + f / self.q_factor / 2.
