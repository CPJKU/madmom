#!/usr/bin/env python
# encoding: utf-8
"""
This file contains filter and filterbank related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np

from madmom import Processor

A4 = 440


# Mel frequency scale
def hz2mel(f):
    """
    Convert Hz frequencies to Mel.

    :param f: input frequencies [Hz]
    :return:  frequencies in Mel

    """
    return 1127.01048 * np.log(np.asarray(f) / 700. + 1.)


def mel2hz(m):
    """
    Convert Mel frequencies to Hz.

    :param m: input frequencies [Mel]
    :return:  frequencies in Hz

    """
    return 700. * (np.exp(np.asarray(m) / 1127.01048) - 1.)


def mel_frequencies(num_bands, fmin, fmax):
    """
    Generates a list of frequencies aligned on the Mel scale.

    :param num_bands: number of bands
    :param fmin:      the minimum frequency [Hz]
    :param fmax:      the maximum frequency [Hz]
    :return:          a list of frequencies

    """
    # convert fmin and fmax to the Mel scale and return an array of frequencies
    return mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), num_bands))


# Bark frequency scale
def hz2bark(f):
    """
    Convert Hz frequencies to Bark.

    :param f: input frequencies [Hz]
    :return:  frequencies in Bark.

    """
    raise NotImplementedError('please check this function, it produces '
                              'negative values')
    # TODO: use Zwicker's formula?
    #       return 13 * arctan(0.00076 * f) + 3.5 * arctan((f / 7500.) ** 2)
    return (26.81 / (1. + 1960. / np.asarray(f))) - 0.53


def bark2hz(z):
    """
    Convert Bark frequencies to Hz.

    :param z: input frequencies [Bark]
    :return:  frequencies in Hz.

    """
    raise NotImplementedError('please check this function, it produces weird '
                              'values')
    # TODO: use Zwicker's formula? what's the inverse of the above?
    return 1960. / (26.81 / (np.asarray(z) + 0.53) - 1.)


def bark_frequencies(fmin=20, fmax=15500):
    """
    Generates a list of corner frequencies aligned on the Bark-scale.

    :param fmin: the minimum frequency [Hz]
    :param fmax: the maximum frequency [Hz]
    :return:     a list of frequencies

    """
    # frequencies aligned to the Bark-scale
    frequencies = np.array([20, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
                            1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
                            4400, 5300, 6400, 7700, 9500, 12000, 15500])
    # filter frequencies
    frequencies = frequencies[np.searchsorted(frequencies, fmin):]
    frequencies = frequencies[:np.searchsorted(frequencies, fmax, 'right')]
    # return
    return frequencies


def bark_double_frequencies(fmin=20, fmax=15500):
    """
    Generates a list of corner frequencies aligned on the Bark-scale.
    The list includes also center frequencies between the corner frequencies.

    :param fmin: the minimum frequency [Hz]
    :param fmax: the maximum frequency [Hz]
    :return:     a list of frequencies

    """
    # frequencies aligned to the Bark-scale, also includes center frequencies
    frequencies = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 450,
                            510, 570, 630, 700, 770, 840, 920, 1000, 1080,
                            1170, 1270, 1370, 1480, 1600, 1720, 1850, 2000,
                            2150, 2320, 2500, 2700, 2900, 3150, 3400, 3700,
                            4000, 4400, 4800, 5300, 5800, 6400, 7000, 7700,
                            8500, 9500, 10500, 12000, 13500, 15500])
    # filter frequencies
    frequencies = frequencies[np.searchsorted(frequencies, fmin):]
    frequencies = frequencies[:np.searchsorted(frequencies, fmax, 'right')]
    # return
    return frequencies


# logarithmic frequency scale
def log_frequencies(bands_per_octave, fmin, fmax, fref=A4):
    """
    Generates a list of frequencies aligned on a logarithmic frequency scale.

    :param bands_per_octave: number of filter bands per octave
    :param fmin:             the minimum frequency [Hz]
    :param fmax:             the maximum frequency [Hz]
    :param fref:             tuning frequency [Hz]
    :return:                 a list of frequencies

    Note: if 12 bands per octave and a4=440 are used, the frequencies are
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


def semitone_frequencies(fmin, fmax, a4=A4):
    """
    Generates a list of frequencies separated by semitones.

    :param fmin: the minimum frequency [Hz]
    :param fmax: the maximum frequency [Hz]
    :param a4:   tuning frequency of A4 [Hz]
    :return:     a list of frequencies of semitones

    """
    # return MIDI frequencies
    return log_frequencies(12, fmin, fmax, a4)


# MIDI
def midi2hz(m, fref=A4):
    """
    Convert frequencies to the corresponding MIDI notes.

    :param m:    input MIDI notes
    :param fref: tuning frequency of A4 [Hz]
    :return:     frequencies in Hz

    For details see: http://www.phys.unsw.edu.au/jw/notes.html

    """
    return 2. ** ((m - 69.) / 12.) * fref


def hz2midi(f, fref=A4):
    """
    Convert MIDI notes to corresponding frequencies.

    :param f:    input frequencies [Hz]
    :param fref: tuning frequency of A4 [Hz]
    :return:     MIDI notes

    For details see: at http://www.phys.unsw.edu.au/jw/notes.html

    Note: This function does not necessarily return a valid MIDI Note, you may
          need to round it to the nearest integer.

    """
    return (12. * np.log2(f / float(fref))) + 69.


# provide an alias to semitone_frequencies
midi_frequencies = semitone_frequencies


# ERB frequency scale
def hz2erb(f):
    """
    Convert Hz to ERB.

    :param f: input frequencies [Hz]
    :return:  frequencies in ERB

    Information about the ERB scale can be found at:
    https://ccrma.stanford.edu/~jos/bbt/Equivalent_Rectangular_Bandwidth.html

    """
    return 21.4 * np.log10(1 + 4.37 * np.asarray(f) / 1000.)


def erb2hz(e):
    """
    Convert ERB scaled frequencies to Hz.

    :param e: input frequencies [ERB]
    :return:  frequencies in Hz

    Information about the ERB scale can be found at:
    https://ccrma.stanford.edu/~jos/bbt/Equivalent_Rectangular_Bandwidth.html

    """
    return (10. ** (np.asarray(e) / 21.4) - 1.) * 1000. / 4.37


# helper functions for filter creation
def frequencies2bins(frequencies, bin_frequencies):
    """
    Map frequencies to the closest corresponding bins.

    :param frequencies:     a list of frequencies [numpy array, Hz]
    :param bin_frequencies: frequencies of the bins [numpy array, Hz]
    :return:                corresponding bins [numpy array]

    """
    # cast as numpy arrays
    frequencies = np.asarray(frequencies)
    bin_frequencies = np.asarray(bin_frequencies)
    # map the frequencies to spectrogram bins
    # solution found at: http://stackoverflow.com/questions/8914491/
    indices = bin_frequencies.searchsorted(frequencies)
    indices = np.clip(indices, 1, len(bin_frequencies) - 1)
    left = bin_frequencies[indices - 1]
    right = bin_frequencies[indices]
    indices -= frequencies - left < right - frequencies
    # return the indices of the closest matches
    return indices


def bins2frequencies(bins, bin_frequencies):
    """
    Convert bins to the corresponding frequencies.

    :param bins:            a list of bins [numpy array]
    :param bin_frequencies: frequencies of the bins [numpy array, Hz]
    :return:                corresponding frequencies [numpy array, Hz]

    """
    # map the frequencies to spectrogram bins
    return np.asarray(bin_frequencies, dtype=np.float)[np.asarray(bins)]


# filter classes
class Filter(np.ndarray):
    """
    Generic filter class.

    """

    def __new__(cls, data, start=0):
        """
        Creates a new Filter instance.

        :param data:  1D numpy array
        :param start: start position [int]

        The start position is mandatory if this Filter should be used for the
        creation of a Filterbank. If not set, a start position of 0 is assumed.

        """
        # input is an numpy ndarray instance
        if isinstance(data, np.ndarray):
            # cast as Filter
            obj = np.asarray(data).view(cls)
        else:
            raise TypeError('wrong input data for Filter, must be np.ndarray')
        # right now, allow only 1D
        if data.ndim != 1:
            raise NotImplementedError('please add multi-dimension support')
        # set attributes
        obj.start = int(start)
        obj.stop = int(start + len(data))
        # return the object
        return obj

    def __reduce__(self):
        # needed for correct pickling
        # source: http://stackoverflow.com/questions/26598109/
        # get the parent's __reduce__ tuple
        pickled_state = super(Filter, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.start, self.stop,)
        # return a tuple that replaces the parent's __setstate__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for correct un-pickling
        # set the sample_rate
        self.start = state[-2]
        self.stop = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(Filter, self).__setstate__(state[0:-2])

    @classmethod
    def band_bins(cls, bins, **kwargs):
        """
        Must yield the center/crossover bins needed for filter creation.

        :param bins:   center/crossover bins of filters [list or numpy array]
        :param kwargs: additional parameters
        :return:       bins and normalisation information for filter creation

        """
        raise NotImplementedError('needs to be implemented by sub-classes')

    @classmethod
    def filters(cls, bins, **kwargs):
        """
        Creates a list with filters for the the given bins.

        :param bins:   (center/crossover) bins of filters [list or numpy array]
        :param kwargs: additional parameters passed to band_bins()
        :return:       list with filters

        """
        # generate a list of filters for the given center/crossover bins
        filters = []
        for filter_args in cls.band_bins(bins, **kwargs):
            # create a filter and append it to the list
            filters.append(cls(*filter_args))
        # return the filters
        return filters


class TriangularFilter(Filter):
    """
    Triangular filter class.

    """

    def __new__(cls, start, center, stop, norm=False):
        """
        Creates a new TriangularFilter instance.

        :param start:  start bin
        :param center: center bin (of height 1, unless filter is normalized)
        :param stop:   stop bin
        :param norm:   normalize the area of the filter(s) to 1
        :return:       a triangular shaped filter with length 'stop', height 1
                       (unless normalized) with indices <= 'start' set to 0

        """
        # center must be between start & stop
        if start >= center >= stop:
            raise ValueError('center must be between start and stop')
        # make center and stop relative
        center -= start
        stop -= start
        # set the height of the filter, normalized if necessary.
        # A standard filter is at least 3 bins wide, and stop - start = 2
        # thus the filter has an area of 1 if normalized this way
        height = 2. / stop if norm else 1.
        # create filter
        data = np.zeros(stop)
        # rising edge (without the center)
        data[:center] = np.linspace(0, height, center, endpoint=False)
        # falling edge (including the center, but without the last bin)
        data[center:] = np.linspace(height, 0, stop - center, endpoint=False)
        # cast to TriangularFilter
        obj = Filter.__new__(cls, data, start)
        # set the center bin
        obj.center = start + center
        # return the filter
        return obj

    def __reduce__(self):
        # needed for correct pickling
        # source: http://stackoverflow.com/questions/26598109/
        # get the parent's __reduce__ tuple
        pickled_state = super(TriangularFilter, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.center,)
        # return a tuple that replaces the parent's __setstate__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for correct un-pickling
        # set the sample_rate
        self.center = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(TriangularFilter, self).__setstate__(state[0:-1])

    @classmethod
    def band_bins(cls, bins, norm=True, duplicates=False, overlap=True):
        """
        Yields start, center and stop bins and normalisation info for creation
        of triangular filters.

        :param bins:       center bins of filters [list or numpy array]
        :param norm:       normalize the area of the filter(s) to 1 [bool]
        :param duplicates: keep duplicate filters resulting from insufficient
                           resolution of low frequencies
        :param overlap:    filters should overlap [bool]
        :return:           start, center and stop bins & normalisation info

        Note: If `duplicates` is set, duplicate filter bins are kept as is,
              otherwise they are removed, i.e. any filter bin is included only
              1 time at most.
              If `overlap` is 'False', the 'start' and 'stop' bins of the
              filters are interpolated between the centre bins, normal rounding
              applies.

        """
        # only keep unique bins if requested
        # Note: this can be important to do so, otherwise the lower frequency
        #       bins can be given too much weight if simply summed up (as in
        #       the spectral flux)
        if not duplicates:
            bins = np.unique(bins)
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
                start = int(round((center + start) / 2.))
                stop = int(round((center + stop) / 2.))
            # consistently handle too-small filters
            if duplicates and (stop - start < 2):
                center = start
                stop = start + 1
            # yield the bins and continue
            yield start, center, stop, norm
            # increase counter
            index += 1


class RectangularFilter(Filter):
    """
    Rectangular filter class.

    """

    def __new__(cls, start, stop, norm=False):
        """
        Creates a new RectangularFilter instance.

        :param start: start bin of the filter
        :param stop:  stop bin of the filter
        :param norm:  normalize the area of the filter(s) to 1
        :return:      a rectangular shaped filter with length 'stop', height 1
                      (unless normalized) with indices <= 'start' set to 0

        """
        # start must be smaller than stop
        if start >= stop:
            raise ValueError('start must be smaller than stop')
        # length of the filter
        length = stop - start
        # set the height of the filter, normalized if necessary
        height = 1. / length if norm else 1.
        # create filter
        data = np.ones(length, dtype=np.float) * height
        # cast to RectangularFilter and return it
        return Filter.__new__(cls, data, start)

    @classmethod
    def band_bins(cls, bins, norm=True, duplicates=False, overlap=False):
        """
        Yields start and stop bins and normalisation info for creation of
        rectangular filters.

        :param bins:       crossover bins of filters [numpy array]
        :param norm:       normalize the area of the filter(s) to 1
        :param duplicates: keep duplicate filters resulting from insufficient
                           resolution of low frequencies
        :param overlap:    filters should overlap
        :return:           start and stop bins & normalisation info

        Note: If `duplicates` is set, duplicate filter bins are kept as is,
              otherwise they are removed, i.e. any filter bin is included only
              1 time at most.

        """
        # only keep unique bins if requested
        # Note: this can be important to do so, otherwise the lower frequency
        #       bins can be given too much weight if simply summed up (as in
        #       the spectral flux)
        if not duplicates:
            bins = np.unique(bins)
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
            yield start, stop, norm
            # increase counter
            index += 1

# default values for filter banks
FMIN = 30.
FMAX = 17000.
BANDS = 12
NORM_FILTERS = True
DUPLICATE_FILTERS = False
OVERLAP_FILTERS = True


class Filterbank(np.ndarray):
    """
    Generic filterbank class.

    A Filterbank is a simple numpy array enhanced with several additional
    attributes, e.g. number of bands.

    A Filterbank has a shape of (num_bins x num_bands) and can be used to
    filter a spectrogram of shape (num_frames x num_bins) to (num_frames x
    num_bands).

    """
    fref = None

    def __new__(cls, data, bin_frequencies):
        """
        Creates a new Filterbank instance.

        :param data:            2D numpy array (num_bins x num_bands)
        :param bin_frequencies: frequencies of the bins (must be of length
                                num_bins)

        """
        # input is an numpy ndarray instance
        if isinstance(data, np.ndarray) and data.ndim == 2:
            # cast as Filterbank
            obj = np.asarray(data).view(cls)
        else:
            raise TypeError('wrong input data for Filterbank, must be a 2D '
                            'np.ndarray')
        # set bin frequencies
        if len(bin_frequencies) != obj.shape[0]:
            raise ValueError("'bin_frequencies' must have the same length as "
                             "the first dimension of 'data'.")
        obj.bin_frequencies = np.asarray(bin_frequencies, dtype=np.float)
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.bin_frequencies = getattr(obj, 'bin_frequencies', None)

    def __reduce__(self):
        # needed for correct pickling
        # source: http://stackoverflow.com/questions/26598109/
        # get the parent's __reduce__ tuple
        pickled_state = super(Filterbank, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.bin_frequencies,)
        # return a tuple that replaces the parent's __setstate__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for correct un-pickling
        # set the sample_rate
        self.bin_frequencies = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(Filterbank, self).__setstate__(state[0:-1])

    @classmethod
    def _put_filter(cls, filt, band):
        """
        Puts a filter in the band, internal helper function.

        :param filt: Filter instance
        :param band: band in which the filter should be put [numpy array]

        Note: The `band` must be an existing numpy array where the filter
              `filt` is put in, given the position of the filter.
              Out of range filters are truncated.
              If there are non-zero values in the filter band at the respective
              positions, the maximum value of the `band` and the `filt` is
              used.

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
        Creates a filterbank with possibly multiple filters per band.

        :param filters:         list containing the Filters per band; if
                                multiple filters per band are desired, they
                                should be also contained in a list, resulting
                                in a list of lists of Filters
        :param bin_frequencies: frequencies of the bins (needed to determine
                                the expected size of the filterbank)
        :return:                filterbank with respective filter elements

        """
        # create filterbank
        fb = np.zeros((len(bin_frequencies), len(filters)))
        # iterate over all filters
        for band_id, band_filter in enumerate(filters):
            # get the band's corresponding slice of the filterbank
            band = fb[:, band_id]
            # if there's a list of filters for the current band, put them all
            # into this band
            if type(band_filter) is list:
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

    def process(self, data):
        """
        Filter the given data with the Filterbank.

        :param data: data [2D numpy array]
        :return:     filtered data

        """
        # this method makes the Filterbank act as a Processor
        # Note: we do not inherit from Processor, since instantiation gets
        #       messed up
        return np.dot(data, self)

    @staticmethod
    def add_arguments(parser, filter_type=None, fmin=None, fmax=None,
                      bands=None, norm_filters=None, duplicate_filters=None):
        """
        Add filter related arguments to an existing parser.

        :param parser:            existing argparse parser
        :param filter_type:       type of filter to be used
        :param fmin:              the minimum frequency [Hz, float]
        :param fmax:              the maximum frequency [Hz, float]
        :param bands:             number of filter bands (per octave) [int]
        :param norm_filters:      normalize the area of the filter [bool]
        :param duplicate_filters: keep duplicate filters [bool]
        :return:                  filtering argument parser group

        Parameters are included in the group only if they are not 'None'.

        """
        # TODO: split this among the individual classes
        # add filter related options to the existing parser
        g = parser.add_argument_group('filterbank related arguments')
        if filter_type is not None:
            # FIXME: how to handle this option?
            g.add_argument('--filter_type', action='store',
                           default=filter_type,
                           help='filter type [default=%(default)s]')
        if bands is not None:
            g.add_argument('--bands', action='store', type=int, default=bands,
                           help='number of bands (per octave) '
                                '[default=%(default)i]')
        if fmin is not None:
            g.add_argument('--fmin', action='store', type=float, default=fmin,
                           help='minimum frequency of filter '
                                '[Hz, default=%(default).2f]')
        if fmax is not None:
            g.add_argument('--fmax', action='store', type=float, default=fmax,
                           help='maximum frequency of filter '
                                '[Hz, default=%(default).2f]')
        if norm_filters is False:
            # switch to turn it on
            g.add_argument('--norm_filters', action='store_true',
                           default=norm_filters,
                           help='normalize filters to have equal area')
        elif norm_filters is True:
            g.add_argument('--no_norm_filters', dest='norm_filters',
                           action='store_false', default=norm_filters,
                           help='do not equalize filters to have equal area')
        if duplicate_filters is False:
            # switch to turn it on
            g.add_argument('--duplicate_filters', action='store_true',
                           default=duplicate_filters,
                           help='keep duplicate filters')
        elif duplicate_filters is True:
            g.add_argument('--no_duplicate_filters', dest='norm_filters',
                           action='store_false', default=duplicate_filters,
                           help='remove duplicate filters')
        # return the argument group so it can be modified if needed
        return g


class MelFilterbank(Filterbank):
    """
    Mel filterbank class.

    """
    BANDS = 40
    FMIN = 20.
    FMAX = 17000.
    NORM_FILTERS = True
    DUPLICATE_FILTERS = False

    def __new__(cls, bin_frequencies, bands=BANDS, fmin=FMIN, fmax=FMAX,
                norm_filters=NORM_FILTERS, duplicate_filters=DUPLICATE_FILTERS,
                **kwargs):
        """
        Creates a new MelFilterbank instance.

        :param bin_frequencies:   frequencies of the bins [Hz]
        :param bands:             number of filter bands
        :param fmin:              the minimum frequency [Hz]
        :param fmax:              the maximum frequency [Hz]
        :param norm_filters:      normalize the filters to area 1
        :param duplicate_filters: keep duplicate filters resulting from
                                  insufficient resolution of low frequencies

        Note: Because of rounding and mapping of frequencies to bins and vice
              versa, the actual minimum, maximum and center frequencies do not
              necessarily reflect the given parameters.

        """
        # get a list of frequencies
        # request 2 more bands, because these are the edge frequencies
        frequencies = mel_frequencies(bands + 2, fmin, fmax)
        # convert to bins
        bins = frequencies2bins(frequencies, bin_frequencies)
        # get overlapping triangular filters
        filters = TriangularFilter.filters(bins, norm=norm_filters,
                                           duplicates=duplicate_filters,
                                           overlap=True)
        # create a MelFilterbank from the filters
        return cls.from_filters(filters, bin_frequencies)


class BarkFilterbank(Filterbank):
    """
    Bark filterbank class.

    """
    FMIN = 20.
    FMAX = 15500.
    BANDS = 'simple'
    NORM_FILTERS = True
    DUPLICATE_FILTERS = False

    def __new__(cls, bin_frequencies, bands=BANDS, fmin=FMIN, fmax=FMAX,
                norm_filters=NORM_FILTERS, duplicate_filters=DUPLICATE_FILTERS,
                **kwargs):
        """
        Creates a new BarkFilterbank instance.

        :param bin_frequencies:   frequencies of the bins [Hz]
        :param bands:             number of filter bands
        :param fmin:              the minimum frequency [Hz]
        :param fmax:              the maximum frequency [Hz]
        :param norm_filters:      normalize the filters to area 1
        :param duplicate_filters: keep duplicate filters resulting from
                                  insufficient resolution of low frequencies

        """
        # get a list of frequencies
        if bands == 'simple':
            frequencies = bark_frequencies(fmin, fmax)
        elif bands == 'double':
            frequencies = bark_double_frequencies(fmin, fmax)
        else:
            raise ValueError("bands must be either 'simple' or 'double'")
        # convert to bins
        bins = frequencies2bins(frequencies, bin_frequencies)
        # get non-overlapping rectangular filters
        filters = RectangularFilter.filters(bins, norm=norm_filters,
                                            duplicates=duplicate_filters,
                                            overlap=False)
        # create a BarkFilterbank from the filters
        return cls.from_filters(filters, bin_frequencies)


class LogarithmicFilterbank(Filterbank):
    """
    Logarithmic filterbank class.

    """
    BANDS_PER_OCTAVE = 12

    def __new__(cls, bin_frequencies, bands=BANDS_PER_OCTAVE, fmin=FMIN,
                fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                duplicate_filters=DUPLICATE_FILTERS):
        """
        Creates a new LogarithmicFilterbank instance.

        :param bin_frequencies:   frequencies of the bins [Hz]
        :param bands:             number of filter bands per octave
        :param fmin:              the minimum frequency [Hz]
        :param fmax:              the maximum frequency [Hz]
        :param fref:              tuning frequency of the filterbank [Hz]
        :param norm_filters:      normalize the filters to area 1
        :param duplicate_filters: keep duplicate filters resulting from
                                  insufficient resolution of low frequencies

        Note: `bands` sets the number of bands per octave, not the overall
              number of filter bands; if set to 12 it results in a semitone
              filterbank.

        """
        # get a list of frequencies
        frequencies = log_frequencies(bands, fmin, fmax, fref)
        # convert to bins
        bins = frequencies2bins(frequencies, bin_frequencies)
        # get overlapping triangular filters
        filters = TriangularFilter.filters(bins, norm=norm_filters,
                                           duplicates=duplicate_filters,
                                           overlap=True)
        # create a LogarithmicFilterbank from the filters
        obj = cls.from_filters(filters, bin_frequencies)
        # set additional attributes
        obj.bands_per_octave = bands
        obj.fref = fref
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.bands_per_octave = getattr(obj, 'bands_per_octave',
                                        self.BANDS_PER_OCTAVE)
        self.fref = getattr(obj, 'fref', A4)

    def __reduce__(self):
        # needed for correct pickling
        # source: http://stackoverflow.com/questions/26598109/
        # get the parent's __reduce__ tuple
        pickled_state = super(LogarithmicFilterbank, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.bands_per_octave, self.fref,)
        # return a tuple that replaces the parent's __setstate__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for correct un-pickling
        # set the sample_rate
        self.bands_per_octave = state[-2]
        self.fref = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(LogarithmicFilterbank, self).__setstate__(state[0:-2])


# alias
LogFilterbank = LogarithmicFilterbank


# chroma / harmonic filterbanks
class SimpleChromaFilterbank(Filterbank):
    """
    A simple chroma filterbank based on a (semitone) filterbank.

    """
    BANDS = 12

    def __new__(cls, bin_frequencies, bands=BANDS, fmin=FMIN,
                fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                duplicate_filters=DUPLICATE_FILTERS):
        """
        Creates a new SimpleChromaFilterbank instance.

        :param bin_frequencies:   frequencies of the bins [Hz]
        :param bands:             number of filter bands per octave
        :param fmin:              the minimum frequency [Hz]
        :param fmax:              the maximum frequency [Hz]
        :param fref:              tuning frequency of the filterbank [Hz]
        :param norm_filters:      normalize the filters to area 1
        :param duplicate_filters: keep duplicate filters resulting from
                                  insufficient resolution of low frequencies

        """
        raise NotImplementedError("please check if produces correct/expected "
                                  "results and enable if yes.")
        # TODO: add comments!
        stf = LogarithmicFilterbank(bin_frequencies, bands=bands, fmin=fmin,
                                    fmax=fmax, fref=fref,
                                    norm_filters=norm_filters,
                                    duplicate_filters=duplicate_filters)
        # create an empty filterbank
        fb = np.empty((stf.shape[0], 12))
        spacing = np.arange(8) * 12
        for i in range(12):
            cur_spacing = spacing + i
            cur_spacing = cur_spacing[cur_spacing < stf.shape[1]]
            fb[:, i] = stf[:, cur_spacing].sum(1)
        # TODO: check if this should depend on the norm_filters parameter
        fb /= fb.sum(0)
        # cast to Filterbank
        obj = Filterbank.__new__(cls, fb, bin_frequencies)
        # set additional attributes
        obj.fref = fref
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.fref = getattr(obj, 'fref', A4)

    def __reduce__(self):
        # needed for correct pickling
        # source: http://stackoverflow.com/questions/26598109/
        # get the parent's __reduce__ tuple
        pickled_state = super(SimpleChromaFilterbank, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.fref,)
        # return a tuple that replaces the parent's __setstate__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for correct un-pickling
        # set the sample_rate
        self.fref = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(SimpleChromaFilterbank, self).__setstate__(state[0:-1])


class HarmonicFilterbank(Filterbank):
    """
    Harmonic filterbank class.

    """
    # Note: the last commit with the old harmonic filterbank stuff is
    #       8a73a0c8eec455928f241d3199309e075afe91c1
    #       https://jobim.ofai.at/gitlab/madmom/madmom/blob/
    #       8a73a0c8eec455928f241d3199309e075afe91c1/madmom/audio/filters.py

    def __new__(cls):
        """
        Creates a new HarmonicFilterbank instance.
        """
        raise NotImplementedError('please implement if needed!')


class PitchClassProfileFilterbank(Filterbank):
    """
    Filterbank for extracting pitch class profiles (PCP).

    "Realtime chord recognition of musical sound: a system using Common Lisp
     Music"
    T. Fujishima
    Proceedings of the International Computer Music Conference (ICMC 1999),
    Beijing, China

    """
    CLASSES = 12

    def __new__(cls, bin_frequencies, num_classes=CLASSES,
                fmin=FMIN, fmax=FMAX, fref=A4):
        """
        Creates a new PitchClassProfile (PCP) filterbank instance.

        :param bin_frequencies: frequencies of the bins [Hz]
        :param num_classes:     number of pitch classes
        :param fmin:            the minimum frequency [Hz]
        :param fmax:            the maximum frequency [Hz]
        :param fref:            reference frequency for the first PCP bin [Hz]

        """
        raise NotImplementedError("please check if produces correct/expected "
                                  "results and enable if yes.")
        # init a filterbank
        fb = np.zeros((len(bin_frequencies), num_classes))
        # log deviation from the reference frequency
        log_dev = np.log2(bin_frequencies / fref)
        # map the log deviation to the closets pitch class profiles
        num_class = np.round(num_classes * log_dev) % num_classes
        # define the pitch class profile filterbank
        # skip log_dev[0], since it is NaN
        fb[np.arange(1, len(bin_frequencies)), num_class.astype(int)[1:]] = 1
        # set all bins outside the allowed frequency range to 0
        fb[np.searchsorted(bin_frequencies, fmax, 'right'):] = 0
        fb[:np.searchsorted(bin_frequencies, fmin)] = 0
        # cast to Filterbank
        obj = Filterbank.__new__(cls, fb, bin_frequencies)
        # set additional attributes
        obj.fref = fref
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.fref = getattr(obj, 'fref', A4)

    def __reduce__(self):
        # needed for correct pickling
        # source: http://stackoverflow.com/questions/26598109/
        # get the parent's __reduce__ tuple
        pickled_state = super(PitchClassProfileFilterbank, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.fref,)
        # return a tuple that replaces the parent's __setstate__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for correct un-pickling
        # set the sample_rate
        self.fref = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(PitchClassProfileFilterbank, self).__setstate__(state[0:-1])


class HarmonicPitchClassProfileFilterbank(Filterbank):
    """
    Filterbank for extracting harmonic pitch class profiles (HPCP).

    "Tonal Description of Music Audio Signals"
    E. Gómez
    PhD thesis, Universitat Pompeu Fabra, Barcelona, Spain

    """
    FMIN = 100
    FMAX = 5000
    CLASSES = 36
    WINDOW = 4

    def __new__(cls, bin_frequencies, num_classes=CLASSES,
                fmin=FMIN, fmax=FMAX, fref=A4, window=WINDOW):
        """
        Creates a new HarmonicPitchClassProfile (HPCP) filterbank instance.

        :param bin_frequencies: frequencies of the bins [Hz]
        :param num_classes:     number of harmonic pitch classes
        :param fmin:            the minimum frequency [Hz]
        :param fmax:            the maximum frequency [Hz]
        :param fref:            reference frequency for the first HPCP bin [Hz]
        :param window:          length of the weighting window [bins]

        """
        raise NotImplementedError("please check if produces correct/expected "
                                  "results and enable if yes.")
        # init a filterbank
        fb = np.zeros((len(bin_frequencies), num_classes))
        # log deviation from the reference frequency
        log_dev = np.log2(bin_frequencies / fref)
        # map the log deviation to pitch class profiles
        num_class = (num_classes * log_dev) % num_classes
        # weight the bins
        for c in range(num_classes):
            # calculate the distance of the bins to the current class
            distance = num_class - c
            # unwrap
            distance[distance < -num_classes / 2.] += num_classes
            distance[distance > num_classes / 2.] -= num_classes
            # get all bins which are within the defined window
            idx = np.abs(distance) < window / 2.
            # apply the weighting function
            fb[idx, c] = np.cos((num_class[idx] - c) * np.pi / window) ** 2.
        # set all bins outside the allowed frequency range to 0
        fb[np.searchsorted(bin_frequencies, fmax, 'right'):] = 0
        fb[:np.searchsorted(bin_frequencies, fmin)] = 0
        # cast to Filterbank
        obj = Filterbank.__new__(cls, fb, bin_frequencies)
        # set additional attributes
        obj.fref = fref
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.fref = getattr(obj, 'fref', A4)

    def __reduce__(self):
        # needed for correct pickling
        # source: http://stackoverflow.com/questions/26598109/
        # get the parent's __reduce__ tuple
        pickled_state = super(HarmonicPitchClassProfileFilterbank,
                              self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.fref,)
        # return a tuple that replaces the parent's __setstate__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for correct un-pickling
        # set the sample_rate
        self.fref = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(HarmonicPitchClassProfileFilterbank,
              self).__setstate__(state[0:-1])


# comb filters
def feed_forward_comb_filter(signal, tau, alpha):
    """
    Filter the signal with a feed forward comb filter.

    :param signal: signal
    :param tau:    delay length
    :param alpha:  scaling factor
    :return:       comb filtered signal

    """
    # y[n] = x[n] + α * x[n - τ]
    if tau <= 0:
        raise ValueError('tau must be greater than 0')
    y = np.copy(signal)
    # add the delayed signal
    y[tau:] += alpha * signal[:-tau]
    # return
    return y


def _feed_backward_comb_filter(signal, tau, alpha):
    """
    Filter the signal with a feed backward comb filter.

    :param signal: signal
    :param tau:    delay length
    :param alpha:  scaling factor
    :return:       comb filtered signal

    Note: this function is extremely slow, use the faster cython version!

    """
    # y[n] = x[n] + α * y[n - τ]
    # Note: saw this formula somewhere, but it seems to produce less accurate
    #       tempo predictions...
    #       y[n, d] = (1. - alpha) * x[n, d] + alpha * y[n - tau, d]
    if tau <= 0:
        raise ValueError('tau must be greater than 0')
    y = np.copy(signal)
    # loop over the complete signal
    for n in range(tau, len(signal)):
        # add a delayed version of the output signal
        y[n] += alpha * y[n - tau]
    # return
    return y

try:
    from .comb_filters import feed_backward_comb_filter
except ImportError:
    import warnings
    warnings.warn('The feed_backward_comb_filter function will be extremely '
                  'slow! Please consider installing cython and build the '
                  'faster comb_filter module.')
    feed_backward_comb_filter = _feed_backward_comb_filter


def comb_filter(signal, filter_function, tau, alpha):
    """
    Filter the signal with a bank of either feed forward or backward comb
    filters.

    :param signal:          signal [numpy array]
    :param filter_function: filter function to use (feed forward or backward)
    :param tau:             delay length(s) [list / array of samples]
    :param alpha:           corresponding scaling factor(s)
                            [list / array of floats]
    :return:                comb filtered signal with the different taus
                            aligned along the (new) first dimension

    """
    # convert tau to a integer numpy array
    tau = np.asarray(tau, dtype=int)
    if tau.ndim != 1:
        raise ValueError('tau must be a 1D numpy array')
    # convert alpha to a numpy array
    alpha = np.asarray(alpha, dtype=float)
    if alpha.ndim != 1:
        raise ValueError('alpha must be a 1D numpy array')
    # alpha & tau must have the same size
    if tau.size != alpha.size:
        raise AssertionError('alpha & tau must have the same size')
    # determine output array size
    size = list(signal.shape)
    # add dimension of tau range size (new 1st dim)
    size.insert(0, len(tau))
    # init output array
    y = np.zeros(tuple(size))
    for i, t in np.ndenumerate(tau):
        y[i] = filter_function(signal, t, alpha[i])
    return y


class CombFilterbank(Processor):
    """
    CombFilterbank class.

    """

    def __init__(self, filter_function, tau, alpha):
        """
        Creates a new CombFilterbank.

        :param filter_function: comb filter function to use (either a function
                                or one of the literals {'forward', 'backward'})
        :param tau:             delay length(s) [int, list / array of samples]
        :param alpha:           corresponding scaling factor(s) [float, list /
                                array of floats]

        """
        # convert tau to a numpy array
        if isinstance(tau, int):
            self.tau = np.asarray([tau], dtype=int)
        elif isinstance(tau, (list, np.ndarray)):
            self.tau = np.asarray(tau, dtype=int)
        else:
            raise ValueError('tau must be cast-able as an int numpy array')

        # set the filter function
        if filter_function in ['forward', feed_forward_comb_filter]:
            self.comb_filter_function = feed_forward_comb_filter
        elif filter_function in ['backward', feed_backward_comb_filter]:
            self.comb_filter_function = feed_backward_comb_filter
        else:
            raise ValueError('wrong comb_filter type')

        # convert alpha to a numpy array
        if isinstance(alpha, (float, int)):
            self.alpha = np.asarray([alpha] * len(tau), dtype=float)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = np.asarray(alpha, dtype=float)
        else:
            raise ValueError('alpha must be cast-able as float numpy array')

    def process(self, data):
        """
        Filter the given data with the CombFilterbank.

        :param data: data to be filtered
        :return:     comb filtered data with the different taus aligned
                     along the (new) first dimension

        """
        # this method makes the Filterbank act as a Processor
        return comb_filter(data, self.comb_filter_function,
                           self.tau, self.alpha)
