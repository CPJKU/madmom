#!/usr/bin/env python
# encoding: utf-8
"""
This file contains filter and filterbank related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np


FILTER_DTYPE = np.float32
A4 = 440.


# Mel frequency scale
def hz2mel(f):
    """
    Convert Hz frequencies to Mel.

    :param f: input frequencies [Hz, numpy array]
    :return:  frequencies in Mel [Mel, numpy array]

    """
    return 1127.01048 * np.log(np.asarray(f) / 700. + 1.)


def mel2hz(m):
    """
    Convert Mel frequencies to Hz.

    :param m: input frequencies [Mel, numpy array]
    :return:  frequencies in Hz [Hz, numpy array]

    """
    return 700. * (np.exp(np.asarray(m) / 1127.01048) - 1.)


def mel_frequencies(num_bands, fmin, fmax):
    """
    Generates a list of frequencies aligned on the Mel scale.

    :param num_bands: number of bands [int]
    :param fmin:      the minimum frequency [Hz, float]
    :param fmax:      the maximum frequency [Hz, float]
    :return:          frequencies with Mel spacing [Hz, numpy array]

    """
    # convert fmin and fmax to the Mel scale and return an array of frequencies
    return mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), num_bands))


# Bark frequency scale
def hz2bark(f):
    """
    Convert Hz frequencies to Bark.

    :param f: input frequencies [Hz, numpy array]
    :return:  frequencies in Bark [Bark, numpy array]

    """
    raise NotImplementedError('please check this function, it produces '
                              'negative values')
    # TODO: use Zwicker's formula?
    #       return 13 * arctan(0.00076 * f) + 3.5 * arctan((f / 7500.) ** 2)
    return (26.81 / (1. + 1960. / np.asarray(f))) - 0.53


def bark2hz(z):
    """
    Convert Bark frequencies to Hz.

    :param z: input frequencies [Bark, numpy array]
    :return:  frequencies in Hz [Hz, numpy array]

    """
    raise NotImplementedError('please check this function, it produces weird '
                              'values')
    # TODO: use Zwicker's formula? what's the inverse of the above?
    return 1960. / (26.81 / (np.asarray(z) + 0.53) - 1.)


def bark_frequencies(fmin=20., fmax=15500.):
    """
    Generates a list of corner frequencies aligned on the Bark-scale.

    :param fmin: the minimum frequency [Hz, float]
    :param fmax: the maximum frequency [Hz, float]
    :return:     frequencies with Bark spacing [Hz, numpy array]

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


def bark_double_frequencies(fmin=20., fmax=15500.):
    """
    Generates a list of corner frequencies aligned on the Bark-scale.
    The list includes also center frequencies between the corner frequencies.

    :param fmin: the minimum frequency [Hz, float]
    :param fmax: the maximum frequency [Hz, float]
    :return:     frequencies with Bark spacing [Hz, numpy array]

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

    :param bands_per_octave: number of filter bands per octave [int]
    :param fmin:             the minimum frequency [Hz, float]
    :param fmax:             the maximum frequency [Hz, float]
    :param fref:             tuning frequency [Hz, float]
    :return:                 logarithmically spaced frequencies
                             [Hz, numpy array]

    Note: If 12 bands per octave and a4=440 are used, the frequencies are
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

    :param fmin: the minimum frequency [Hz, float]
    :param fmax: the maximum frequency [Hz, float]
    :param a4:   tuning frequency of A4 [Hz, float]
    :return:     semitones frequencies [Hz, numpy array]

    """
    # return MIDI frequencies
    return log_frequencies(12, fmin, fmax, a4)


# MIDI
def hz2midi(f, fref=A4):
    """
    Convert frequencies to the corresponding MIDI notes.

    :param f:    input frequencies [Hz, numpy array]
    :param fref: tuning frequency of A4 [Hz, float]
    :return:     MIDI notes [numpy array]

    For details see: at http://www.phys.unsw.edu.au/jw/notes.html

    Note: This function does not necessarily return a valid MIDI Note, you may
          need to round it to the nearest integer.

    """
    return (12. * np.log2(np.asarray(f, dtype=np.float) / fref)) + 69.


def midi2hz(m, fref=A4):
    """
    Convert MIDI notes to corresponding frequencies.

    :param m:    input MIDI notes [numpy array]
    :param fref: tuning frequency of A4 [Hz, float]
    :return:     frequencies in Hz [Hz, numpy array]

    For details see: http://www.phys.unsw.edu.au/jw/notes.html

    """
    return 2. ** ((np.asarray(m, dtype=np.float) - 69.) / 12.) * fref


# provide an alias to semitone_frequencies
midi_frequencies = semitone_frequencies


# ERB frequency scale
def hz2erb(f):
    """
    Convert Hz to ERB.

    :param f: input frequencies [Hz, numpy array]
    :return:  frequencies in ERB [ERB, numpy array]

    Information about the ERB scale can be found at:
    https://ccrma.stanford.edu/~jos/bbt/Equivalent_Rectangular_Bandwidth.html

    """
    return 21.4 * np.log10(1 + 4.37 * np.asarray(f) / 1000.)


def erb2hz(e):
    """
    Convert ERB scaled frequencies to Hz.

    :param e: input frequencies [ERB, numpy array]
    :return:  frequencies in Hz [Hz, numpy array]

    Information about the ERB scale can be found at:
    https://ccrma.stanford.edu/~jos/bbt/Equivalent_Rectangular_Bandwidth.html

    """
    return (10. ** (np.asarray(e) / 21.4) - 1.) * 1000. / 4.37


# helper functions for filter creation
def frequencies2bins(frequencies, bin_frequencies, unique_bins=False):
    """
    Map frequencies to the closest corresponding bins.

    :param frequencies:     list with frequencies [Hz, numpy array]
    :param bin_frequencies: frequencies of the bins [Hz, numpy array]
    :param unique_bins:     return only unique bins, i.e. remove all duplicate
                            bins resulting from insufficient resolution at low
                            frequencies [bool]
    :return:                corresponding (unique) bins [numpy array]

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
    # Note: this can be important to do so, otherwise the lower frequency
    #       bins can be given too much weight if simply summed up (as in
    #       the spectral flux)
    if unique_bins:
        indices = np.unique(indices)
    # return the (unique) bin indices of the closest matches
    return indices


def bins2frequencies(bins, bin_frequencies):
    """
    Convert bins to the corresponding frequencies.

    :param bins:            (a list of) bins [list or numpy array]
    :param bin_frequencies: frequencies of the bins [Hz, numpy array]
    :return:                corresponding frequencies [Hz, numpy array]

    """
    # map the frequencies to spectrogram bins
    return np.asarray(bin_frequencies, dtype=np.float)[np.asarray(bins)]


# filter classes
class Filter(np.ndarray):
    """
    Generic filter class.

    """

    def __new__(cls, data, start=0, norm=False):
        """
        Creates a new Filter instance.

        :param data:  1D numpy array
        :param start: start position [int]
        :param norm:  normalize the filter area to 1 [bool]

        The start position is mandatory if this Filter should be used for the
        creation of a Filterbank. If not set, a start position of 0 is assumed.

        """
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

    def __reduce__(self):
        # needed for correct pickling
        # source: http://stackoverflow.com/questions/26598109/
        # get the parent's __reduce__ tuple
        pickled_state = super(Filter, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.start, self.stop,)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # needed for correct un-pickling
        # set the start and stop bins
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
        :return:       bins and normalization information for filter creation

        """
        raise NotImplementedError('needs to be implemented by sub-classes')

    @classmethod
    def filters(cls, bins, norm, **kwargs):
        """
        Creates a list with filters for the the given bins.

        :param bins:   center/crossover bins of filters [list or numpy array]
        :param norm:   normalize the area of the filter(s) to 1 [bool]
        :param kwargs: additional parameters passed to band_bins()
        :return:       list with filters

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

    """

    def __new__(cls, start, center, stop, norm=False):
        """
        Creates a new TriangularFilter instance.

        :param start:  start bin [int]
        :param center: center bin [int]
        :param stop:   stop bin [int]
        :param norm:   normalize the area of the filter to 1 [bool]
        :return:       triangular shaped filter with length `stop`, height 1
                       (unless normalized) with indices <= `start` set to 0

        """
        # center must be between start & stop
        if not start <= center < stop:
            raise ValueError('center must be between start and stop')
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

    def __reduce__(self):
        # get the parent's __reduce__ tuple
        pickled_state = super(TriangularFilter, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.center,)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # in addition to the start and stop bins, also set the center bin
        self.center = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(TriangularFilter, self).__setstate__(state[0:-1])

    @classmethod
    def band_bins(cls, bins, overlap=True):
        """
        Yields start, center and stop bins for creation of triangular filters.

        :param bins:    center bins of filters [list or numpy array]
        :param overlap: filters should overlap [bool]
        :return:        start, center and stop bins

        Note: If `overlap` is 'False', the `start` and `stop` bins of the
              filters are interpolated between the centre bins, normal rounding
              applies.

        """
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

    """

    def __new__(cls, start, stop, norm=False):
        """
        Creates a new RectangularFilter instance.

        :param start: start bin of the filter [int]
        :param stop:  stop bin of the filter [int]
        :param norm:  normalize the area of the filter to 1 [bool]
        :return:      rectangular shaped filter with length `stop`, height 1
                      (unless normalized) with indices <= `start` set to 0

        """
        # start must be smaller than stop
        if start >= stop:
            raise ValueError('start must be smaller than stop')
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

        :param bins:    crossover bins of filters [numpy array]
        :param overlap: filters should overlap [bool]
        :return:        start and stop bins

        """
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

    A Filterbank has a shape of (num_bins x num_bands) and can be used to
    filter a spectrogram of shape (num_frames x num_bins) to (num_frames x
    num_bands).

    """
    fref = None

    def __new__(cls, data, bin_frequencies):
        """
        Creates a new Filterbank instance.

        :param data:            2D numpy array (num_bins x num_bands)
        :param bin_frequencies: frequencies of the bins [numpy array]
                                (length must be equal to the first dimension
                                 of the given data array)

        """
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

    def __reduce__(self):
        # get the parent's __reduce__ tuple
        pickled_state = super(Filterbank, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.bin_frequencies,)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # set the bin frequencies
        self.bin_frequencies = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(Filterbank, self).__setstate__(state[0:-1])

    @classmethod
    def _put_filter(cls, filt, band):
        """
        Puts a filter in the band, internal helper function.

        :param filt: filter to be put into the band [Filter]
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

        :param filters:         list of Filters (per band)
                                if multiple filters per band are desired, they
                                should be also contained in a list, resulting
                                in a list of lists of Filters
        :param bin_frequencies: frequencies of the bins [numpy array]
                                (needed to determine the expected size of the
                                filterbank)
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


class MelFilterbank(Filterbank):
    """
    Mel filterbank class.

    """
    NUM_BANDS = 40
    FMIN = 20.
    FMAX = 17000.
    NORM_FILTERS = True
    UNIQUE_FILTERS = True

    def __new__(cls, bin_frequencies, num_bands=NUM_BANDS, fmin=FMIN,
                fmax=FMAX, norm_filters=NORM_FILTERS,
                unique_filters=UNIQUE_FILTERS, **kwargs):
        """
        Creates a new MelFilterbank instance.

        :param bin_frequencies: frequencies of the bins [Hz, float array]
        :param num_bands:       number of filter bands [int]
        :param fmin:            the minimum frequency [Hz, float]
        :param fmax:            the maximum frequency [Hz, float]
        :param norm_filters:    normalize the filters to area 1 [bool]
        :param unique_filters:  keep only unique filters, i.e. remove duplicate
                                filters resulting from insufficient resolution
                                at low frequencies [bool]

        Note: Because of rounding and mapping of frequencies to bins and back
              to frequencies, the actual minimum, maximum and center
              frequencies do not necessarily match the arguments given.

        """
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


class BarkFilterbank(Filterbank):
    """
    Bark filterbank class.

    """
    FMIN = 20.
    FMAX = 15500.
    NUM_BANDS = 'normal'
    NORM_FILTERS = True
    UNIQUE_FILTERS = True

    def __new__(cls, bin_frequencies, num_bands=NUM_BANDS, fmin=FMIN,
                fmax=FMAX, norm_filters=NORM_FILTERS,
                unique_filters=UNIQUE_FILTERS, **kwargs):
        """
        Creates a new BarkFilterbank instance.

        :param bin_frequencies: frequencies of the bins [Hz, float array]
        :param num_bands:       number of filter bands [int]
        :param fmin:            the minimum frequency [Hz, float]
        :param fmax:            the maximum frequency [Hz, float]
        :param norm_filters:    normalize the filters to area 1 [bool]
        :param unique_filters:  keep only unique filters, i.e. remove duplicate
                                filters resulting from insufficient resolution
                                at low frequencies [bool]

        """
        # get a list of frequencies
        if num_bands == 'normal':
            frequencies = bark_frequencies(fmin, fmax)
        elif num_bands == 'double':
            frequencies = bark_double_frequencies(fmin, fmax)
        else:
            raise ValueError("`num_bands` must be {'normal', 'double'}")
        # convert to bins
        bins = frequencies2bins(frequencies, bin_frequencies,
                                unique_bins=not unique_filters)
        # get non-overlapping rectangular filters
        filters = RectangularFilter.filters(bins, norm=norm_filters,
                                            overlap=False)
        # create a BarkFilterbank from the filters
        return cls.from_filters(filters, bin_frequencies)


class LogarithmicFilterbank(Filterbank):
    """
    Logarithmic filterbank class.

    """
    NUM_BANDS_PER_OCTAVE = 12

    def __new__(cls, bin_frequencies, num_bands=NUM_BANDS_PER_OCTAVE,
                fmin=FMIN, fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                unique_filters=UNIQUE_FILTERS, bands_per_octave=True):
        """
        Creates a new LogarithmicFilterbank instance.

        :param bin_frequencies:  frequencies of the bins [Hz, float array]
        :param num_bands:        number of filter bands (per octave) [int]
        :param fmin:             the minimum frequency [Hz, float]
        :param fmax:             the maximum frequency [Hz, float]
        :param fref:             tuning frequency of the filterbank [Hz, float]
        :param norm_filters:     normalize the filters to area 1 [bool]
        :param unique_filters:   keep only unique filters, i.e. remove
                                 duplicate filters resulting from insufficient
                                 resolution at low frequencies [bool]
        :param bands_per_octave: indicates whether `num_bands` is given as
                                 number of bands per octave ('True') or as an
                                 absolute number of bands ('False') [bool]

        Note: `num_bands` sets either the number of bands per octave or the
              total number of bands, depending on the setting of
              `bands_per_octave`. `num_bands` is used to set also the number of
              bands per octave to keep the argument for all classes the same.
              If 12 bands per octave are used, a filterbank with semitone
              spacing is created.

        """
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

    def __reduce__(self):
        # get the parent's __reduce__ tuple
        pickled_state = super(LogarithmicFilterbank, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.num_bands_per_octave, self.fref)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # set the number of bands per octave and reference frequency
        self.num_bands_per_octave = state[-2]
        self.fref = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(LogarithmicFilterbank, self).__setstate__(state[0:-2])


# alias
LogFilterbank = LogarithmicFilterbank


class RectangularFilterbank(Filterbank):
    """
    Rectangular filterbank class.

    """

    def __new__(cls, bin_frequencies, crossover_frequencies, fmin=FMIN,
                fmax=FMAX, norm_filters=NORM_FILTERS,
                unique_filters=UNIQUE_FILTERS):
        """
        Creates a new LogarithmicFilterbank instance.

        :param bin_frequencies:       frequencies of the bins [Hz, float array]
        :param crossover_frequencies: crossover frequencies of the bands
                                      [Hz, list or array of floats]
        :param fmin:                  the minimum frequency [Hz, float]
        :param fmax:                  the maximum frequency [Hz, float]
        :param norm_filters:          normalize the filters to area 1 [bool]
        :param unique_filters:        keep only unique filters, i.e. remove
                                      duplicate filters resulting from
                                      insufficient resolution at low
                                      frequencies [bool]

        """
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

    def __reduce__(self):
        # get the parent's __reduce__ tuple
        pickled_state = super(RectangularFilterbank, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.crossover_frequencies, )
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # set the additional attributes
        self.crossover_frequencies = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(RectangularFilterbank, self).__setstate__(state[0:-1])


# chroma / harmonic filterbanks
class SimpleChromaFilterbank(Filterbank):
    """
    A simple chroma filterbank based on a (semitone) filterbank.

    """
    NUM_BANDS = 12

    def __new__(cls, bin_frequencies, num_bands=NUM_BANDS, fmin=FMIN,
                fmax=FMAX, fref=A4, norm_filters=NORM_FILTERS,
                unique_filters=UNIQUE_FILTERS):
        """
        Creates a new SimpleChromaFilterbank instance.

        :param bin_frequencies: frequencies of the bins [Hz, float array]
        :param num_bands:       number of filter bands per octave [int]
        :param fmin:            the minimum frequency [Hz, float]
        :param fmax:            the maximum frequency [Hz, float]
        :param fref:            tuning frequency of the filterbank [Hz, float]
        :param norm_filters:    normalize the filters to area 1 [bool]
        :param unique_filters:  keep only unique filters, i.e. remove duplicate
                                filters resulting from insufficient resolution
                                at low frequencies [bool]
        """
        raise NotImplementedError("please check if produces correct/expected "
                                  "results and enable if yes.")
        # TODO: add comments!
        stf = LogFilterbank(bin_frequencies, num_bands=num_bands, fmin=fmin,
                            fmax=fmax, fref=fref, norm_filters=norm_filters,
                            unique_filters=unique_filters)
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
        # get the parent's __reduce__ tuple
        pickled_state = super(SimpleChromaFilterbank, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.fref,)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # set the reference frequency
        self.fref = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(SimpleChromaFilterbank, self).__setstate__(state[0:-1])


class HarmonicFilterbank(Filterbank):
    """
    Harmonic filterbank class.

    """
    # Note: old code: https://jobim.ofai.at/gitlab/madmom/madmom/snippets/1

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
    FMIN = 100
    FMAX = 5000

    def __new__(cls, bin_frequencies, num_classes=CLASSES, fmin=FMIN,
                fmax=FMAX, fref=A4):
        """
        Creates a new PitchClassProfile (PCP) filterbank instance.

        :param bin_frequencies: frequencies of the bins [Hz]
        :param num_classes:     number of pitch classes
        :param fmin:            the minimum frequency [Hz]
        :param fmax:            the maximum frequency [Hz]
        :param fref:            reference frequency for the first PCP bin [Hz]

        """
        # init a filterbank
        fb = np.zeros((len(bin_frequencies), num_classes))
        # log deviation from the reference frequency
        log_dev = np.log2(bin_frequencies / fref)
        # map the log deviation to the closest pitch class profiles
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
        # get the parent's __reduce__ tuple
        pickled_state = super(PitchClassProfileFilterbank, self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.fref,)
        # return a tuple that replaces the parent's __reduce__ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # set the reference frequency
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
    CLASSES = 36
    FMIN = 100
    FMAX = 5000
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
        # get the parent's __reduce__ tuple
        pickled_state = super(HarmonicPitchClassProfileFilterbank,
                              self).__reduce__()
        # create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.fref,)
        # return a tuple that replaces the parent's __reduce____ tuple
        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        # set the reference frequency
        self.fref = state[-1]
        # call the parent's __setstate__ with the other tuple elements
        super(HarmonicPitchClassProfileFilterbank,
              self).__setstate__(state[0:-1])
