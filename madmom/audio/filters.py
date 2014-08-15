#!/usr/bin/env python
# encoding: utf-8
"""
This file contains filter and filterbank related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np

from .signal import segment_axis

# default values for filters
FMIN = 30
FMAX = 17000
BANDS_PER_OCTAVE = 12
NORM_FILTERS = True
DUPLICATE_FILTERS = False
OVERLAP_FILTERS = True
A4 = 440


# Mel frequency scale
def hz2mel(f):
    """
    Convert Hz frequencies to Mel.

    :param f: input frequencies [Hz]
    :return:  frequencies in Mel

    """
    return 1127.01048 * np.log(f / 700. + 1.)


def mel2hz(m):
    """
    Convert Mel frequencies to Hz.

    :param m: input frequencies [Mel]
    :return:  frequencies in Hz

    """
    return 700. * (np.exp(m / 1127.01048) - 1.)


def mel_frequencies(num_bands, fmin, fmax):
    """
    Generates a list of frequencies aligned on the Mel scale.

    :param num_bands: number of bands
    :param fmin:      the minimum frequency [Hz]
    :param fmax:      the maximum frequency [Hz]
    :return:          a list of frequencies

    """
    # convert fmin and fmax to the Mel scale and return a list of frequencies
    return mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), num_bands))


# Bark frequency scale
def hz2bark(f):
    """
    Convert Hz frequencies to Bark.

    :param f: input frequencies [Hz]
    :return:  frequencies in Bark.

    """
    # TODO: use Zwicker's formula?
    # return 13. * arctan(0.00076*f) + 3.5 * arctan((f/7500.)**2)
    return (26.81 / (1. + 1960. / f)) - 0.53


def bark2hz(z):
    """
    Convert Bark frequencies to Hz.

    :param z: input frequencies [Bark]
    :return:  frequencies in Hz.

    """
    # TODO: use Zwicker's formula?
    # return 13. * arctan(0.00076*f) + 3.5 * arctan((f/7500.)**2)
    return 1960. / (26.81 / (z + 0.53) - 1.)


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
def log_frequencies(bands_per_octave, fmin, fmax, a4=A4):
    """
    Generates a list of frequencies aligned on a logarithmic frequency scale.

    :param bands_per_octave: number of filter bands per octave
    :param fmin:             the minimum frequency [Hz]
    :param fmax:             the maximum frequency [Hz]
    :param a4:               tuning frequency of A4 [Hz]
    :return:                 a list of frequencies

    Note: if 12 bands per octave and a4=440 are used, the frequencies are
          equivalent to MIDI notes.

    """
    # get the range
    left = np.floor(np.log2(float(fmin) / a4) * bands_per_octave)
    right = np.ceil(np.log2(float(fmax) / a4) * bands_per_octave)
    # generate frequencies
    frequencies = a4 * 2. ** (np.arange(left, right) / float(bands_per_octave))
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
def midi2hz(m, a4=A4):
    """
    Convert frequencies to the corresponding MIDI notes.

    :param m:  input MIDI notes
    :param a4: tuning frequency of A4 [Hz]
    :return:   frequencies in Hz

    For details see: http://www.phys.unsw.edu.au/jw/notes.html

    """
    return 2. ** ((m - 69.) / 12.) * a4


def hz2midi(f, a4=A4):
    """
    Convert MIDI notes to corresponding frequencies.

    :param f:  input frequencies [Hz]
    :param a4: tuning frequency of A4 [Hz]
    :return:   MIDI notes

    For details see: at http://www.phys.unsw.edu.au/jw/notes.html

    Note: This function does not necessarily return a valid MIDI Note, you may
          need to round it to the nearest integer.

    """
    return (12. * np.log2(f / float(a4))) + 69.


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
    return 21.4 * np.log10(1 + 4.37 * f / 1000.)


def erb2hz(e):
    """
    Convert ERB scaled frequencies to Hz.

    :param e: input frequencies [ERB]
    :return:  frequencies in Hz

    Information about the ERB scale can be found at:
    https://ccrma.stanford.edu/~jos/bbt/Equivalent_Rectangular_Bandwidth.html

    """
    return (10. ** (e / 21.4) - 1.) * 1000. / 4.37


# helper functions for filter creation
def fft_freqs(num_fft_bins, sample_rate):
    """
    Frequencies of the FFT bins.

    :param num_fft_bins: number of FFT bins (= half the FFT size)
    :param sample_rate:  sample rate of the signal
    :return:             corresponding FFT bin frequencies

    """
    # slower: np.fft.fftfreq(num_fft_bins * 2)[:num_fft_bins] * sample_rate
    return np.linspace(0, sample_rate / 2., num_fft_bins + 1)[:-1]


def frequencies2bins(frequencies, num_fft_bins, sample_rate):
    """
    Convert frequencies to the corresponding bins.


    :param frequencies:  a list of frequencies [Hz]
    :param num_fft_bins: number of FFT bins (= half the FFT size)
    :param sample_rate:  sample rate of the audio signal [Hz]
    :return:             corresponding bins

    """
    # map the frequencies to spectrogram bins
    factor = (sample_rate / 2.0) / num_fft_bins
    bins = np.round(np.asarray(frequencies) / factor).astype(int)
    # return all bins within the valid range
    return bins[:np.searchsorted(bins, num_fft_bins)]


def bins2frequencies(bins, num_fft_bins, sample_rate):
    """
    Convert bins to the corresponding frequencies.


    :param bins:         a list of bins
    :param num_fft_bins: number of FFT bins (= half the FFT size)
    :param sample_rate:  sample rate of the audio signal [Hz]
    :return:             corresponding frequencies [Hz]

    """
    # map the frequencies to spectrogram bins
    factor = (sample_rate / 2.0) / num_fft_bins
    return np.asarray(bins) * factor


# filter classes
class Filter(np.ndarray):
    """
    Generic Filter class.

    """
    def __new__(cls, data, start=0):
        """
        Creates a new Filter.

        :param data:  1D numpy array
        :param start: start position

        The start position is mandatory if this Filter should be used for the
        creation of a Filterbank. If not set, a start position of o is assumed.

        """
        # input is an numpy ndarray instance
        if isinstance(data, np.ndarray):
            # cast as Filterbank
            obj = np.asarray(data).view(cls)
        else:
            raise TypeError('wrong input data for Filter, must be np.ndarray')
        # right now, allow only 1D
        if data.ndim != 1:
            raise NotImplementedError('please add multi-dimension support')
        # set attributes
        obj.start = start
        obj.stop = start + len(data)
        # return the object
        return obj

    @classmethod
    def band_bins(cls, bins, **kwargs):
        """
        Must yields the centre/crossover bins needed for filter creation.

        :param bins:   centre/crossover bins of filters [list or numpy array]
        :param kwargs: additional parameters
        :return:       bins and normalisation information for filter creation

        """
        raise NotImplementedError('needs to be implemented by sub-classes')

    @classmethod
    def filters(cls, bins, **kwargs):
        """
        Creates a list with filters for the the given bins.

        :param bins:   (centre/crossover) bins of filters [list or numpy array]
        :param kwargs: additional parameters passed to band_bins()
        :return:       list with filters

        """
        # generate a list of filters for the given centre/crossover bins
        filters = []
        for filter_args in cls.band_bins(bins, **kwargs):
            # create a filter and append it to the list
            filters.append(cls(*filter_args))
        # return the filters
        return filters


class TriangularFilter(Filter):
    """
    Triangular Filter.

    """
    DUPLICATES = False
    NORM = True
    OVERLAP = True

    def __new__(cls, start, centre, stop, norm=NORM):
        """
        Create a triangular filter.

        :param start:  start bin
        :param centre: centre bin (of height 1, unless filter is normalised).
        :param stop:   stop bin
        :param norm:   normalise the area of the filter(s) to 1 [bool]
        :return:       a triangular shaped filter with length 'stop', height 1
                       (unless normalised) with indices <= 'start' set to 0

        """
        # center must be within start & stop
        if start >= centre >= stop:
            raise ValueError('center must be within start and stop')
        # make centre and stop relative
        centre -= start
        stop -= start
        # set the height of the filter, normalised if necessary.
        # A standard filter is at least 3 bins wide, and stop - start = 2
        # thus the filter has an area of 1 if normalised this way
        height = 2. / stop if norm else 1.
        # create filter
        data = np.zeros(stop)
        # rising edge (without the centre)
        data[:centre] = np.linspace(0, height, centre, endpoint=False)
        # falling edge (including the centre, but without the last bin)
        data[centre:] = np.linspace(height, 0, stop - centre, endpoint=False)
        # cast to TriangularFilter
        obj = Filter.__new__(cls, data, start)
        # set the centre bin
        obj.centre = start + centre
        # return the filter
        return obj

    @classmethod
    def band_bins(cls, bins, norm=NORM, duplicates=DUPLICATES,
                  overlap=OVERLAP):
        """
        Yields start, centre and stop bins and normalisation info for creation
        of triangular filters.

        :param bins:       centre bins of filters [list or numpy array]
        :param norm:       normalise the area of the filter(s) to 1 [bool]
        :param duplicates: keep duplicate filters resulting from insufficient
                           resolution of low frequencies [bool]
        :param overlap:    filters should overlap [bool]
        :return:           start, centre and stop bins & normalisation info

        """
        # only keep unique bins if requested
        # Note: this can be important to do so, otherwise the lower frequency
        #       bins can be given too much weight if simply summed up (as in
        #       the spectral flux)
        if not duplicates:
            bins = np.unique(bins)
        # make sure enough frequencies are given
        if len(bins) < 3:
            raise ValueError('Cannot create filterbank with less than 1 band')
        # return the frequencies
        for start, center, stop in segment_axis(bins, 3, 1):
            # create non-overlapping filters
            if not overlap:
                # re-arrange the start and stop positions
                start = np.round(float(center + start) / 2)
                stop = np.round(float(center + stop) / 2)
            # consistently handle too-small filters
            if duplicates and (stop - start < 2):
                center = start
                stop = start + 1
            # yield the frequencies and continue
            yield start, center, stop, norm


class RectangularFilter(Filter):
    """
    Rectangular Filter.

    """
    DUPLICATES = False
    NORM = True
    OVERLAP = False

    def __new__(cls, start, stop, norm=NORM):
        """
        Create a rectangular filter.

        :param start: start bin of the filter
        :param stop:  stop bin of the filter
        :param norm:  normalise the area of the filter(s) to 1 [bool]
        :return:      a rectangular shaped filter with length 'stop', height 1
                      (unless normalised) with indices <= 'start' set to 0

        """
        # center must be within start & stop
        if start >= stop:
            raise ValueError('start must be smaller than stop')
        # make stop relative
        stop -= start
        # set the height of the filter, normalised if necessary
        height = 1. / stop if norm else 1.
        # create filter
        data = np.ones(stop) * height
        # cast to RectangularFilter and return it
        return Filter.__new__(cls, data, start)

    @classmethod
    def band_bins(cls, bins, norm=NORM, duplicates=DUPLICATES,
                  overlap=OVERLAP):
        """
        Yields start and stop bins and normalisation info for creation of
        rectangular filters.

        :param bins:       crossover bins of filters [numpy array]
        :param norm:       normalise the area of the filter(s) to 1 [bool]
        :param duplicates: keep duplicate filters resulting from insufficient
                           resolution of low frequencies [bool]
        :param overlap:    filters should overlap [bool]
        :return:           start and stop bins & normalisation info

        """
        # only keep unique bins if requested
        # Note: this can be important to do so, otherwise the lower frequency
        #       bins can be given too much weight if simply summed up (as in
        #       the spectral flux)
        if not duplicates:
            bins = np.unique(bins)
        # make sure enough frequencies are given
        if len(bins) < 2:
            raise ValueError('Cannot create filterbank with less than 1 band')
        # overlapping filters?
        if overlap:
            raise NotImplementedError('please implement if needed!')
        # return the frequencies
        for start, stop in segment_axis(bins, 2, 1):
            # yield the frequencies and continue
            yield start, stop, norm


# TODO: if someone needs this code, please adapt the harmonic_filterbank stuff
#       as a new FilterType and then delete all the commented stuff below.
#
# HARMONIC_ENVELOPE = lambda x: np.sqrt(1. / x)
# HARMONIC_WIDTH = lambda x: 50 * 1.1 ** x
# INHARMONICITY_COEFF = 0.0
#
# # actual filter
# Filter = namedtuple('Filter', ['filter', 'start_pos'])
#
#
# def _put_filter(filt, band):
#     """
#     Puts a filter in the band, internal helper function.
#
#     :param filt: filter as named tuple "Filter"
#     :param band: band in which the filter should be put (numpy array)
#
#     """
#     start = filt.start_pos
#     stop = start + len(filt.filter)
#     filter_ = filt.filter
#     # truncate the filter if it starts before the 0th frequency bin
#     if start < 0:
#         filter_ = filter_[-start:]
#         start = 0
#     # truncate the filter if it ends after the last frequency bin
#     if stop > len(band):
#         filter_ = filter_[:stop - len(band)]
#         stop = len(band)
#     # put the filter in place
#     filter_pos = band[start:stop]
#     # TODO: if needed, allow other handling (like adding values)
#     np.maximum(filter_, filter_pos, out=filter_pos)
#
#
# def assemble_filterbank(filters, num_fft_bins, norm):
#     """
#     Creates a filterbank with possibly multiple filters per band.
#
#     :param filters:      list containing the filters per band; if multiple
#                          filters per band are desired, they should be also
#                          contained in a list, resulting in a list of lists of
#                          filters. a filter is represented by the named tuple
#                          "Filter"
#     :param num_fft_bins: number of FFT bins (= half the FFT size)
#     :param norm:         normalise the area of each filter band to 1 [bool]
#     :return:             filterbank with respective filter elements
#
#     """
#     # create filterbank
#     fb = np.zeros((num_fft_bins, len(filters)))
#     # iterate over all filters
#     for band_id, band_filter in enumerate(filters):
#         band = fb[:, band_id]
#         # if there's a list of filters for the current band, put them all
#         if type(band_filter) is list:
#             for filt in band_filter:
#                 _put_filter(filt, band)
#         else:
#             _put_filter(band_filter, band)
#     # normalise filterbank
#     if norm:
#         fb /= fb.sum(axis=0)
#     # return filterbank
#     return fb
#
#
# def band_bins(center_bins, duplicates, overlap):
#     """
#     Yields start, center and stop frequencies for filters.
#
#     :param center_bins: center bins of filters [numpy array]
#     :param duplicates:  keep duplicate filters resulting from insufficient
#                         resolution of low frequencies [bool]
#     :param overlap:     filters should overlap [bool]
#     :return:            start, center and stop frequencies for filters
#
#     """
#     # only keep unique bins if requested
#     # Note: this can be important to do so, otherwise the lower frequency bins
#     # are given too much weight if simply summed up (as in the spectral flux)
#     if not duplicates:
#         center_bins = np.unique(center_bins)
#     # make sure enough frequencies are given
#     if len(center_bins) < 3:
#         raise ValueError('Cannot create filterbank with less than 1 band')
#     # return the frequencies
#     for start, center, stop in segment_axis(center_bins, 3, 1):
#         # create non-overlapping filters
#         if not overlap:
#             # re-arrange the start and stop positions
#             start = np.round(float(center + start) / 2)
#             stop = np.round(float(center + stop) / 2)
#         # consistently handle too-small filters
#         if duplicates and (stop - start < 2):
#             center = start
#             stop = start + 1
#         # yield the frequencies and continue
#         yield start, center, stop
#
#
# # filterbank creation functions
# def filterbank(filter_type, frequencies, num_fft_bins, sample_rate,
#                norm=NORM_FILTERS, duplicates=DUPLICATE_FILTERS,
#                overlap=OVERLAP_FILTERS):
#     """
#     Creates a filterbank with one filter per band.
#
#     :param filter_type:  function that creates a filter and thus define its
#                          shape. the function must return a numpy array. the
#                          following parameters will be passed to this function:
#                          - width:  filter width [bins]
#                          - center: filter center position (< width) [bin]
#                          - norm:   normalise the filter (sum=1) or not [bool]
#                          Examples: triangular_filter, rectangular_filter
#     :param frequencies:  a list of frequencies used for filter creation [Hz]
#     :param num_fft_bins: number of FFT bins (= half the FFT size)
#     :param sample_rate:  sample rate of the audio signal [Hz]
#     :param norm:         normalise the area of the filters to 1 [bool]
#     :param duplicates:   keep duplicate filters resulting from insufficient
#                          resolution of low frequencies [bool]
#     :param overlap:      filters should overlap [bool]
#     :return:             filterbank
#
#     """
#     # map the frequencies to the spectrogram bins
#     bins = frequencies2bins(frequencies, num_fft_bins, sample_rate)
#     # create filterbank
#     filters = []
#     # get (overlapping) start, center and stop frequencies from a list of bins
#     for start, center, stop in band_bins(bins, duplicates, overlap):
#         kwargs = {'width': stop - start,
#                   'center': center - start,
#                   'norm': norm}
#         # create a filter of filter_type with the given arguments
#         filters.append(FilterType(filter_type(**kwargs), start))
#     # create and return the filterbank
#     # Note: no normalisation here, since each filter is already normalised
#     return assemble_filterbank(filters, num_fft_bins, norm=False)
#
#
# def harmonic_filterbank(filter_type, fundamentals, num_harmonics, num_fft_bins,
#                         sample_rate, harmonic_envelope=HARMONIC_ENVELOPE,
#                         harmonic_width=HARMONIC_WIDTH,
#                         inharmonicity_coeff=INHARMONICITY_COEFF):
#     """
#     Creates a filterbank in which each band represents a fundamental frequency
#     and its harmonics.
#
#     :param filter_type:         function that creates a filter. the function
#                                 must return a numpy array. the following
#                                 parameters will be passed to this function:
#                                 - width:  filter width [bins]
#                                 - center: filter center position (< width)
#                                 - norm:   boolean indicating whether to
#                                           normalise the filter (sum=1) or not
#     :param fundamentals:        list of fundamental frequencies
#     :param num_harmonics:       number of harmonics for each fundamental freq.
#     :param num_fft_bins:        number of FFT bins (= half the FFT size)
#     :param sample_rate:         sample rate of the audio signal [Hz]
#     :param harmonic_envelope:   function returning a weight for each harmonic
#                                 and the f0. [default=lambda x: np.sqrt(1. / x)]
#     :param harmonic_width:      function returning the width for each harmonic
#                                 and the f0. [default=50 * 1.1 ** x]
#     :param inharmonicity_coeff: coefficient for calculating the drift of
#                                 harmonics for not perfectly harmonic
#                                 instruments
#     :return:                    harmonic filterbank
#
#     Note: harmonic_envelope and harmonic_width must accept a numpy array of
#           the harmonic ids, where the fundamental's id is 1, the second
#           harmonic is 2, etc...
#
#           TODO: inharmonicity_coeff should depend on the fundamental
#                 frequency, and thus also be a function.
#     """
#     # fundamental frequencies
#     fundamentals = np.asarray(fundamentals)
#     # compute the frequencies of the harmonics, which equal the filter centers;
#     # h represents the factors for each harmonic, which are then multiplied
#     # with the fundamental
#     h = np.arange(num_harmonics + 1) + 1
#     h_inh = h * np.sqrt(1 + h * h * inharmonicity_coeff)
#     filter_centers = fundamentals * h_inh[:, np.newaxis]
#     # compute filter start and end frequencies, based on the harmonic_width
#     # function. Also the weights for each harmonic filter are computed.
#     # TODO: allow using a list of weights/widths instead of a function
#     filter_widths = harmonic_width(h) / 2
#     filter_weights = harmonic_envelope(h)
#     filter_starts = filter_centers - filter_widths[:, np.newaxis]
#     filter_ends = filter_centers + filter_widths[:, np.newaxis]
#     # map the filter start, center and end frequencies to frequency bins
#     # of the spectrogram
#     factor = (sample_rate / 2.0) / num_fft_bins
#     filter_centers = np.round(filter_centers / factor).astype(int)
#     filter_starts = np.round(filter_starts / factor).astype(int)
#     filter_starts = np.minimum(filter_starts, filter_centers - 1)
#     filter_ends = np.round(filter_ends / factor).astype(int)
#     filter_ends = np.maximum(filter_ends, filter_centers + 1)
#     # create a list of filters per band
#     filters = [[] for _ in range(len(fundamentals))]
#     # iterate over filters for each harmonic in each filter band
#     for harm_id, band_id in np.ndindex(filter_starts.shape):
#         # determine the filter positions
#         start = filter_starts[harm_id, band_id]
#         center = filter_centers[harm_id, band_id]
#         end = filter_ends[harm_id, band_id]
#         # skip if the complete filter would be outside the allowed range
#         if start > num_fft_bins or end < 0:
#             continue
#         # set filter arguments
#         params = {'width': end - start,
#                   'center': center - start,
#                   'norm': False}
#         # create a filter of filter_type with the given arguments and
#         # weight it accordingly
#         filt = filter_type(**params) * filter_weights[harm_id]
#         # add this filter to the list of filters for multi_filterbank
#         filters[band_id].append(FilterType(filt, start))
#     # create and return the filterbank
#     return assemble_filterbank(filters, num_fft_bins, norm=True)


class Filterbank(np.ndarray):
    """
    Generic Filterbank class.

    """

    def __new__(cls, data, sample_rate):
        """
        Creates a new Filterbank array.

        :param data:        2-d numpy array
        :param sample_rate: sample rate of the audio signal [Hz]

        """
        # input is an numpy ndarray instance
        if isinstance(data, np.ndarray):
            # cast as Filterbank
            obj = np.asarray(data).view(cls)
        else:
            raise TypeError('wrong input data for Filterbank, must be '
                            'np.ndarray')
        # set attributes
        obj._sample_rate = sample_rate
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self._norm = getattr(obj, '_norm', None)

    @classmethod
    def _put_filter(cls, filt, band):
        """
        Puts a filter in the band, internal helper function.

        :param filt: numpy array
        :param band: band in which the filter should be put (numpy array)

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
            filt = filt[:stop - len(band)]
            stop = len(band)
        # put the filter in place
        filter_position = band[start:stop]
        # TODO: if needed, allow other handling (like summing values)
        np.maximum(filt, filter_position, out=filter_position)

    @classmethod
    def from_filters(cls, filters, num_fft_bins, sample_rate):
        """
        Creates a filterbank with possibly multiple filters per band.

        :param filters:      list containing the filters per band; if multiple
                             filters per band are desired, they should be also
                             contained in a list, resulting in a list of lists
                             of filters.
        :param num_fft_bins: number of FFT bins (= half the FFT size)
        :param sample_rate:  sample rate of the audio signal
        :return:             filterbank with respective filter elements

        """
        # create filterbank
        fb = np.zeros((num_fft_bins, len(filters)))
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
        return Filterbank.__new__(cls, fb, sample_rate)

    @property
    def num_fft_bins(self):
        """Number of FFT bins."""
        return self.shape[0]

    @property
    def num_bands(self):
        """Number of bands."""
        return self.shape[1]

    @property
    def sample_rate(self):
        """Sample rate of the signal."""
        return self._sample_rate

    @property
    def bin_frequencies(self):
        """Frequencies of FFT bins."""
        return fft_freqs(self.num_fft_bins, self.sample_rate)

    @property
    def filter_corner_frequencies(self):
        """Corner frequencies of the filters."""
        freqs = []
        for band in range(self.num_bands):
            bins = np.nonzero(self[:, band])[0]
            fmin, fmax = bins2frequencies((np.min(bins) - 1, np.max(bins)),
                                          self.num_fft_bins, self.sample_rate)
            freqs.append([fmin, fmax])
        return np.asarray(freqs)

    @property
    def filter_centre_frequencies(self):
        """Centre frequencies of the filters."""
        freqs = []
        for band in range(self.num_bands):
            freqs.append(bins2frequencies(np.argmax(self[:, band]),
                                          self.num_fft_bins, self.sample_rate))
        return np.asarray(freqs)

    @property
    def fmin(self):
        """Minimum frequency of the filterbank."""
        return self.bin_frequencies[np.nonzero(self)[0][0]]

    @property
    def fmax(self):
        """Maximum frequency of the filterbank."""
        return self.bin_frequencies[np.nonzero(self)[0][-1]]

    @property
    def norm(self):
        """Filters are normalised."""
        return self._norm

    def __str__(self):
        return "Filterbank: %d FFT bins; %d bands; fmin: %.1f; fmax: %.1f" % \
               (self.num_fft_bins, self.num_bands, self.fmin, self.fmax)

    @staticmethod
    def add_arguments(parser, default=None, fmin=FMIN, fmax=FMAX,
                      bands=BANDS_PER_OCTAVE, norm_filters=NORM_FILTERS):
        """
        Add filter related arguments to an existing parser object.

        :param parser:       existing argparse parser object
        :param default:      set the default (adds a switch to negate)
        :param fmin:         the minimum frequency
        :param fmax:         the maximum frequency
        :param bands:        number of filter bands per octave
        :param norm_filters: normalise the area of the filter
        :return:             filtering argument parser group object

        """
        # TODO: split this among the individual classes
        # add filter related options to the existing parser
        g = parser.add_argument_group('filterbank related arguments')
        if default is False:
            g.add_argument('--filter', action='store_true', default=default,
                           help='filter the magnitude spectrogram with a '
                                'filterbank (apply values below)')
        elif default is True:
            g.add_argument('--no_filter', dest='filter', action='store_false',
                           default=default,
                           help='do not filter the magnitude spectrogram with '
                                'a filterbank (ignore values below)')
        if bands is not None:
            g.add_argument('--bands', action='store', type=int, default=bands,
                           help='filter bands per octave '
                                '[default=%(default)i]')
        if fmin is not None:
            g.add_argument('--fmin', action='store', type=float, default=fmin,
                           help='minimum frequency of filter in Hz [default='
                                '%(default)i]')
        if fmax is not None:
            g.add_argument('--fmax', action='store', type=float, default=fmax,
                           help='maximum frequency of filter in Hz [default='
                                '%(default)i]')
        if norm_filters is False:
            # switch to turn it on
            g.add_argument('--norm_filters', action='store_true',
                           default=norm_filters,
                           help='normalise filters to have equal area')
        if norm_filters is True:
            g.add_argument('--no_norm_filters', dest='norm_filters',
                           action='store_false', default=norm_filters,
                           help='do not equalize filters to have equal area')
        # return the argument group so it can be modified if needed
        return g


class MelFilterbank(Filterbank):
    """
    Mel filterbank class.

    """
    MEL_BANDS = 40

    def __new__(cls, num_fft_bins, sample_rate, fmin=FMIN, fmax=FMAX,
                bands=MEL_BANDS, norm=NORM_FILTERS,
                duplicates=DUPLICATE_FILTERS):
        """
        Creates a new MelFilterbank instance.

        :param num_fft_bins: number of FFT bins (= half the FFT size)
        :param sample_rate:  sample rate of the audio file [Hz]
        :param fmin:         the minimum frequency [Hz]
        :param fmax:         the maximum frequency [Hz]
        :param bands:        number of filter bands
        :param norm:         normalise the filters to area 1
        :param duplicates:   keep duplicate filters resulting from insufficient
                             resolution of low frequencies

        """
        # get a list of frequencies
        # request 2 more bands, because these are the edge frequencies
        frequencies = mel_frequencies(bands + 2, fmin, fmax)
        # convert to bins
        bins = frequencies2bins(frequencies, num_fft_bins, sample_rate)
        # get overlapping triangular filters
        filters = TriangularFilter.filters(bins, norm=norm,
                                           duplicates=duplicates, overlap=True)
        # create a MelFilterbank from the filters
        obj = cls.from_filters(filters, num_fft_bins, sample_rate)
        # set additional attributes
        obj._norm = norm
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self._norm = getattr(obj, '_norm', NORM_FILTERS)


class BarkFilterbank(Filterbank):
    """
    Bark filterbank Class.

    """
    BARK_FMIN = 20
    BARK_FMAX = 15500
    BARK_DOUBLE = False

    def __new__(cls, num_fft_bins, sample_rate, fmin=BARK_FMIN, fmax=BARK_FMAX,
                double=BARK_DOUBLE, norm=NORM_FILTERS,
                duplicates=DUPLICATE_FILTERS):
        """
        Creates a new BarkFilterbank instance.

        :param num_fft_bins: number of FFT bins (= half the FFT size)
        :param sample_rate:  sample rate of the audio file [Hz]
        :param fmin:         the minimum frequency [Hz]
        :param fmax:         the maximum frequency [Hz]
        :param double:       double the number of frequency bands
        :param norm:         normalise the area of the filter to 1
        :param duplicates:   keep duplicate filters resulting from insufficient
                             resolution of low frequencies

        """
        # get a list of frequencies
        if double:
            frequencies = bark_double_frequencies(fmin, fmax)
        else:
            frequencies = bark_frequencies(fmin, fmax)
        # convert to bins
        bins = frequencies2bins(frequencies, num_fft_bins, sample_rate)
        # get non-overlapping rectangular filters
        filters = RectangularFilter.filters(bins, norm=norm,
                                            duplicates=duplicates,
                                            overlap=False)
        # create a BarkFilterbank from the filters
        obj = cls.from_filters(filters, num_fft_bins, sample_rate)
        # set additional attributes
        obj._norm = norm
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self._norm = getattr(obj, '_norm', NORM_FILTERS)


class LogarithmicFilterbank(Filterbank):
    """
    Logarithmic filterbank class.

    """
    def __new__(cls, num_fft_bins, sample_rate,
                bands_per_octave=BANDS_PER_OCTAVE, fmin=FMIN, fmax=FMAX,
                norm=NORM_FILTERS, duplicates=DUPLICATE_FILTERS, a4=A4):
        """
        Creates a new LogarithmicFilterbank instance.

        :param num_fft_bins:     number of FFT bins (=half the FFT size)
        :param sample_rate:      sample rate of the audio file [Hz]
        :param bands_per_octave: number of filter bands per octave
        :param fmin:             the minimum frequency [Hz]
        :param fmax:             the maximum frequency [Hz]
        :param norm:             normalise the area of the filter to 1
        :param duplicates:       keep duplicate filters resulting from
                                 insufficient resolution of low frequencies
        :param a4:               tuning frequency of A4 [Hz]

        """
        # get a list of frequencies
        frequencies = log_frequencies(bands_per_octave, fmin, fmax, a4)
        # convert to bins
        bins = frequencies2bins(frequencies, num_fft_bins, sample_rate)
        # get overlapping triangular filters
        filters = TriangularFilter.filters(bins, norm=norm,
                                           duplicates=duplicates, overlap=True)
        # create a LogarithmicFilterbank from the filters
        obj = cls.from_filters(filters, num_fft_bins, sample_rate)
        # set additional attributes
        obj._bands_per_octave = bands_per_octave
        obj._norm = norm
        obj._a4 = a4
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self._bands_per_octave = getattr(obj, '_bands_per_octave',
                                         BANDS_PER_OCTAVE)
        self._norm = getattr(obj, '_norm', NORM_FILTERS)
        self._a4 = getattr(obj, '_a4', A4)

    @property
    def bands_per_octave(self):
        """Number of bands per octave."""
        return self._bands_per_octave

    @property
    def a4(self):
        """Tuning frequency of A4."""
        return self._a4

# alias
LogFilterbank = LogarithmicFilterbank


class SemitoneFilterbank(LogarithmicFilterbank):
    """
    Semitone filterbank class.

    """
    def __new__(cls, num_fft_bins, sample_rate, fmin=FMIN, fmax=FMAX,
                norm=NORM_FILTERS, duplicates=DUPLICATE_FILTERS, a4=A4):
        """
        Creates a new SemitoneFilterbank instance.

        :param num_fft_bins: number of FFT bins (= half the FFT size)
        :param sample_rate:  sample rate of the audio file [Hz]
        :param fmin:         the minimum frequency [Hz]
        :param fmax:         the maximum frequency [Hz]
        :param norm:         normalise the area of the filter to 1
        :param duplicates:   keep duplicate filters resulting from insufficient
                             resolution of low frequencies
        :param a4:           tuning frequency of A4 [Hz]

        """
        # return a LogarithmicFilterbank with 12 bands per octave
        return LogarithmicFilterbank.__new__(cls, num_fft_bins, sample_rate,
                                             12, fmin, fmax, norm, duplicates,
                                             a4)


class SimpleChromaFilterbank(Filterbank):
    """
    A simple chroma filterbank based on the semitone filter.

    """
    def __new__(cls, num_fft_bins, sample_rate, fmin=FMIN, fmax=FMAX,
                norm=NORM_FILTERS, duplicates=DUPLICATE_FILTERS, a4=A4):
        """
        Creates a new SimpleChromaFilterbank instance.

        :param num_fft_bins: number of FFT bins (= half the FFT size)
        :param sample_rate:  sample rate of the audio file [Hz]
        :param fmin:         the minimum frequency [Hz]
        :param fmax:         the maximum frequency [Hz]
        :param norm:         normalise the area of the filter to 1
        :param duplicates:   omit duplicate filters resulting from insufficient
                             resolution of low frequencies
        :param a4:           tuning frequency of A4 [Hz]

        """

        stf = SemitoneFilterbank(num_fft_bins, sample_rate, fmin, fmax, norm,
                                 duplicates, a4)

        fb = np.empty((stf.shape[0], 12))
        spacing = np.arange(8) * 12

        for i in range(12):
            cur_spacing = spacing + i
            cur_spacing = cur_spacing[cur_spacing < stf.shape[1]]
            fb[:, i] = stf[:, cur_spacing].sum(1)

        # TODO: check if this should depend on the norm parameter
        fb /= fb.sum(0)

        # cast to Filterbank
        obj = Filterbank.__new__(cls, fb, sample_rate)
        # set additional attributes
        obj._norm = norm
        obj._a4 = a4
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self._norm = getattr(obj, '_norm', NORM_FILTERS)
        self._a4 = getattr(obj, '_a4', A4)

    @property
    def norm(self):
        """Filters are normalised."""
        return self._norm

    @property
    def a4(self):
        """Tuning frequency of A4."""
        return self._a4


class PitchClassProfileFilterbank(Filterbank):
    """
    Filterbank for extracting pitch class profiles (PCP).

    "Realtime chord recognition of musical sound: a system using Common Lisp
     Music"
    T. Fujishima
    Proceedings of the International Computer Music Conference (ICMC 1999),
    Beijing, China

    """
    # default values
    PCP_CLASSES = 12

    def __new__(cls, num_fft_bins, sample_rate, num_classes=PCP_CLASSES,
                fmin=FMIN, fmax=FMAX, fref=A4):
        """
        Creates a new PitchClassProfile (PCP) filterbank instance.

        :param num_fft_bins: number of FFT bins (= half the FFT size)
        :param sample_rate:  sample rate of the audio file [Hz]
        :param num_classes:  number of pitch classes
        :param fmin:         the minimum frequency [Hz]
        :param fmax:         the maximum frequency [Hz]
        :param fref:         reference frequency for the first PCP bin [Hz]

        """
        # init a filterbank
        fb = np.zeros((num_fft_bins, num_classes))
        # frequencies of the bins
        bin_freqs = fft_freqs(num_fft_bins, sample_rate)
        # log deviation from the reference frequency
        log_dev = np.log2(bin_freqs / fref)
        # map the log deviation to the closets pitch class profiles
        num_class = np.round(num_classes * log_dev) % num_classes
        # define the pitch class profile filterbank
        # skip log_dev[0], since it is NaN
        fb[np.arange(1, num_fft_bins), num_class.astype(int)[1:]] = 1
        # set all bins outside the allowed frequency range to 0
        fb[np.searchsorted(bin_freqs, fmax, 'right'):] = 0
        fb[:np.searchsorted(bin_freqs, fmin)] = 0
        # cast to Filterbank
        obj = Filterbank.__new__(cls, fb, sample_rate)
        # set additional attributes
        obj._fref = fref
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self._fref = getattr(obj, '_fref', A4)

    @property
    def fref(self):
        """Reference frequency of the first PCP bin."""
        return self._fref


class HarmonicPitchClassProfileFilterbank(Filterbank):
    """
    Filterbank for extracting harmonic pitch class profiles (HPCP).

    "Tonal Description of Music Audio Signals"
    E. Gómez
    PhD thesis, Universitat Pompeu Fabra, Barcelona, Spain

    """
    # default values
    HPCP_FMIN = 100
    HPCP_FMAX = 5000
    HPCP_CLASSES = 36
    HPCP_WINDOW = 4

    def __new__(cls, num_fft_bins, sample_rate, num_classes=HPCP_CLASSES,
                fmin=HPCP_FMIN, fmax=HPCP_FMAX, fref=A4, window=HPCP_WINDOW):
        """
        Creates a new HarmonicPitchClassProfile (HPCP) filterbank instance.

        :param num_fft_bins:  number of FFT bins (= half the FFT size)
        :param sample_rate:   sample rate of the audio file [Hz]
        :param num_classes:   number of harmonic pitch classes
        :param fmin:          the minimum frequency [Hz]
        :param fmax:          the maximum frequency [Hz]
        :param fref:          reference frequency for the first HPCP bin [Hz]
        :param window:        length of the weighting window [bins]

        """
        # init a filterbank
        fb = np.zeros((num_fft_bins, num_classes))
        # frequencies of the FFT bins
        bin_freqs = fft_freqs(num_fft_bins, sample_rate)
        # log deviation from the reference frequency
        log_dev = np.log2(bin_freqs / fref)
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
        fb[np.searchsorted(bin_freqs, fmax, 'right'):] = 0
        fb[:np.searchsorted(bin_freqs, fmin)] = 0
        # cast to Filterbank
        obj = Filterbank.__new__(cls, fb, sample_rate)
        # set additional attributes
        obj._fref = fref
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self._fref = getattr(obj, '_fref', A4)

    @property
    def fref(self):
        """Reference frequency of the first HPCP bin."""
        return self._fref


# time filters
def feed_forward_comb_filter(x, tau, alpha):
    """
    Filter the signal with a feed forward comb filter.

    :param x:     signal
    :param tau:   delay length
    :param alpha: scaling factor
    :return:      comb filtered signal

    """
    # y[n] = x[n] + α * x[n - τ]
    if tau <= 0:
        raise ValueError('tau must be greater than 0')
    y = np.copy(x)
    # add the delayed signal
    y[tau:] += alpha * x[:-tau]
    # return
    return y


def _feed_backward_comb_filter(x, tau, alpha):
    """
    Filter the signal with a feed backward comb filter.

    :param x:     signal
    :param tau:   delay length
    :param alpha: scaling factor
    :return:      comb filtered signal

    """
    # y[n] = x[n] + α * y[n - τ]
    if tau <= 0:
        raise ValueError('tau must be greater than 0')
    y = np.copy(x)
    # loop over the complete signal
    for n in range(tau, len(x)):
        # add a delayed version of the output signal
        y[n] = x[n] + alpha * y[n - tau]
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


def comb_filterbank(x, comb_filter, tau, alpha):
    """
    Filter the signal with a bank of either feed forward or backward comb
    filters.

    :param x:           signal [numpy array]
    :param comb_filter: comb filter to use (feed forward or backward)
    :param tau:         delay length(s) [list/array of samples]
    :param alpha:       corresponding scaling factor(s) [list/array of floats]
    :return:            comb filtered signal with the different taus aligned
                        along the (new) 1st dimension

    """
    # convert tau to a integer numpy array
    tau = np.asarray(tau, dtype=int)
    if tau.ndim != 1:
        raise ValueError('tau must be 1D numpy array')
    # convert alpha to a numpy array
    alpha = np.asarray(alpha, dtype=float)
    if alpha.ndim != 1:
        raise ValueError('alpha must be 1D numpy array')
    # alpha & tau must have the same size
    if tau.size != alpha.size:
        raise AssertionError('alpha & tau must have the same size')
    # determine output array size
    size = list(x.shape)
    # add dimension of tau range size (new 1st dim)
    size.insert(0, len(tau))
    # init output array
    y = np.zeros(tuple(size))
    for i, t in np.ndenumerate(tau):
        y[i] = comb_filter(x, t, alpha[i])
    return y


class CombFilterbank(np.ndarray):
    """
    Comb Filterbank class.

    """

    def __new__(cls, data, comb_filter, tau, alpha):
        """
        Creates a new CombFilterbank array, i.e. a comb filtered version of
        the input data with the different tau values aligned along the (new)
        1st dimension.

        :param data:             numpy array
        :param comb_filter:      comb filter to use (feed forward or backward)
        :param tau:              delay length(s)
                                 [samples, list/array of samples]
        :param alpha:            scaling factor(s)
                                 [float, list/array of floats]

        """
        # convert tau to a numpy array
        if isinstance(tau, int):
            tau = np.asarray([tau], dtype=int)
        elif isinstance(tau, list):
            tau = np.asarray(tau, dtype=int)
        elif isinstance(tau, np.ndarray):
            tau = np.asarray(tau, dtype=int)
        else:
            raise ValueError('tau must be convertible to an int numpy array')

        # set the filter function
        if comb_filter in ['forward', feed_backward_comb_filter]:
            comb_filter = feed_backward_comb_filter
        elif comb_filter in ['backward', feed_backward_comb_filter]:
            comb_filter = feed_backward_comb_filter
        else:
            raise ValueError('wrong comb_filter type')

        # convert alpha to a numpy array
        if isinstance(alpha, (float, int)):
            alpha = np.asarray([alpha] * len(tau), dtype=float)
        elif isinstance(alpha, list):
            alpha = np.asarray(alpha, dtype=float)
        elif isinstance(alpha, np.ndarray):
            alpha = np.asarray(alpha, dtype=float)
        else:
            raise ValueError('alpha must be cast-able as float numpy array')

        # comb filter the signal
        cfb = comb_filterbank(data, comb_filter, tau, alpha)
        # cast to CombFilterbank
        obj = np.asarray(cfb).view(cls)
        # set attributes
        obj._data = data
        obj._comb_filter = comb_filter
        obj._tau = tau
        obj._alpha = alpha
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def data(self):
        """Original non-comb filtered data."""
        return self._data

    @property
    def comb_filter(self):
        """Comb filter function."""
        return self._comb_filter

    @property
    def tau(self):
        """Tau value(s), i.e. the delay length(s) of the filterbank."""
        return self._tau

    @property
    def alpha(self):
        """Scaling factor(s)."""
        return self._alpha
