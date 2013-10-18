#!/usr/bin/env python
# encoding: utf-8
"""
This file contains filter and filterbank related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np
from collections import namedtuple
from functools import partial


# default values for filters
FMIN = 30
FMAX = 17000
MEL_BANDS = 40
BARK_DOUBLE = False
BANDS_PER_OCTAVE = 12
NORM_FILTER = True
OMIT_DUPLICATES = True
OVERLAP_FILTERS = True
A4 = 440

HARMONIC_ENVELOPE = lambda x: np.sqrt(1. / x)
HARMONIC_WIDTH = lambda x: 50 * 1.1 ** x
INHARMONICITY_COEFF = 0.0

# Mel frequency scale
def hz2mel(f):
    """
    Convert Hz frequencies to Mel.

    :param f: input frequencies [Hz]
    :returns: frequencies in Mel

    """
    return 1127.01048 * np.log(f / 700. + 1.)


def mel2hz(m):
    """
    Convert Mel frequencies to Hz.

    :param m: input frequencies [Mel]
    :returns: frequencies in Hz

    """
    return 700. * (np.exp(m / 1127.01048) - 1.)


def mel_frequencies(bands, fmin, fmax):
    """
    Generates a list of frequencies aligned on the Mel scale.

    :param bands: number of bands
    :param fmin:  the minimum frequency [Hz]
    :param fmax:  the maximum frequency [Hz]
    :returns:     a list of frequencies

    """
    # convert fmin and fmax to the Mel scale
    mmin = hz2mel(fmin)
    mmax = hz2mel(fmax)
    # return a list of frequencies
    return mel2hz(np.linspace(mmin, mmax, bands))


# Bark frequency scale
def hz2bark(f):
    """
    Convert Hz frequencies to Bark.

    :param f: input frequencies [Hz]
    :returns: frequencies in Bark.

    """
    # TODO: use Zwicker's formula?
    # return 13. * arctan(0.00076*f) + 3.5 * arctan((f/7500.)**2)
    return (26.81 / (1. + 1960. / f)) - 0.53


def bark2hz(z):
    """
    Convert Bark frequencies to Hz.

    :param z: input frequencies [Bark]
    :returns: frequencies in Hz.

    """
    # TODO: use Zwicker's formula?
    # return 13. * arctan(0.00076*f) + 3.5 * arctan((f/7500.)**2)
    return 1960. / (26.81 / (z + 0.53) - 1.)


def bark_frequencies(fmin=20, fmax=15500):
    """
    Generates a list of corner frequencies aligned on the Bark-scale.

    :param fmin: the minimum frequency [Hz, default=20]
    :param fmax: the maximum frequency [Hz, default=15550]
    :returns:    a list of frequencies

    """
    # frequencies aligned to the Bark-scale
    freqs = np.array([20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270,
                      1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300,
                      6400, 7700, 9500, 12000, 15500])
    # filter frequencies
    freqs = freqs[freqs >= fmin]
    freqs = freqs[freqs <= fmax]
    # return
    return freqs


def bark_double_frequencies(fmin=20, fmax=15500):
    """
    Generates a list of corner frequencies aligned on the Bark-scale.
    The list includes also center frequencies between the corner frequencies.

    :param fmin: the minimum frequency [Hz, default=20]
    :param fmax: the maximum frequency [Hz, default=15550]
    :returns:    a list of frequencies

    """
    # frequencies aligned to the Bark-scale, does also including center frequencies
    freqs = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 510, 570,
                      630, 700, 770, 840, 920, 1000, 1080, 1170, 1270, 1370,
                      1480, 1600, 1720, 1850, 2000, 2150, 2320, 2500, 2700,
                      2900, 3150, 3400, 3700, 4000, 4400, 4800, 5300, 5800,
                      6400, 7000, 7700, 8500, 9500, 10500, 12000, 13500, 15500])
    # filter frequencies
    freqs = freqs[freqs >= fmin]
    freqs = freqs[freqs <= fmax]
    # return
    return freqs


## (pseudo) Constant-Q frequency scale
#
### FIXME: this is NOT faster!
##def cq_frequencies(bands, fmin, fmax, a=440):
##    emin = np.floor(np.log2(float(fmin) / a))
##    emax = np.ceil(np.log2(float(fmax) / a))
##    freqs = np.logspace(emin, emax, (emax - emin) * bands, base=2, endpoint=False) * a
##    return [x for x in freqs if (x >= fmin and x <= fmax)]
#
#
#def cq_frequencies(bands_per_octave, fmin, fmax, a4=440):
#    """
#    Generates a list of frequencies aligned on a logarithmic frequency scale.
#
#    :param bands_per_octave: number of filter bands per octave
#    :param fmin: the minimum frequency [Hz]
#    :param fmax: the maximum frequency [Hz]
#    :param a4: tuning frequency of A4 [Hz, default=440]
#    :returns: a list of frequencies
#
#    Note: frequencies are aligned to MIDI notes with the default a4=440 and
#    12 bands_per_octave.
#
#    """
#    # factor by which 2 frequencies are located apart from each other
#    factor = 2.0 ** (1.0 / bands_per_octave)
#    # start with A4
#    freq = a4
#    frequencies = []
#    # go upwards till fmax
#    while freq <= fmax:
#        frequencies.append(freq)
#        freq *= factor
#    # restart with a and go downwards till fmin
#    freq = a4 / factor
#    while freq >= fmin:
#        frequencies.append(freq)
#        freq /= factor
#    # sort frequencies
#    frequencies.sort()
#    # return the list
#    return frequencies


def log_frequencies(bands_per_octave, fmin, fmax, a4=A4):
    """
    Generates a list of frequencies aligned on a logarithmic frequency scale.

    :param bands_per_octave: number of filter bands per octave
    :param fmin:             the minimum frequency [Hz]
    :param fmax:             the maximum frequency [Hz]
    :param a4:               tuning frequency of A4 [Hz, default=440]
    :returns:                a list of frequencies

    Note: if 12 bands per octave and a4=440 are used, the frequencies are
          equivalent to MIDI notes.

    """
    # get the range
    left = np.floor(np.log2(float(fmin) / a4) * bands_per_octave)
    right = np.ceil(np.log2(float(fmax) / a4) * bands_per_octave)
    # generate frequencies
    freqs = a4 * 2 ** (np.arange(left, right) / float(bands_per_octave))
    # filter frequencies
    # needed, because range might be bigger because of the use of floor/ceil
    freqs = freqs[freqs >= fmin]
    freqs = freqs[freqs <= fmax]
    # return
    return freqs


def semitone_frequencies(fmin, fmax, a4=A4):
    """
    Generates a list of frequencies separated by semitones.

    :param fmin: the minimum frequency [Hz]
    :param fmax: the maximum frequency [Hz]
    :param a4:   tuning frequency of A4 [Hz, default=440]
    :returns:    a list of frequencies of semitones

    Note: frequencies are aligned to MIDI notes with the default a4=440.

    """
    # return MIDI frequencies
    return log_frequencies(12, fmin, fmax, a4)


# MIDI
def midi2hz(m, a4=A4):
    """
    Convert frequencies to the corresponding MIDI notes.

    :param m:  input MIDI notes
    :param a4: tuning frequency of A4 [Hz, default=440]
    :returns:  frequencies in Hz

    For details see: http://www.phys.unsw.edu.au/jw/notes.html

    """
    return 2. ** ((m - 69.) / 12.) * a4


def hz2midi(f, a4=A4):
    """
    Convert MIDI notes to corresponding frequencies.

    :param f:  input frequencies [Hz]
    :param a4: tuning frequency of A4 [Hz, default=440]
    :returns:  MIDI notes

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
    :returns: frequencies in ERB

    Information about the ERB scale can be found at:
    https://ccrma.stanford.edu/~jos/bbt/Equivalent_Rectangular_Bandwidth.html

    """
    return 21.4 * np.log10(1 + 4.37 * f / 1000.)


def erb2hz(e):
    """
    Convert ERB scaled frequencies to Hz.

    :param e: input frequencies [ERB]
    :returns: frequencies in Hz

    Information about the ERB scale can be found at:
    https://ccrma.stanford.edu/~jos/bbt/Equivalent_Rectangular_Bandwidth.html

    """
    return (10. ** (e / 21.4) - 1.) * 1000. / 4.37


# Cent Scale
# FIXME: check the formulas, they seem to generate weird results
#        additionally, cents are intervals and not absolute values...
#def hz2cent(f, a4=440):
#    """
#    Convert Hz to Cent.
#
#    :param f: input frequencies [Hz]
#    :param a4: tuning frequency of A4 [Hz, default=440]
#    :returns: frequencies in Cent
#
#    """
#    # FIXME: why this == 0 check?
#    if f == 0:
#        return 0
#    return 1200.0 * np.log2(f / (a4 * 2.0 ** ((3 / 12.0) - 5.0)))
#
#
#def cent2hz(c, a4=440):
#    """
#    Convert Cent to Hz.
#
#    :param c: input frequencies [Cent]
#    :param a4: tuning frequency of A4 [Hz, default=440]
#    :returns: frequencies in Hz
#
#    """
#    return (a4 * 2.0 ** ((3 / 12.0) - 5.0)) * np.exp(c / 1200.)


# helper functions for filter creation
def fft_freqs(fft_bins, sample_rate):
    # faster than: np.fft.fftfreq(fft_bins * 2)[:fft_bins] * sample_rate
    return np.linspace(0, sample_rate / 2., fft_bins + 1)


class FilterElement(np.ndarray):

    def __new__(cls, input_array, start, stop):
        obj = np.asarray(input_array).view(cls)
        obj.__start = start
        obj.__stop = stop
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.__fft_bins = getattr(obj, '__start', None)
        self.__bands = getattr(obj, '__stop', None)

    @property
    def start(self):
        return self.__start

    @property
    def stop(self):
        return self.__stop


def triang_filter(start, center, stop, norm):
    """
    Calculate a triangular window of the given size.

    :param start:  starting bin (with value 0, included in the returned filter)
    :param center: center bin (of height 1, unless norm is True)
    :param stop:   end bin (with value 0, not included in the returned filter)
    :param norm:   normalize the area of the filter to 1
    :returns:      a triangular shaped filter with height 1 (unless normalized)

    """
    # Set the height of the filter, normalised if necessary.
    # A standard filter is at least 3 bins wide, and stop - start = 2
    # thus the filter has an area of 1 if normalised this way
    height = 2. / (stop - start) if norm else 1.

    triang_filter = np.empty(stop - start)

    # rising edge (without the center)
    triang_filter[:center - start] = np.linspace(0, height, (center - start), endpoint=False)
    # falling edge (including the center, but without the last bin since it's 0)
    triang_filter[center - start:] = np.linspace(height, 0, (stop - center), endpoint=False)

    return FilterElement(triang_filter, start=start, stop=stop)


def rectang_filter(start, stop, norm, **kwargs):
    """
    Calculate a rectangular window of the given size.

    :param start:  starting bin (with value 0, included in the returned filter)
    :param stop:   end bin (with value 0, not included in the returned filter)
    :param norm:   normalize the area of the filter to 1
    :returns:      a rectangular shaped filter with height 1 (unless normalized)

    """
    # Set the height of the filter, normalised if necessary
    height = 1. / (stop - start) if norm else 1.
    rectang_filter = np.ones(stop - start) * height

    return FilterElement(rectang_filter, start=start, stop=stop)


def multi_filterbank(filters, fft_bins, bands, norm):
    """
    Creates a filterbank with multiple filter elements per band.

    :param filters:  Dictionary containing lists of filters per band. Keys are
                     band ids.
    :param fft_bins: Number of FFT bin
    :param bands:    Number of bands
    :param norm:     Normalise the area of each filterband to 1 if True
    :returns:        Filterbank with respective filter elements

    """
    bank = np.zeros((fft_bins, bands))

    for band_id, band_filts in filters.iteritems():
        for filt in band_filts:
            filt_pos = bank[filt.start:filt.stop, band_id]
            np.maximum(filt, filt_pos, out=filt_pos)

    if norm:
        bank /= bank.sum(0)

    return bank


def filterbank(filter_type, frequencies, fft_bins, sample_rate, norm=NORM_FILTER,
               omit_duplicates=OMIT_DUPLICATES, overlap=OVERLAP_FILTERS):
    """
    Creates a filterbank with one filter per band.

    :param filter_type: method that creates a filter element
    :param frequencies: a list of frequencies used for filter creation [Hz]
    :param fft_bins:    number of fft bins
    :param sample_rate: sample rate of the audio signal [Hz]
    :param norm:        normalise the area of the filters to 1 [default=True]
    :param omit_duplicates: omit duplicate filters resulting from insufficient
        resolution of low frequencies [default=True]
    :param overlap:     overlapping filters or not
    :returns:           filterbank

    Note: Depending on whether filters are overlapping or not, each filter is 
          characterized by 2 (no overlap) or 3 (overlap) frequencies. 

          In case of no overlap, start and stop frequencies are used, the
          center frequency of each band is exactly in between these
          frequencies.

          In case of overlap, the start, center and stop frequency are given
          as parameters. Thus the frequencies array must contain the first
          starting frequency, all center frequencies and the last stopping
          frequency.

    """
    factor = (sample_rate / 2.0) / fft_bins
    # map the frequencies to the spectrogram bins
    frequencies = np.round(np.asarray(frequencies) / factor).astype(int)
    # filter out all frequencies outside the valid range
    frequencies = frequencies[frequencies < fft_bins]
    # FIXME: skip the DC bin 0?
    # only keep unique bins if requested
    # Note: this is important to do so, otherwise the lower frequency bins are
    # given too much weight if simply summed up (as in the spectral flux)
    if omit_duplicates:
        frequencies = np.unique(frequencies)
    # number of bands

    bands = len(frequencies) - 1
    if overlap:
        bands -= 1

    if bands < 1:
        raise ValueError("Cannot create filterbank with less than 1 band")

    filters = {}
    for band in range(bands):
        # edge & center frequencies
        if overlap:
            start, mid, stop = frequencies[band:band + 3]
        else:
            start, stop = frequencies[band:band + 2]
            mid = int((start + stop) / 2)

        # consistently handle too-small filters
        # FIXME: Does this have any meaningful effect except when
        #        start == mid == stop
        if not omit_duplicates and (stop - start < 2):
            mid = start
            stop = start + 1

        # create the filter
        kwargs = {'start': start, 'center': mid, 'stop': stop, 'norm': norm}
        filters[band] = [filter_type(**kwargs)]

    # no normalisation here, since each filter is already normalised
    return multi_filterbank(filters, fft_bins, len(filters), norm=False)


def harmonic_filterbank(filter_type, fundamentals, num_harmonics, fft_bins, sample_rate,
                        harmonic_envelope=HARMONIC_ENVELOPE, harmonic_width=HARMONIC_WIDTH,
                        inharmonicity_coeff=INHARMONICITY_COEFF):
    """
    Creates a filterbank in which each band represents a fundamental frequency
    and its harmonics.

    :param filter_type:   function that creates a filter
    :param fundamentals:  list of fundamental frequencies
    :param num_harmonics: number of harmonics for each fundamental frequency
    :param fft_bins:      number of fft bins
    :param sample_rate:   sample rate of the audio signal [Hz]
    :param harmonic_envelope: function returning a weight for each harmonic
                          and the f0. [default=lambda x: np.sqrt(1. / x)]
    :param harmonic_width: function returning the width for each harmonic and
                          the f0. [default=50 * 1/1 ** x]
    :param inharmonicity_coeff: coefficient for calculating the drift of
                          harmonics for not perfectly harmonic instruments.

    :returns: harmonic filterbank

    Notes: harmonic_envelope and harmonic_width must accept a numpy array of
           the harmonic ids, where the fundamental's id is 1, the second
           harmonic is 2, etc...

           TODO: inharmonicity_coeff should depend on the fundamental
                 frequency, and thus also be a function.
    """

    fundamentals = np.asarray(fundamentals)
    h = np.arange(num_harmonics + 1) + 1
    h_inh = h * np.sqrt(1 + h * h * inharmonicity_coeff)
    filter_centers = fundamentals * h_inh[:, np.newaxis]

    filter_widths = harmonic_width(h) / 2
    filter_weights = harmonic_envelope(h)
    filter_starts = filter_centers - filter_widths[:, np.newaxis]
    filter_ends = filter_centers + filter_widths[:, np.newaxis]

    factor = (sample_rate / 2.0) / fft_bins
    filter_centers = np.round(filter_centers / factor).astype(int)
    filter_starts = np.round(filter_starts / factor).astype(int)
    filter_starts = np.minimum(filter_starts, filter_centers - 1)
    filter_ends = np.round(filter_ends / factor).astype(int)
    filter_ends = np.maximum(filter_ends, filter_centers + 1)

    filters = {num: [] for num in range(len(fundamentals))}

    for index, filt_start in np.ndenumerate(filter_starts):
        filt_end = filter_ends[index]
        filt_center = filter_centers[index]

        params = {'start': filt_start, 'center': filt_center, 'stop': filt_end,
                  'norm': False}

        filt = filter_type(**params) * filter_weights[index[0]]

        filters[index[1]] += [filt]

    return multi_filterbank(filters, fft_bins, len(filters), True)


def triang_filterbank(frequencies, fft_bins, sample_rate, norm=NORM_FILTER, omit_duplicates=OMIT_DUPLICATES):
    """
    Creates a filterbank with triangular filters.

    :param frequencies: a list of frequencies used for filter creation [Hz]
    :param fft_bins:    number of fft bins
    :param sample_rate: sample rate of the audio signal [Hz]
    :param norm:        normalize the area of the filters to 1 [default=True]
    :param omit_duplicates: omit duplicate filters resulting from insufficient
        resolution of low frequencies [default=True]
    :returns:           filterbank

    Note: each filter is characterized by 3 frequencies, the start, center and
          stop frequency. Thus the frequencies array must contain the first
          starting frequency, all center frequencies and the last stopping
          frequency.

    """
    return filterbank(triang_filter, frequencies, fft_bins, sample_rate, norm,
                      omit_duplicates, overlap=True)


def rectang_filterbank(frequencies, fft_bins, sample_rate, norm=NORM_FILTER, omit_duplicates=OMIT_DUPLICATES):
    """
    Creates a filterbank with rectangular filters.

    :param frequencies: a list of frequencies used for filter creation [Hz]
    :param fft_bins:    number of fft bins
    :param sample_rate: sample rate of the audio signal [Hz]
    :param norm:        normalize the area of the filters to 1 [default=True]
    :param omit_duplicates: omit duplicate filters resulting from insufficient
        resolution of low frequencies [default=True]
    :returns:           filterbank

    """
    return filterbank(rectang_filter, frequencies, fft_bins, sample_rate, norm,
                      omit_duplicates, overlap=False)


def triang_harmonic_filterbank(fundamentals, num_harmonics, fft_bins, sample_rate,
                               harmonic_envelope=HARMONIC_ENVELOPE, harmonic_width=HARMONIC_WIDTH,
                               inharmonicity_coeff=INHARMONICITY_COEFF):
    """
    Creates a harmonic filterbank with triangular filters.
    See harmonic_filterbank for a description of the parameters.
    """
    return harmonic_filterbank(triang_filter, fundamentals, num_harmonics, fft_bins, 
                               sample_rate, harmonic_envelope, harmonic_width,
                               inharmonicity_coeff)


def rectang_harmonic_filterbank(fundamentals, num_harmonics, fft_bins, sample_rate,
                                harmonic_envelope=HARMONIC_ENVELOPE, harmonic_width=HARMONIC_WIDTH,
                                inharmonicity_coeff=INHARMONICITY_COEFF):
    """
    Creates a harmonic filterbank with rectangular filters.
    See harmonic_filterbank for a description of the parameters.
    """
    return harmonic_filterbank(rectang_filter, fundamentals, num_harmonics, fft_bins, 
                               sample_rate, harmonic_envelope, harmonic_width,
                               inharmonicity_coeff)


class Filter(np.ndarray):

    def __new__(cls, data, sample_rate):
        # input is an numpy ndarray instance
        if isinstance(data, np.ndarray):
            # cast as Filter
            obj = np.asarray(data).view(cls)
        else:
            raise TypeError("wrong input data for Filter")
        # set attributes
        obj.__fft_bins, obj.__bands = obj.shape
        obj.__sample_rate = sample_rate
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.__fft_bins = getattr(obj, '__fft_bins')
        self.__bands = getattr(obj, '__bands')

    @property
    def fft_bins(self):
        return self.__fft_bins

    @property
    def bands(self):
        return self.__bands

    @property
    def sample_rate(self):
        return self.__sample_rate

    @property
    def bin_freqs(self):
        return fft_freqs(self.fft_bins, self.sample_rate)

    @property
    def fmin(self):
        return self.bin_freqs[np.nonzero(self)[0][0]]

    @property
    def fmax(self):
        return self.bin_freqs[np.nonzero(self)[0][-1]]


class MelFilter(Filter):
    """
    Mel Filter Class.

    """
    def __new__(cls, fft_bins, sample_rate, fmin=FMIN, fmax=FMAX, bands=MEL_BANDS, norm=NORM_FILTER, omit_duplicates=OMIT_DUPLICATES):
        """
        Creates a new Mel Filter object instance.

        :param fft_bins:    number of FFT bins (= half the window size of the FFT)
        :param sample_rate: sample rate of the audio file [Hz]
        :param fmin:        the minimum frequency [Hz, default=30]
        :param fmax:        the maximum frequency [Hz, default=16000]
        :param bands:       number of filter bands [default=40]
        :param norm:        normalize the area of the filter to 1 [default=True]
        :param omit_duplicates: omit duplicate filters resulting from
            insufficient resolution of low frequencies [default=True]

        """
        # get a list of frequencies
        # request 2 more bands, becuase these are the edge frequencies
        frequencies = mel_frequencies(bands + 2, fmin, fmax)
        # create filterbank
        filterbank = triang_filterbank(frequencies, fft_bins, sample_rate, norm, omit_duplicates)
        # cast to Filter
        obj = Filter.__new__(cls, filterbank, sample_rate)
        # set additional attributes
        obj.__norm = norm
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.__norm = getattr(obj, '__norm', NORM_FILTER)

    @property
    def norm(self):
        return self.__norm


class BarkFilter(Filter):
    """
    Bark Filter CLass.

    """
    def __new__(cls, fft_bins, sample_rate, fmin=FMIN, fmax=FMAX, double=BARK_DOUBLE, norm=NORM_FILTER, omit_duplicates=OMIT_DUPLICATES):
        """
        Creates a new Bark Filter object instance.

        :param fft_bins:    number of FFT bins (= half the window size of the FFT)
        :param sample_rate: sample rate of the audio file [Hz]
        :param fmin:        the minimum frequency [Hz, default=20]
        :param fmax:        the maximum frequency [Hz, default=15500]
        :param double:      double the number of frequency bands [default=False]
        :param norm:        normalize the area of the filter to 1 [default=True]
        :param omit_duplicates: omit duplicate filters resulting from
            insufficient resolution of low frequencies [default=True]

        """
        # get a list of frequencies
        if double:
            frequencies = bark_double_frequencies(fmin, fmax)
        else:
            frequencies = bark_frequencies(fmin, fmax)
        # create filterbank
        filterbank = triang_filterbank(frequencies, fft_bins, sample_rate, norm, omit_duplicates)
        # cast to Filter
        obj = Filter.__new__(cls, filterbank, sample_rate)
        # set additional attributes
        obj.__norm = norm
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.__norm = getattr(obj, '__norm', NORM_FILTER)

    @property
    def norm(self):
        return self.__norm


# TODO: this is very similar to the Cent-Scale. Unify it?
class LogarithmicFilter(Filter):
    """
    Logarithmic Filter class.

    """
    def __new__(cls, fft_bins, sample_rate,
                bands_per_octave=BANDS_PER_OCTAVE, fmin=FMIN, fmax=FMAX,
                norm=NORM_FILTER, omit_duplicates=OMIT_DUPLICATES, a4=A4):
        """
        Creates a new Logarithmic Filter object instance.

        :param fft_bins:         number of FFT bins (= half the window size of the FFT)
        :param sample_rate:      sample rate of the audio file [Hz]
        :param bands_per_octave: number of filter bands per octave [default=6]
        :param fmin:             the minimum frequency [Hz, default=20]
        :param fmax:             the maximum frequency [Hz, default=17000]
        :param norm:             normalize the area of the filter to 1 [default=True]
        :param omit_duplicates: omit duplicate filters resulting from
            insufficient resolution of low frequencies [default=True]
        :param a4:               tuning frequency of A4 [Hz, default=440]

        """
        # get a list of frequencies
        frequencies = log_frequencies(bands_per_octave, fmin, fmax, a4)
        # create filterbank
        filterbank = triang_filterbank(frequencies, fft_bins, sample_rate, norm, omit_duplicates)
        # cast to Filter
        obj = Filter.__new__(cls, filterbank, sample_rate)
        # set additional attributes
        obj.__bands_per_octave = bands_per_octave
        obj.__norm = norm
        obj.__a4 = a4
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.__bands_per_octave = getattr(obj, '__bands_per_octave', BANDS_PER_OCTAVE)
        self.__norm = getattr(obj, '__norm', NORM_FILTER)
        self.__a4 = getattr(obj, '__a4', A4)

    @property
    def bands_per_octave(self):
        return self.__bands_per_octave

    @property
    def norm(self):
        return self.__norm

    @property
    def a4(self):
        return self.__a4

# alias
LogFilter = LogarithmicFilter


class SemitoneFilter(LogarithmicFilter):
    """
    Semitone Filter class.

    """
    def __new__(cls, fft_bins, sample_rate,
                fmin=FMIN, fmax=FMAX, norm=NORM_FILTER, omit_duplicates=OMIT_DUPLICATES, a4=A4):
        """
        Creates a new Semitone Filter object instance.

        :param fft_bins:    number of FFT bins (= half the window size of the FFT)
        :param sample_rate: sample rate of the audio file [Hz]
        :param fmin:        the minimum frequency [Hz, default=27]
        :param fmax:        the maximum frequency [Hz, default=17000]
        :param norm:        normalize the area of the filter to 1 [default=True]
        :param omit_duplicates: omit duplicate filters resulting from
            insufficient resolution of low frequencies [default=True]
        :param a4:          tuning frequency of A4 [Hz, default=440]

        """
        # return a LogarithmicFilter with 12 bands per octave
        return LogarithmicFilter.__new__(cls, fft_bins, sample_rate, 12, fmin, fmax, norm, omit_duplicates, a4)


class SimpleChromaFilter(Filter):
    # FIXME: check if the result is the one expected
    """
    A simple chroma filter. Each frequency bin of a magnitude spectrum
    is assigned a chroma class, and all it's contents are added to this class.
    No diffusion, just discrete assignment.
    """

    def __new__(cls, fft_bins, sample_rate,
                fmin=FMIN, fmax=FMAX, norm=NORM_FILTER, a4=A4):
        """
        Creates a new Chroma Filter object instance.

        :param fft_bins:    number of FFT bins (= half the window size of the FFT)
        :param sample_rate: sample rate of the audio file [Hz]
        :param fmin:        the minimum frequency [Hz, default=20]
        :param fmax:        the maximum frequency [Hz, default=15500]
        :param norm:        normalize the area of the filter to 1 [default=True]
        :param a4:          tuning frequency of A4 [Hz, default=440]

        """
        # get a list of frequencies
        frequencies = semitone_frequencies(fmin, fmax, a4)
        # conversion factor for mapping of frequencies to spectrogram bins
        factor = (sample_rate / 2.0) / fft_bins
        # map the frequencies to the spectrogram bins
        frequencies = np.round(np.asarray(frequencies) / factor).astype(int)
        # filter out all frequencies outside the valid range
        frequencies = frequencies[frequencies < fft_bins]
        # init the filter matrix with size: fft_bins x filter bands
        filterbank = np.zeros([fft_bins, 12])
        # process all bands
        for band in range(len(frequencies) - 1):
            # edge frequencies
            # the start bin is included in the filter,
            # the stop bin is not (=start bin of the next filter)
            start = frequencies[band]
            stop = frequencies[band + 1] + 1
            # set the height of the filter
            height = 1.
            # normalize the area of the filter
            if norm:
                # a standard filter is at least 2 bins wide, and stop - start = 1
                # thus the filter has an area of 1 if the height is set to
                height /= (stop - start)
            # create a rectangular filter and map it to the 12 bins
            filterbank[start:stop, band % 12] = height
        # cast to Filter
        obj = Filter.__new__(cls, filterbank, sample_rate)
        # set additional attributes
        obj.__norm = norm
        obj.__a4 = a4
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.__norm = getattr(obj, '__norm', NORM_FILTER)
        self.__a4 = getattr(obj, '__a4', A4)

    @property
    def norm(self):
        return self.__norm

    @property
    def a4(self):
        return self.__a4

### original code from Filip
#    def __init__(self, fft_length, sample_rate, normalise=False):
#        from .. import freq_scales as fs
#        """
#        Initialises the computation.
#
#        :Parameters:
#          - `fft_length`: Specifies the FFT length used to obtain the magnitude
#                          spectrum
#          - `sample_rate`: Sample rate of the audio
#          - `normalise`: Specifies if the chroma vectors shall be normalised,
#                         i.e. divided by it's sum
#        """
#        super(SimpleChromaComputer, self).__init__(normalise)
#
#        mag_spec_length = fft_length / 2 + 1
#        max_note = np.floor(fs.hz_to_midi(sample_rate / 2))
#        note_freqs = fs.midi_to_hz(np.arange(0, max_note))
#        fft_freqs = abs(np.fft.fftfreq(fft_length) * sample_rate)[:mag_spec_length]
#
#        note_to_fft_distances = abs(note_freqs[:, np.newaxis] - fft_freqs)
#        note_assignments = np.argmin(note_to_fft_distances, axis=0) % 12
#
#        self.bin_assignments = np.mgrid[:12, :mag_spec_length][0] == note_assignments



