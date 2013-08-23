#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2012-2013 Sebastian Böck <sebastian.boeck@jku.at>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np


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
    Generates a list of corner frequencies aligned on the Mel scale.

    :param bands: number of bands
    :param fmin: the minimum frequency [Hz]
    :param fmax: the maximum frequency [Hz]
    :returns: a list of frequencies

    """
    # convert fmin and fmax to the Mel scale
    mmin = hz2mel(fmin)
    mmax = hz2mel(fmax)
    # calculate the width of each Mel filter
    mwidth = (mmax - mmin) / (bands + 1)
    # create a list of frequencies
    frequencies = []
    for i in range(bands + 2):
        frequencies.append(mel2hz(mmin + mwidth * i))
    return frequencies


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

    """
    # frequencies aligned to the Bark-scale
    f = [20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000,
         2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]
    # filter frequencies
    f = f[f >= fmin]
    f = f[f <= fmax]
    # return
    return f


def bark_double_frequencies(fmin=20, fmax=15500):
    """
    Generates a list of corner frequencies aligned on the Bark-scale.
    The list includes also center frequencies between the corner frequencies.

    """
    # frequencies aligned to the Bark-scale, does also including center frequencies
    f = [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 510, 570, 630, 700, 770,
         840, 920, 1000, 1080, 1170, 1270, 1370, 1480, 1600, 1720, 1850, 2000,
         2150, 2320, 2500, 2700, 2900, 3150, 3400, 3700, 4000, 4400, 4800, 5300,
         5800, 6400, 7000, 7700, 8500, 9500, 10500, 12000, 13500, 15500]
    # filter frequencies
    f = f[f >= fmin]
    f = f[f <= fmax]
    # return
    return f

# (pseudo) Constant-Q frequency scale

## FIXME: this is NOT faster!
#def cq_frequencies(bands, fmin, fmax, a=440):
#    emin = np.floor(np.log2(float(fmin) / a))
#    emax = np.ceil(np.log2(float(fmax) / a))
#    freqs = np.logspace(emin, emax, (emax - emin) * bands, base=2, endpoint=False) * a
#    return [x for x in freqs if (x >= fmin and x <= fmax)]


def cq_frequencies(bands_per_octave, fmin, fmax, a4=440):
    """
    Generates a list of frequencies aligned on a logarithmic frequency scale.

    :param bands_per_octave: number of filter bands per octave
    :param fmin: the minimum frequency [Hz]
    :param fmax: the maximum frequency [Hz]
    :param a4: tuning frequency of A4 [Hz, default=440]
    :returns: a list of frequencies

    Note: frequencies are aligned to MIDI notes with the default a4=440 and
    12 bands_per_octave.

    """
    # factor by which 2 frequencies are located apart from each other
    factor = 2.0 ** (1.0 / bands_per_octave)
    # start with A4
    freq = a4
    frequencies = []
    # go upwards till fmax
    while freq <= fmax:
        frequencies.append(freq)
        freq *= factor
    # restart with a and go downwards till fmin
    freq = a4 / factor
    while freq >= fmin:
        frequencies.append(freq)
        freq /= factor
    # sort frequencies
    frequencies.sort()
    # return the list
    return frequencies


def log_frequencies(bands_per_octave, fmin=27.5, fmax=17000, a4=440):
    """
    Generates a list of frequencies aligned on a logarithmic frequency scale.

    :param bands_per_octave: number of filter bands per octave
    :param fmin: the minimum frequency [Hz, default=27.5]
    :param fmax: the maximum frequency [Hz, default=17000]
    :param a4: tuning frequency of A4 [Hz, default=440]
    :returns: a list of frequencies

    Note: if 12 bands per octave and a4=440 are used, the frequencies are
          equivalent to MIDI notes.

    """
    # get the range
    left = np.floor(np.log2(float(fmin) / a4) * bands_per_octave)
    right = np.ceil(np.log2(float(fmax) / a4) * bands_per_octave)
    # generate frequencies
    freqs = a4 * 2 ** (np.arange(left, right) / float(bands_per_octave))
    # filter frequencies
    # (needed, because range might be bigger because of the use of floor/ceil)
    freqs = freqs[freqs >= fmin]
    freqs = freqs[freqs <= fmax]
    # return
    return freqs


def semitone_frequencies(fmin, fmax, a4=440):
    """
    Generates a list of frequencies separated by semitones.

    :param fmin: the minimum frequency [Hz]
    :param fmax: the maximum frequency [Hz]
    :param a4: tuning frequency of A4 [Hz, default=440]
    :returns: a list of frequencies of semitones

    Note: frequencies are aligned to MIDI notes with the default a4=440.

    """
    # return MIDI frequencies
    return cq_frequencies(12, fmin, fmax, a4)


# MIDI
def midi2hz(m, a4=440):
    """
    Convert frequencies to the corresponding MIDI notes.

    :param m: input MIDI notes
    :param a4: tuning frequency of A4 [Hz, default=440]
    :returns: frequencies in Hz

    For details see: http://www.phys.unsw.edu.au/jw/notes.html

    """
    return 2. ** ((m - 69.) / 12.) * a4


def hz2midi(f, a4=440):
    """
    Convert MIDI notes to corresponding frequencies.

    :param f: input frequencies [Hz]
    :param a4: tuning frequency of A4 [Hz, default=440]
    :returns: MIDI notes

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
def fft_freqs(fft_bins, fs):
    # faster than: np.fft.fftfreq(fft_bins * 2)[:fft_bins] * fs
    return np.linspace(0, fs / 2., fft_bins + 1)


def triang_filter(start, center, stop, norm=False):
    """
    Calculate a triangular window of the given size.

    :param start: starting bin (with value 0, included in the returned filter)
    :param center: center bin (of height 1, unless norm is True)
    :param stop: end bin (with value 0, not included in the returned filter)
    :param norm: normalize the area of the filter to 1 [default=False]
    :returns: a triangular shaped filter with height 1

    """
    # set the height of the filter
    height = 1.
    # normalize the area of the filter
    if norm:
        # a standard filter is at least 3 bins wide, and stop - start = 2
        # thus the filter has an area of 1 if the height is set to
        height = 2. / (stop - start)
    # init the filter
    triang_filter = np.empty(stop - start)
    # rising edge (without the center)
    triang_filter[:center - start] = np.linspace(0, height, (center - start), endpoint=False)
    # falling edge (including the center, but without the last bin since it's 0)
    triang_filter[center - start:] = np.linspace(height, 0, (stop - center), endpoint=False)
    # return
    return triang_filter


def triang_filterbank(frequencies, fft_bins, fs, norm=True):
    """
    Creates a filterbank with overlapping triangular filters.

    :param frequencies: a list of frequencies used for filter creation [Hz]
    :param fft_bins: number of fft bins
    :param fs: sample rate of the audio signal [Hz]
    :param norm: normalize the area of the filters to 1 [default=True]
    :returns: the filterbank

    Note: each filter is characterized by 3 frequencies, the start, center and
    stop frequency. Thus the frequencies array must contain the first starting
    frequency, all center frequencies and the last stopping frequency.

    """
    # conversion factor for mapping of frequencies to spectrogram bins
    factor = (fs / 2.0) / fft_bins
    # map the frequencies to the spectrogram bins
    frequencies = np.round(np.asarray(frequencies) / factor).astype(int)
    # filter out all frequencies outside the valid range
    frequencies = [f for f in frequencies if f < fft_bins]
    # only keep unique bins
    # Note: this is important to do so, otherwise the lower frequency bins are
    # given too much weight if simply summed up (as in the spectral flux)
    frequencies = np.unique(frequencies)
    # number of bands
    bands = len(frequencies) - 2
    if bands < 3:
        raise ValueError("cannot create filterbank with less than 3 frequencies")
    # init the filter matrix with size: fft_bins x filter bands
    filterbank = np.zeros([fft_bins, bands])
    # process all bands
    for band in range(bands):
        # edge & center frequencies
        start, mid, stop = frequencies[band:band + 3]
        # create a triangular filter
        filterbank[start:stop, band] = triang_filter(start, mid, stop, norm)
    # return the filterbank
    return filterbank


def rectang_filterbank(frequencies, fft_bins, fs, norm=True):
    """
    Creates a filterbank with rectangular filters.

    :param frequencies: a list of frequencies used for filter creation [Hz]
    :param fft_bins: number of fft bins
    :param fs: sample rate of the audio signal [Hz]
    :param norm: normalize the area of the filters to 1 [default=True]
    :returns: the filterbank

    """
    # conversion factor for mapping of frequencies to spectrogram bins
    factor = (fs / 2.0) / fft_bins
    # map the frequencies to the spectrogram bins
    frequencies = np.round(np.asarray(frequencies) / factor).astype(int)
    # filter out all frequencies outside the valid range
    frequencies = [f for f in frequencies if f < fft_bins]
    # only keep unique bins
    # Note: this is important to do so, otherwise the lower frequency bins are
    # given too much weight if simply summed up (as in the spectral flux)
    frequencies = np.unique(frequencies)
    # number of bands
    bands = len(frequencies) - 1
    if bands < 2:
        raise ValueError("cannot create filterbank with less than 2 frequencies")
    # init the filter matrix with size: fft_bins x filter bands
    filterbank = np.zeros([fft_bins, bands])
    # process all bands
    for band in range(bands):
        # edge frequencies
        # the start bin is included in the filter,
        # the stop bin is not (=start bin of the next filter)
        start, stop = frequencies[band:band + 1]
        # set the height of the filter
        height = 1.
        # normalize the area of the filter
        if norm:
            # a standard filter is at least 2 bins wide, and stop - start = 1
            # thus the filter has an area of 1 if the height is set to
            height /= (stop - start)
        # create a rectangular filter
        filterbank[start:stop, band] = height
    # return the filterbank
    return filterbank


class Filter(np.ndarray):

    def __new__(cls, data, fs):
        # input is an numpy ndarray instance
        if isinstance(data, np.ndarray):
            # cast as Wav
            obj = np.asarray(data).view(cls)
            # can't set sample rate, default values are set in __array_finalize__
        else:
            raise TypeError("wrong input data for Filter")
        # set attributes
        obj.__fft_bins, obj.__bands = obj.shape
        obj.__fs = fs
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
    def fs(self):
        return self.__fs

    @property
    def bin_freqs(self):
        return fft_freqs(self.fft_bins, self.fs)

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
    def __new__(cls, fft_bins, fs, fmin=30, fmax=16000, bands=40, norm=True):
        """
        Creates a new Mel Filter object instance.

        :param fft_bins: number of FFT bins (= half the window size of the FFT)
        :param fs: sample rate of the audio file [Hz]
        :param fmin: the minimum frequency [Hz, default=30]
        :param fmax: the maximum frequency [Hz, default=16000]
        :param bands: number of filter bands [default=40]
        :param norm: normalize the area of the filter to 1 [default=True]

        """
        # get a list of frequencies
        frequencies = mel_frequencies(bands, fmin, fmax)
        # create filterbank
        filterbank = triang_filterbank(frequencies, fft_bins, fs, norm)
        # cast to Filter type
        obj = Filter.__new__(cls, filterbank, fs)
        # set additional attributes
        obj.__norm = norm
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.__norm = getattr(obj, '__norm', True)

    @property
    def norm(self):
        return self.__norm


class BarkFilter(Filter):
    """
    Bark Filter CLass.

    """
    def __new__(cls, fft_bins, fs, fmin=20, fmax=15500, double=False, norm=True):
        """
        Creates a new Bark Filter object instance.

        :param fft_bins: number of FFT bins (= half the window size of the FFT)
        :param fs: sample rate of the audio file [Hz]
        :param fmin: the minimum frequency [Hz, default=20]
        :param fmax: the maximum frequency [Hz, default=15500]
        :param double: double the number of frequency bands [default=False]
        :param norm: normalize the area of the filter to 1 [default=True]

        """
        # get a list of frequencies
        if double:
            frequencies = bark_double_frequencies(fmin, fmax)
        else:
            frequencies = bark_frequencies(fmin, fmax)
        # create filterbank
        filterbank = triang_filterbank(frequencies, fft_bins, fs, norm)
        # cast to Filter type
        obj = Filter.__new__(cls, filterbank, fs)
        # set additional attributes
        obj.__norm = norm
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.__norm = getattr(obj, '__norm', True)

    @property
    def norm(self):
        return self.__norm


# TODO: this is very similar to the Cent-Scale. Unify it?
class LogFilter(Filter):
    """
    Logarithmic Filter class.

    """
    def __new__(cls, fft_bins, fs, bands_per_octave=6, fmin=20, fmax=17000, norm=True, a4=440):
        """
        Creates a new Logarithmic Filter object instance.

        :param fft_bins: number of FFT bins (= half the window size of the FFT)
        :param fs: sample rate of the audio file [Hz]
        :param bands_per_octave: number of filter bands per octave [default=6]
        :param fmin: the minimum frequency [Hz, default=20]
        :param fmax: the maximum frequency [Hz, default=17000]
        :param norm: normalize the area of the filter to 1 [default=True]
        :param a4: tuning frequency of A4 [Hz, default=440]

        """
        # get a list of frequencies
        frequencies = log_frequencies(bands_per_octave, fmin, fmax, a4)
        # create filterbank
        filterbank = triang_filterbank(frequencies, fft_bins, fs, norm)
        # cast to Filter type
        obj = Filter.__new__(cls, filterbank, fs)
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
        self.__bands_per_octave = getattr(obj, '__bands_per_octave', 6)
        self.__norm = getattr(obj, '__norm', True)
        self.__a4 = getattr(obj, '__a4', 440)

    @property
    def bands_per_octave(self):
        return self.__bands_per_octave

    @property
    def norm(self):
        return self.__norm

    @property
    def a4(self):
        return self.__a4


class SemitoneFilter(LogFilter):
    """
    Semitone Filter class.

    """
    def __new__(cls, fft_bins, fs, fmin=27, fmax=17000, norm=True, a4=440):
        """
        Creates a new Semitone Filter object instance.

        :param fft_bins: number of FFT bins (= half the window size of the FFT)
        :param fs: sample rate of the audio file [Hz]
        :param fmin: the minimum frequency [Hz, default=27]
        :param fmax: the maximum frequency [Hz, default=17000]
        :param norm: normalize the area of the filter to 1 [default=True]
        :param a4: tuning frequency of A4 [Hz, default=440]

        """
        # return a LogFilter with 12 bands per octave
        return LogFilter.__new__(cls, fft_bins, fs, 12, fmin, fmax, norm, a4)


class ChromaFilter(Filter):
    # FIXME: check if the result is the one expected
    """
    A simple chroma filter. Each frequency bin of a magnitude spectrum
    is assigned a chroma class, and all it's contents are added to this class.
    No diffusion, just discrete assignment.
    """

    def __new__(cls, fft_bins, fs, fmin=20, fmax=15500, norm=True, a4=440):
        """
        Creates a new Chroma Filter object instance.

        :param fft_bins: number of FFT bins (= half the window size of the FFT)
        :param fs: sample rate of the audio file [Hz]
        :param fmin: the minimum frequency [Hz, default=20]
        :param fmax: the maximum frequency [Hz, default=15500]
        :param norm: normalize the area of the filter to 1 [default=True]
        :param a4: tuning frequency of A4 [Hz, default=440]

        """
        # get a list of frequencies
        frequencies = semitone_frequencies(fmin, fmax, a4)
        # conversion factor for mapping of frequencies to spectrogram bins
        factor = (fs / 2.0) / fft_bins
        # map the frequencies to the spectrogram bins
        frequencies = np.round(np.asarray(frequencies) / factor).astype(int)
        # filter out all frequencies outside the valid range
        frequencies = [f for f in frequencies if f < fft_bins]
        # init the filter matrix with size: fft_bins x filter bands
        filterbank = np.zeros([fft_bins, 12])
        # process all bands
        for band in range(frequencies):
            # edge frequencies
            # the start bin is included in the filter,
            # the stop bin is not (=start bin of the next filter)
            start, stop = frequencies[band:band + 1]
            # set the height of the filter
            height = 1.
            # normalize the area of the filter
            if norm:
                # a standard filter is at least 2 bins wide, and stop - start = 1
                # thus the filter has an area of 1 if the height is set to
                height /= (stop - start)
            # create a rectangular filter and map it to the 12 bins
            filterbank[start:stop, band % 12] = height
        # cast filterbank to Filter type
        obj = Filter.__new__(cls, filterbank, fs)
        # set additional attributes
        obj.__norm = norm
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.__norm = getattr(obj, '__norm', True)

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
