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


def hz2mel(f):
    """Convert Hz to Mel."""
    return 1127.01048 * np.log(f / 700. + 1.)


def mel2hz(m):
    """Convert Mel to Hz."""
    return 700. * (np.exp(m / 1127.01048) - 1.)


def mel_frequencies(bands, fmin, fmax):
    """
    Generates a list of corner frequencies aligned on the Mel-scale.

    :param bands: number of bands
    :param fmin: the minimum frequency [in Hz]
    :param fmax: the maximum frequency [in Hz]
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


def hz2bark(f):
    """Convert Hz to Bark."""
    # use Zwicker' formula:
    # return 13. * arctan(0.00076*f) + 3.5 * arctan((f/7500.)**2)
    return (26.81 / (1 + 1960 / f)) - 0.53


def bark2hz(z):
    """Convert Bark to Hz."""
    # use Zwicker's formula:
    # return 13. * arctan(0.00076*f) + 3.5 * arctan((f/7500.)**2)
    return 1960. / (26.81 / (z + 0.53) - 1)


def bark_frequencies():
    """
    Generates a list of corner frequencies aligned on the Bark-scale.

    """
    # frequencies aligned to the Bark-scale
    f = [20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000,
         2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]
    return f


def bark_double_frequencies():
    """
    Generates a list of corner frequencies aligned on the Bark-scale.
    The list includes also center frequencies between the corner frequencies.

    """
    # frequencies aligned to the Bark-scale, does also including center frequencies
    f = [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 510, 570, 630, 700, 770,
         840, 920, 1000, 1080, 1170, 1270, 1370, 1480, 1600, 1720, 1850, 2000,
         2150, 2320, 2500, 2700, 2900, 3150, 3400, 3700, 4000, 4400, 4800, 5300,
         5800, 6400, 7000, 7700, 8500, 9500, 10500, 12000, 13500, 15500]
    return f

## FIXME: this is NOT faster!
#def cq_frequencies(bands, fmin, fmax, a=440):
#    emin = np.floor(np.log2(float(fmin) / a))
#    emax = np.ceil(np.log2(float(fmax) / a))
#    freqs = np.logspace(emin, emax, (emax - emin) * bands, base=2, endpoint=False) * a
#    return [x for x in freqs if (x >= fmin and x <= fmax)]


def cq_frequencies(bands, fmin, fmax, a=440):
    """
    Returns a list of frequencies aligned on a logarithmic scale.

    :param bands: number of filter bands per octave
    :param fmin: the minimum frequency [in Hz]
    :param fmax: the maximum frequency [in Hz]
    :param a: tuning frequency of A0 [in Hz, default = 440]
    :returns: a list of frequencies

    When using 12 bands per octave and a=440 as the tuning frequency,
    the returned frequencies correspond to those of MIDI notes.

    """
    # factor 2 frequencies are apart
    factor = 2.0 ** (1.0 / bands)
    # start with A0
    freq = a
    frequencies = []
    # go upwards till fmax
    while freq <= fmax:
        frequencies.append(freq)
        freq *= factor
    # restart with a and go downwards till fmin
    freq = a / factor
    while freq >= fmin:
        frequencies.append(freq)
        freq /= factor
    # sort frequencies
    frequencies.sort()
    # return the list
    return frequencies


def semitone_frequencies(fmin, fmax, a=440):
    """
    Returns a list of frequencies separated by semitones.

    :param fmin: the minimum frequency [in Hz]
    :param fmax: the maximum frequency [in Hz]
    :param a: tuning frequency of A0 [in Hz, default = 440]
    :returns: a list of frequencies of semitones

    NOTE: frequencies are aligned to MIDI notes.

    """
    # return MIDI frequencies
    return cq_frequencies(12, fmin, fmax, a)

# provide an alias
midi_frequencies = semitone_frequencies


def hz2erb(f):
    """
    Convert Hz to ERB.

    Information about the ERB scale can be found at
    https://ccrma.stanford.edu/~jos/bbt/Equivalent_Rectangular_Bandwidth.html

    :Parameters:
      - `f`: Input frequencies in Hz

    :Returns:
        ERB-scaled frequencies

    """
    return 21.4 * np.log10(1 + 4.37 * f / 1000)


def erb2hz(e):
    """
    Convert ERB scaled frequencies to Hz

    Information about the ERB scale can be found at
    https://ccrma.stanford.edu/~jos/bbt/Equivalent_Rectangular_Bandwidth.html

    :Parameters:
      - `e`: ERB scaled input frequencies

    :Returns:
        Frequencies in Hz
    """
    return (10 ** (e / 21.4) - 1) * 1000 / 4.37


def midi2hz(m, a4=440.0):
    """
    Returns the frequencies of the corresponding MIDI note ids

    For details take a look at http://www.phys.unsw.edu.au/jw/notes.html

    :Parameters:
      - `m`: MIDI note ids
      - `a4`: Frequency of the concert pitch

    :Returns:
        Frequencies corresponding the the MIDI note ids
    """
    return 2 ** ((m - 69.0) / 12) * a4


def hz2midi(f, a4=440.0):
    """
    Returns the MIDI note ids corresponding to frequencies

    For details take a look at http://www.phys.unsw.edu.au/jw/notes.html

    Note that if this function does not necessarily return a valid
    MIDI note id, you may need to round it to the nearest integer.

    :Parameters:
      - `f`: Input frequencies
      - `a4`: Frequency of the concert pitch

    :Returns:
        MIDI note ids corresponding to the frequencies
    """
    return (12 * np.log2(f / a4)) + 69


def triang_filter(start, mid, stop, norm=False, height=1.):
    """
    Calculate a triangular window of the given size.

    :param start: starting bin (with value 0, included in the returned filter)
    :param mid: center bin (of height 1, unless norm is True)
    :param stop: end bin (with value 0, not included in the returned filter)
    :param norm: normalize the area of the filter to 1 [default = False]
    :param height: height of the filter [default = 1.0]
    :returns: a triangular shaped filter

    """
    # normalize the height
    if norm:
        height = 2. / (stop - start)
    # init the filter
    triang_filter = np.empty(stop - start)
    # rising edge (without the centre)
    triang_filter[:mid - start] = np.linspace(0, height, (mid - start), endpoint=False)
    # falling edge (including the centre, but without the last bin since it's 0)
    triang_filter[mid - start:] = np.linspace(height, 0, (stop - mid), endpoint=False)
    # return
    return triang_filter


def triang_filterbank(frequencies, ffts, fs, norm=True):
    """
    Creates a filterbank with triangular filters.

    :param frequencies: a list of frequencies used for filter creation
    :param ffts: number of fft bins
    :param fs: sample rate
    :param norm: normalize the area of the filter to 1 [default = True]

    Note: each filter is characterized by 3 frequencies, the start, mid and
    stop frequency. Thus the frequencies array must contain the first starting
    frequency, all center frequencies and the last stopping frequency.

    """
    # conversion factor for mapping of frequencies to spectrogram bins
    factor = (fs / 2.0) / ffts
    # map the frequencies to the spectrogram bins
    frequencies = np.round(np.asarray(frequencies) / factor).astype(int)
    # only keep unique bins
    frequencies = np.unique(frequencies)
    # number of bands
    bands = len(frequencies) - 2
    if bands < 3:
        raise ValueError("cannot create filterbank with less than 3 frequencies")
    # init the filter matrix with size: ffts x filter bands
    filterbank = np.zeros([ffts, bands])
    # process all bands
    for band in range(bands):
        # edge & center frequencies
        start, mid, stop = frequencies[band:band + 3]
        # create a triangular filter
        filterbank[start:stop, band] = triang_filter(start, mid, stop, norm)
    # return the filterbank
    return filterbank


def rectang_filterbank(frequencies, ffts, fs, norm=True, height=1.0):
    """
    Creates a filterbank with rectangular filters.

    :param frequencies: a list of frequencies used for filter creation
    :param ffts: number of fft bins
    :param fs: sample rate
    :param height: height of the filter [default = 1.0]
    :param norm: normalize the area of the filter to height [default = True]

    """
    # conversion factor for mapping of frequencies to spectrogram bins
    factor = (fs / 2.0) / ffts
    # map the frequencies to the spectrogram bins
    frequencies = np.round(np.asarray(frequencies) / factor).astype(int)
    # only keep unique bins
    frequencies = np.unique(frequencies)
    # number of bands
    bands = len(frequencies) - 1
    if bands < 2:
        raise ValueError("cannot create filterbank with less than 2 frequencies")
    # init the filter matrix with size: ffts x filter bands
    filterbank = np.zeros([ffts, bands])
    # process all bands
    for band in range(bands):
        # edge frequencies
        start, stop = frequencies[band:band + 1]
        if norm:
            height /= (stop - start)
        # create a rectangular filter
        filterbank[start:stop, band] = height
    # return the filterbank
    return filterbank


class Filter(object):
    """Filter Class"""

    def __init__(self, ffts, fs, fmin=20, fmax=20000):
        """
        Creates a new Filter object instance.

        :param ffts: number of FFT coefficients (= half the window size of the FFT)
        :param fs: sample rate of the audio file [in Hz]
        :param fmin: the minimum frequency [in Hz, default = 20]
        :param fmax: the maximum frequency [in Hz, default = 20000]

        """
        self.ffts = ffts
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        # reduce fmax if necessary
        if self.fmax > self.fs / 2.:
            self.fmax = self.fs / 2.
        # init filterbank
        self.filterbank = None


class MelFilter(Filter):
    """Mel Filter CLass"""
    def __init__(self, ffts, fs, fmin=30, fmax=16000, bands=40, norm=True):
        """
        Creates a new Mel Filter object instance.

        :param ffts: number of FFT coefficients (= half the window size of the FFT)
        :param bands: number of filter bands [default = 40]
        :param fs: sample rate of the audio file [in Hz, default=44100]
        :param fmin: the minimum frequency [in Hz, default = 20]
        :param fmax: the maximum frequency [in Hz, default = 16000]
        :param norm: normalize the area of the filter to 1 [default = True]

        """
        # set variables of base class
        super(MelFilter, self).__init__(ffts, fs, fmin, fmax)
        # set variables
        self.bands = bands
        self.norm = norm
        # get a list of frequencies
        self.frequencies = mel_frequencies(bands, fmin, fmax)
        # create filterbank
        self.filterbank = triang_filterbank(self.frequencies, self.ffts, self.fs, self.norm)


class BarkFilter(Filter):
    """Bark Filter CLass."""
    def __init__(self, ffts, fs, fmin=20, fmax=15500, double=False, norm=True):
        """
        Creates a new Bark Filter object instance.

        :param ffts: number of FFT coefficients (= half the window size of the FFT)
        :param bands: number of filter bands [default = 40]
        :param fs: sample rate of the audio file [in Hz, default=44100]
        :param fmin: the minimum frequency [in Hz, default = 20]
        :param fmax: the maximum frequency [in Hz, default = 16000]
        :param norm: normalize the area of the filter to 1 [default = True]

        """
        # set variables of base class
        super(BarkFilter, self).__init__(ffts, fs, fmin, fmax)
        # set additional variables
        self.norm = norm
        # get a list of frequencies
        if double:
            self.frequencies = bark_double_frequencies(fmin, fmax)
        else:
            self.frequencies = bark_frequencies(fmin, fmax)
        # use only frequencies within [fmin, fmax]
        self.frequencies = [x for x in self.frequencies if (x >= fmin and x <= fmax)]
        # create filterbank
        self.filterbank = triang_filterbank(self.frequencies, self.ffts, self.fs, self.norm)

    @property
    def bands(self):
        """Number of bands."""
        # use ceil, so that no signal is truncated
        return self.filterbank.shape[1]


class CQFilter(Filter):
    """CQ Filter class."""
    def __init__(self, ffts, fs, bands_per_oktave=6, fmin=30, fmax=16000, norm=True):
        """
        Creates a new Constant-Q Filter object instance.

        :param ffts: number of FFT coefficients (= half the window size of the FFT)
        :param bands_per_oktave: number of filter bands per octave [default = 6]
        :param fs: sample rate of the audio file [in Hz, default=44100]
        :param fmin: the minimum frequency [in Hz, default = 30]
        :param fmax: the maximum frequency [in Hz, default = 16000]
        :param norm: normalize the area of the filter to 1 [default = True]

        """
        # set variables of base class
        super(CQFilter, self).__init__(ffts, fs, fmin, fmax)
        # set additional variables
        self.bands_per_oktave = bands_per_oktave
        self.norm = norm
        # get a list of frequencies
        self.frequencies = cq_frequencies(bands_per_oktave, fmin, fmax)
        # create filterbank
        self.filterbank = triang_filterbank(self.frequencies, self.ffts, self.fs, self.norm)

    @property
    def bands(self):
        """Number of bands."""
        # use ceil, so that no signal is truncated
        return self.filterbank.shape[1]


class SemitoneFilter(Filter):
    """Semitone Filter class."""
    def __init__(self, ffts, fs, fmin=27, fmax=17000, norm=True):
        """
        Creates a new Semitone Filter object instance.

        :param ffts: number of FFT coefficients (= half the window size of the FFT)
        :param fs: sample rate of the audio file [in Hz, default=44100]
        :param fmin: the minimum frequency [in Hz, default = 27]
        :param fmax: the maximum frequency [in Hz, default = 17000]
        :param norm: normalize the area of the filter to 1 [default = True]

        """
        # set variables of base class
        super(SemitoneFilter, self).__init__(ffts, fs, fmin, fmax)
        # set additional variables
        self.norm = norm
        # get a list of frequencies
        self.frequencies = semitone_frequencies(fmin, fmax)
        # create filterbank
        self.filterbank = triang_filterbank(self.frequencies, self.ffts, self.fs, self.norm)

    @property
    def bands(self):
        """Number of bands."""
        # use ceil, so that no signal is truncated
        return self.filterbank.shape[1]

# FIXME: this code is taken from Reini, make it nice and behave the same
#class CentFilter(Filter):
#    """Cent Scale Filter"""
#
#    @staticmethod
#    def hz2cent(f, a=440):
#        """Convert Hz to Cent."""
#        if f == 0:
#            return 0
#        return 1200.0 * np.log2(f / (a * 2. ** ((3 / 12.0) - 5)))
#
#    @staticmethod
#    def cent2hz(c, a=440):
#        """Convert Cent to Hz."""
#        return (a * 2. ** ((3 / 12.0) - 5)) * np.exp(c / 1200.)
#
#    def create_cent_filterbank(self, cent_start, cent_hop, num_linear_filters):
#        """Computes cent-scale filterbank bounds."""
#        # for each bin, the corresponding frequency expressed in Hz
#        #bins_hz = [(float(fs) / self.ffts) * i for i in np.arange(self.ffts / 2)]
#        bins_hz = np.fft.fftfreq(self.ffts)[:self.ffts / 2] * self.fs
#        bins_cent = map(self.hz2cent, bins_hz)
#        bins_cent[0] = float("-inf")
#
#        linear_filters = num_linear_filters
#        start = cent_start
#        upper_bounds = []
#        upper_bounds.append(0)
#        createLinearFilters = True
#        for i in range(self.ffts / 2):
#            if bins_cent[i] > start:
#                if createLinearFilters is True:
#                    sizeInBins = float(i + 1) / linear_filters
#                    cur = 0
#                    for _ in range(linear_filters - 1):
#                        cur = cur + sizeInBins
#                        upper_bounds.append(np.floor(cur))
#                    createLinearFilters = False
#                upper_bounds.append(i + 1)
#                start = start + cent_hop
#        return upper_bounds
