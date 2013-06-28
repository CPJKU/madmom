"""
Copyright (c) 2013 Sebastian Böck <sebastian.boeck@jku.at>
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
import scipy.ndimage as sim


def wraptopi(phase):
    """
    Wrap the phase information to the range -π...π.

    """
    return np.mod(phase + np.pi, 2.0 * np.pi) - np.pi


def diff(self, spec, diff_frames=1, pos=False):
    """
    Calculates the difference of the magnitude spectrogram.

    :param spec: the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame [default=1]
    :param pos: only keep positive values [default=False]

    """
    # init the matrix with 0s, the first N rows are 0 then
    # TODO: under some circumstances it might be helpful to init with the spec
    diff = np.zeros_like(spec)
    if diff_frames < 1:
        raise ValueError("number of diff_frames must be >= 1")
    # calculate the diff
    diff[diff_frames:] = spec[diff_frames:] - spec[0:-diff_frames]
    # keep only positive values
    if pos:
        diff = diff * (diff > 0)
    return diff


def max_diff(self, spec, diff_frames=1, pos=False, diff_bins=3):
    """
    Calculates the difference of bin k of the magnitude spectrogram relative to
    the N-th previous frame with a maximum filter (in the frequency axis) applied.

    :param spec: the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame [default=1]
    :param pos: only keep positive values [default=False]
    :param diff_bins: number of neighboring bins used for maximum filtering [default=3]

    Note: this method works only properly, if the spectrogram is filtered with
    a filterbank of the right frequency spacing. Filterbanks with 24 bands per
    octave (i.e. quartertone resolution) usually yield good results. With the
    default 3 diff_bins, the maximum of the bins k-1, k, k+1 of the frame
    diff_frames to the left is used for the calculation of the difference.

    """
    # init diff matrix
    diff = np.zeros_like(spec)
    if diff_frames < 1:
        raise ValueError("number of diff_frames must be >= 1")
    # calculate the diff
    diff[diff_frames:] = spec[diff_frames:] - sim.maximum_filter(spec, size=[1, diff_bins])[0:-diff_frames]
    # keep only positive values
    if pos:
        diff = diff * (diff > 0)
    return diff


def corr_diff(self, spec, diff_frames=1, pos=False, diff_bins=1):
    """
    Calculates the difference of the magnitude spectrogram relative to the
    N-th previous frame shifted in frequency to achieve the highest
    correlation between these two frames.

    :param spec: the magnitude spectrogram
    :param diff_frames: calculate the difference to the N-th previous frame [default=1]
    :param pos: only keep positive values [default=False]
    :param diff_bins: maximum number of bins shifted for correlation calculation [default=1]

    """
    # init diff matrix
    diff = np.zeros_like(spec)
    if diff_frames < 1:
        raise ValueError("number of diff_frames must be >= 1")
    # calculate the diff
    frames, bins = diff.shape
    corr = np.zeros((frames, diff_bins * 2 + 1))
    for f in range(diff_frames, frames):
        # correlate the frame with the previous one
        # resulting size = bins * 2 - 1
        c = np.correlate(spec[f], spec[f - diff_frames], mode='full')
        # save the middle part
        centre = len(c) / 2
        corr[f] = c[centre - diff_bins: centre + diff_bins + 1]
        # shift the frame for difference calculation according to the
        # highest peak in correlation
        bin_offset = diff_bins - np.argmax(corr[f])
        bin_start = diff_bins + bin_offset
        bin_stop = bins - 2 * diff_bins + bin_start
        diff[f, diff_bins:-diff_bins] = spec[f, diff_bins:-diff_bins] - spec[f - diff_frames, bin_start:bin_stop]
    # keep only positive values
    if pos:
        diff = diff * (diff > 0)
    return diff


class SpectralODF(object):
    """
    The SpectralODF class implements most of the common onset detection function
    based on the magnitude or phase information of a spectrogram.

    """
    def __init__(self, spectrogram, ratio=0.5, diff_frames=None):
        """
        Creates a new ODF object instance.

        :param spectrogram: the spectrogram object on which the detections functions operate
        :param ratio: calculate the difference to the frame which has the given magnitude ratio [default=0.5]
        :param diff_frames: calculate the difference to the N-th previous frame [default=None]
        :param diff_bins: calculate the maximum to N neighboring bins [default=None]

        """
        self.s = spectrogram
        # determine the number off diff frames
        if diff_frames is None:
            # get the first sample with a higher magnitude than given ratio
            sample = np.argmax(self.s.window > ratio)
            diff_samples = self.s.window.size / 2 - sample
            # convert to frames
            diff_frames = int(round(diff_samples / self.s.hop_size))
            # set the minimum to 1
            if diff_frames < 1:
                diff_frames = 1
        # sanity check
        if diff_frames < 1:
            raise ValueError("number of diff_frames must be >= 1")
        self.diff_frames = diff_frames

    # Onset Detection Functions
    def hfc(self):
        """
        High Frequency Content.

        "Computer Modeling of Sound for Transformation and Synthesis of Musical Signals"
        Paul Masri
        PhD thesis, University of Bristol, 1996

        """
        # HFC weights the magnitude spectrogram by the bin number, thus emphasising high frequencies
        return np.mean(self.s.spec * np.arange(self.s.bins), axis=1)

    def sd(self):
        """
        Spectral Diff.

        "A hybrid approach to musical note onset detection"
        Chris Duxbury, Mark Sandler and Matthew Davis
        Proceedings of the 5th International Conference on Digital Audio Effects (DAFx-02), 2002.

        """
        # Spectral diff is the sum of all squared positive 1st order differences
        return np.sum(self.diff(self.s.spec, pos=True) ** 2, axis=1)

    def sf(self):
        """
        Spectral Flux.

        "Computer Modeling of Sound for Transformation and Synthesis of Musical Signals"
        Paul Masri
        PhD thesis, University of Bristol, 1996

        """
        # Spectral flux is the sum of all positive 1st order differences
        return np.sum(self.diff(self.s.spec, pos=True), axis=1)

    def superflux(self):
        """
        SuperFlux with a maximum peak-tracking stage for difference calculation.

        "Maximum Filter Vibrato Suppression for Onset Detection"
        Sebastian Böck and Gerhard Widmer
        Proceedings of the 16th International Conference on Digital Audio Effects (DAFx-13), 2013.

        """
        # Spectral flux is the sum of all positive 1st order differences
        return np.sum(self.max_diff(self.s.spec, pos=True), axis=1)

    def mkl(self, epsilon=0.000001):
        """
        Modified Kullback-Leibler.

        :param epsilon: add epsilon to avoid division by 0 [default=0.000001]

        we use the implenmentation presented in:
        "Automatic Annotation of Musical Audio for Interactive Applications"
        Paul Brossier
        PhD thesis, Queen Mary University of London, 2006

        instead of the original work:
        "Onset Detection in Musical Audio Signals"
        Stephen Hainsworth and Malcolm Macleod
        Proceedings of the International Computer Music Conference (ICMC), 2003

        """
        if epsilon <= 0:
            raise ValueError("a positive value must be added before division")
        mkl = np.zeros_like(self.s.spec)
        mkl[1:] = self.s.spec[1:] / (self.s.spec[0:-1] + epsilon)
        # note: the original MKL uses sum instead of mean, but the range of mean is much more suitable
        return np.mean(np.log(1 + mkl), axis=1)

    def _pd(self):
        """
        Helper method used by pd() & wpd().

        """
        pd = np.zeros_like(self.s.phase)
        # instantaneous frequency is given by the first difference ψ′(n, k) = ψ(n, k) − ψ(n − 1, k)
        # change in instantaneous frequency is given by the second order difference ψ′′(n, k) = ψ′(n, k) − ψ′(n − 1, k)
        pd[2:] = self.s.phase[2:] - 2 * self.s.phase[1:-1] + self.s.phase[:-2]
        # map to the range -pi..pi
        return self.wraptopi(pd)

    def pd(self):
        """
        Phase Deviation.

        "On the use of phase and energy for musical onset detection in the complex domain"
        Juan Pablo Bello, Chris Duxbury, Matthew Davies and Mark Sandler
        IEEE Signal Processing Letters, Volume 11, Number 6, 2004

        """
        # take the mean of the absolute changes in instantaneous frequency
        return np.mean(np.abs(self._pd()), axis=1)

    def wpd(self):
        """
        Weighted Phase Deviation.

        "Onset Detection Revisited"
        Simon Dixon
        Proceedings of the 9th International Conference on Digital Audio Effects (DAFx), 2006

        """
        # make sure the spectrogram is not filtered before
        if np.shape(self.s.phase) != np.shape(self.s.spec):
            raise ValueError("Magnitude spectrogram and phase must be of same shape")
        # wpd = spec * pd
        return np.mean(np.abs(self._pd() * self.s.spec), axis=1)

    def nwpd(self, epsilon=0.000001):
        """
        Normalized Weighted Phase Deviation.

        :param epsilon: add epsilon to avoid division by 0 [default=0.000001]

        "Onset Detection Revisited"
        Simon Dixon
        Proceedings of the 9th International Conference on Digital Audio Effects (DAFx), 2006

        """
        if epsilon <= 0:
            raise ValueError("a positive value must be added before division")
        # normalize WPD by the sum of the spectrogram (add a small amount so that we don't divide by 0)
        return self.wpd() / np.add(np.mean(self.s.spec, axis=1), epsilon)

    def _cd(self):
        """
        Helper method used by cd() & rcd().

        we use the simple implementation presented in:
        "Onset Detection Revisited"
        Simon Dixon
        Proceedings of the 9th International Conference on Digital Audio Effects (DAFx), 2006

        """
        if np.shape(self.s.phase) != np.shape(self.s.spec):
            raise ValueError("Magnitude spectrogram and phase must be of same shape")
        # expected spectrogram
        cd_target = np.zeros_like(self.s.phase)
        # assume constant phase change
        cd_target[1:] = 2 * self.s.phase[1:] - self.s.phase[:-1]
        # add magnitude
        cd_target = self.s.spec * np.exp(1j * cd_target)
        # complex spectrogram
        # note: construct new instead of using self.stft, because pre-processing could have been applied
        cd = self.s.spec * np.exp(1j * self.s.phase)
        # subtract the target values
        cd[1:] -= cd_target[:-1]
        return cd

    def cd(self):
        """
        Complex Domain.

        "On the use of phase and energy for musical onset detection in the complex domain"
        Juan Pablo Bello, Chris Duxbury, Matthew Davies and Mark Sandler
        IEEE Signal Processing Letters, Volume 11, Number 6, 2004

        """
        # take the sum of the absolute changes
        return np.sum(np.abs(self._cd()), axis=1)

    def rcd(self):
        """
        Rectified Complex Domain.

        "Onset Detection Revisited"
        Simon Dixon
        Proceedings of the 9th International Conference on Digital Audio Effects (DAFx), 2006

        """
        # rectified complex domain
        rcd = self._cd()
        # only keep values where the magnitude rises
        rcd[1:] = rcd[1:] * (self.s.spec[1:] > self.s.spec[:-1])
        # take the sum of the absolute changes
        return np.sum(np.abs(rcd), axis=1)
