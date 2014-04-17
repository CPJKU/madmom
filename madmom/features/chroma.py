#!/usr/bin/env python
# encoding: utf-8
"""
This file contains chroma related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np

from ..audio.spectrogram import Spectrogram
from ..audio.filters import (PitchClassProfileFilterbank,
                             HarmonicPitchClassProfileFilterbank, FMIN, FMAX,
                             PCP_CLASSES, HPCP_CLASSES, HPCP_FMIN, HPCP_FMAX,
                             HPCP_WINDOW)


def pcp_chord_transcription(pcp):
    """
    Perform a simple chord transcription based on the given pitch class
    profile (PCP)

    :param pcp: pitch class profile
    :return:    a chord transcription

    "Realtime chord recognition of musical sound: a system using Common Lisp
     Music"
    T. Fujishima
    Proceedings of the International Computer Music Conference (ICMC 1999),
    Beijing, China

    """
    raise NotImplementedError


class PitchClassProfile(object):
    """
    Simple class for extracting pitch class profiles (PCP), i.e. chroma
    vectors from a spectrogram.

    "Realtime chord recognition of musical sound: a system using Common Lisp
     Music"
    T. Fujishima
    Proceedings of the International Computer Music Conference (ICMC 1999),
    Beijing, China

    """
    def __init__(self, spectrogram, num_classes=PCP_CLASSES, fmin=FMIN,
                 fmax=FMAX, fref=None, filterbank=None, *args, **kwargs):
        """
        Creates a new PitchClassProfile instance.

        :param spectrogram: a spectrogram to operate on
        :param num_classes: number of harmonic pitch classes
        :param fmin:        the minimum frequency [Hz]
        :param fmax:        the maximum frequency [Hz]
        :param fref:        reference frequency for the first PCP bin [Hz]
        :param filterbank:  use this chroma filterbank instead of creating one

        Note: If fref is 'None', the reference frequency is estimated on the
              given spectrogram.

        """
        # check spectrogram type
        if isinstance(spectrogram, Spectrogram):
            # already the right format
            self._spectrogram = spectrogram
        else:
            # assume a file name, try to instantiate a Spectrogram object
            # Note: since the filterbank is a proper argument, the created
            #       Spectrogram will always be unfiltered (what we want)
            self._spectrogram = Spectrogram(spectrogram, *args, **kwargs)

        # the spectrogram must not be filtered
        if self._spectrogram.filterbank:
            raise ValueError('Spectrogram must not be filtered.')

        # reference frequency for the filterbank
        if fref is None:
            fref = self.spectrogram.tuning_frequency

        # set filterbank
        self._filterbank = filterbank
        # some hidden parameters for filterbank creation
        self._filterbank_type = PitchClassProfileFilterbank
        self._parameters = {'num_classes': num_classes,
                            'fmin': fmin,
                            'fmax': fmax,
                            'fref': fref}
        # hidden attributes
        self._pcp = None

    @property
    def spectrogram(self):
        """Spectrogram."""
        return self._spectrogram

    @property
    def filterbank(self):
        """Filterbank."""
        # create a filterbank if needed
        if self._filterbank is None:
            self._filterbank = self._filterbank_type(
                self.spectrogram.num_fft_bins,
                self.spectrogram.frames.signal.sample_rate, **self._parameters)
        return self._filterbank

    @property
    def pcp(self):
        """Pitch Class Profile."""
        if self._pcp is None:
            # map the spectrogram to pitch classes
            self._pcp = np.dot(self.spectrogram.spec, self.filterbank)
        return self._pcp


class HarmonicPitchClassProfile(PitchClassProfile):
    """
    Class for extracting harmonic pitch class profiles (HPCP) from a
    spectrogram.

    "Tonal Description of Music Audio Signals"
    E. Gómez
    PhD thesis, Universitat Pompeu Fabra, Barcelona, Spain

    """
    def __init__(self, spectrogram, num_classes=HPCP_CLASSES, fmin=HPCP_FMIN,
                 fmax=HPCP_FMAX, fref=None, window=HPCP_WINDOW,
                 filterbank=None, *args, **kwargs):
        """
        Creates a new HarmonicPitchClassProfile instance.

        :param spectrogram: a spectrogram to operate on
        :param num_classes: number of harmonic pitch classes
        :param fmin:        the minimum frequency [Hz]
        :param fmax:        the maximum frequency [Hz]
        :param fref:        reference frequency for the first HPCP bin [Hz]
        :param window:      length of the weighting window [bins]

        Note: If fref is 'None', the reference frequency is estimated on the
              given spectrogram.

        """
        # pass all arguments (but the window) the the PitchClassProfile class
        super(HarmonicPitchClassProfile, self).__init__(
            spectrogram, num_classes, fmin, fmax, fref, filterbank, *args, **kwargs)
        # set hidden parameters for filterbank creation
        self._filterbank_type = HarmonicPitchClassProfileFilterbank
        self._parameters['window'] = window

    @property
    def hpcp(self):
        """Harmonic Pitch Class Profile."""
        # just map the HPCP to the PCP
        return self.pcp
