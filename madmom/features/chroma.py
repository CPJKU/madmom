#!/usr/bin/env python
# encoding: utf-8
"""
This file contains chroma related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np

from madmom.audio.spectrogram import Spectrogram
from madmom.audio.filters import (Filterbank,
                                  PitchClassProfileFilterbank as PCP,
                                  HarmonicPitchClassProfileFilterbank as HPCP)


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


# we inherit from Spectrogram, since it is the class which is closest related
# and offers a lot o
class PitchClassProfile(Spectrogram):
    """
    Simple class for extracting pitch class profiles (PCP), i.e. chroma
    vectors from a spectrogram.

    "Realtime chord recognition of musical sound: a system using Common Lisp
     Music"
    T. Fujishima
    Proceedings of the International Computer Music Conference (ICMC 1999),
    Beijing, China

    """

    def __new__(cls, spectrogram, filterbank=PCP, num_classes=PCP.CLASSES,
                fmin=PCP.FMIN, fmax=PCP.FMAX, fref=None, **kwargs):
        """
        Creates a new PitchClassProfile instance.

        :param spectrogram: a spectrogram to operate on [Spectrogram]
        :param filterbank:  Filterbank instance or type [Filterbank]
        :param num_classes: number of harmonic pitch classes [int]
        :param fmin:        the minimum frequency [Hz, float]
        :param fmax:        the maximum frequency [Hz, float]
        :param fref:        reference frequency of the first PCP bin
                            [Hz, float]

        Note: If fref is 'None', the reference frequency is estimated from the
              given spectrogram.

        """
        # check spectrogram type
        if not isinstance(spectrogram, Spectrogram):
            spectrogram = Spectrogram(spectrogram, **kwargs)
        # the spectrogram must not be filtered
        if spectrogram.filterbank is not None:
            import warnings
            warnings.warn('Spectrogram should not be filtered.')
        # reference frequency for the filterbank
        if fref is None:
            fref = spectrogram.tuning_frequency()

        # set filterbank
        if issubclass(filterbank, Filterbank):
            filterbank = filterbank(spectrogram.bin_frequencies,
                                    num_classes=num_classes, fmin=fmin,
                                    fmax=fmax, fref=fref)
        if not isinstance(filterbank, Filterbank):
            raise ValueError('not a Filterbank type or instance: %s' %
                             filterbank)
        # filter the spectrogram
        data = np.dot(spectrogram, filterbank)
        # cast as PitchClassProfile
        obj = np.asarray(data).view(cls)
        # save additional attributes
        obj.filterbank = filterbank
        obj.spectrogram = spectrogram
        # and those from the given spectrogram
        obj.stft = spectrogram.stft
        obj.frames = spectrogram.stft.frames
        obj.mul = spectrogram.mul
        obj.add = spectrogram.add
        # return the object
        return obj


class HarmonicPitchClassProfile(PitchClassProfile):
    """
    Class for extracting harmonic pitch class profiles (HPCP) from a
    spectrogram.

    "Tonal Description of Music Audio Signals"
    E. Gómez
    PhD thesis, Universitat Pompeu Fabra, Barcelona, Spain

    """

    def __init__(self, spectrogram, num_classes=HPCP.CLASSES, fmin=HPCP.FMIN,
                 fmax=HPCP.FMAX, fref=None, window=HPCP.WINDOW,
                 filterbank=None, **kwargs):
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
            spectrogram, num_classes, fmin, fmax, fref, filterbank, **kwargs)
        # set hidden parameters for filterbank creation
        self._filterbank_type = HPCP
        self._parameters['window'] = window

    @property
    def hpcp(self):
        """Harmonic Pitch Class Profile."""
        # just map the HPCP to the PCP
        return self.pcp
