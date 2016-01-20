# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains chroma related functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from madmom.audio.spectrogram import Spectrogram, FilteredSpectrogram
from madmom.audio.filters import (A4, Filterbank,
                                  PitchClassProfileFilterbank as PCP,
                                  HarmonicPitchClassProfileFilterbank as HPCP)


def pcp_chord_transcription(pcp):
    """
    Perform a simple chord transcription based on the given pitch class
    profile (PCP).

    Parameters
    ----------
    pcp : numpy array
        Pitch class profile.

    Returns
    -------
    chords : numpy array
        Chord transcription.

    References
    ----------
    .. [1] T. Fujishima,
           "Realtime chord recognition of musical sound: a system using Common
           Lisp Music",
           Proceedings of the International Computer Music Conference (ICMC),
           1999.

    """
    raise NotImplementedError


# inherit from FilteredSpectrogram, since this class is closest related
class PitchClassProfile(FilteredSpectrogram):
    """
    Simple class for extracting pitch class profiles (PCP), i.e. chroma
    vectors from a spectrogram.

    Parameters
    ----------
    spectrogram : :class:`.audio.spectrogram.Spectrogram` instance
        :class:`.audio.spectrogram.Spectrogram` instance.
    filterbank : :class:`.audio.filters.Filterbank` class or instance
        :class:`.audio.filters.Filterbank` class or instance.
    num_classes : int, optional
        Number of pitch classes.
    fmin : float, optional
        Minimum frequency of the PCP filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the PCP filterbank [Hz].
    fref : float, optional
        Reference frequency for the first PCP bin [Hz].
    kwargs : dict, optional
        If no :class:`.audio.spectrogram.Spectrogram` instance was given,
        one is instantiated with these additional keyword arguments.

    Notes
    -----
    If `fref` is 'None', the reference frequency is estimated from the given
    spectrogram.

    References
    ----------
    .. [1] T. Fujishima,
           "Realtime chord recognition of musical sound: a system using Common
           Lisp Music",
           Proceedings of the International Computer Music Conference (ICMC),
           1999.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, spectrogram, filterbank=PCP, num_classes=PCP.CLASSES,
                 fmin=PCP.FMIN, fmax=PCP.FMAX, fref=A4, **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, spectrogram, filterbank=PCP, num_classes=PCP.CLASSES,
                fmin=PCP.FMIN, fmax=PCP.FMAX, fref=A4, **kwargs):
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
            raise ValueError('not a Filterbank type or instance: {0!s}'.format(
                             filterbank))
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
        # return the object
        return obj


class HarmonicPitchClassProfile(PitchClassProfile):
    """
    Class for extracting harmonic pitch class profiles (HPCP) from a
    spectrogram.

    Parameters
    ----------
    spectrogram : :class:`.audio.spectrogram.Spectrogram` instance
        :class:`.audio.spectrogram.Spectrogram` instance.
    filterbank : :class:`.audio.filters.Filterbank` class or instance
        Filterbank class or instance.
    num_classes : int, optional
        Number of harmonic pitch classes.
    fmin : float, optional
        Minimum frequency of the HPCP filterbank [Hz].
    fmax : float, optional
        Maximum frequency of the HPCP filterbank [Hz].
    fref : float, optional
        Reference frequency for the first HPCP bin [Hz].
    window : int, optional
        Length of the weighting window [bins].
    kwargs : dict, optional
        If no :class:`.audio.spectrogram.Spectrogram` instance was given,
        one is instantiated with these additional keyword arguments.

    Notes
    -----
    If `fref` is 'None', the reference frequency is estimated from the given
    spectrogram.

    References
    ----------
    .. [1] Emilia Gómez,
           "Tonal Description of Music Audio Signals",
           PhD thesis, Universitat Pompeu Fabra, Barcelona, Spain, 2006.

    """
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, spectrogram, filterbank=HPCP, num_classes=HPCP.CLASSES,
                 fmin=HPCP.FMIN, fmax=HPCP.FMAX, fref=A4, window=HPCP.WINDOW,
                 **kwargs):
        # this method is for documentation purposes only
        pass

    def __new__(cls, spectrogram, filterbank=HPCP, num_classes=HPCP.CLASSES,
                fmin=HPCP.FMIN, fmax=HPCP.FMAX, fref=A4, window=HPCP.WINDOW,
                **kwargs):
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
                                    fmax=fmax, fref=fref, window=window)
        if not isinstance(filterbank, Filterbank):
            raise ValueError('not a Filterbank type or instance: {0!s}'.format(
                             filterbank))
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
        # return the object
        return obj
