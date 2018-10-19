# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains all harmonic/percussive source separation functionality.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from madmom.processors import Processor

# TODO: keep this as Processors or should it be done as np.ndarray classes?


class HarmonicPercussiveSourceSeparation(Processor):
    """
    HarmonicPercussiveSourceSeparation is a Processor which separates the
    magnitude spectrogram into its harmonic and percussive components with
    median filters.

    Parameters
    ----------
    masking : float or str
        Can be either the literal 'binary' or any float coefficient resulting
        in a soft mask. 'None' translates to a binary mask, too.
    harmonic_filter : tuple of ints
        Tuple with harmonic filter size (frames, bins).
    percussive_filter : tuple of ints
        Tuple with percussive filter size (frames, bins).

    References
    ----------
    .. [1] Derry FitzGerald,
           "Harmonic/percussive separation using median filtering.",
           Proceedings of the 13th International Conference on Digital Audio
           Effects (DAFx), Graz, Austria, 2010.

    """
    MASKING = 'binary'
    HARMONIC_FILTER = (15, 1)
    PERCUSSIVE_FILTER = (1, 15)

    def __init__(self, masking=MASKING, harmonic_filter=HARMONIC_FILTER,
                 percussive_filter=PERCUSSIVE_FILTER):
        # set the parameters, so they get used for computation
        self.masking = masking
        self.harmonic_filter = np.asarray(harmonic_filter, dtype=int)
        self.percussive_filter = np.asarray(percussive_filter, dtype=int)

    def slices(self, data):
        """
        Returns the harmonic and percussive slices of the data.

        Parameters
        ----------
        data : numpy array
            Data to be sliced (usually a magnitude spectrogram).

        Returns
        -------
        harmonic_slice : numpy array
            Harmonic slice.
        percussive_slice : numpy array
            Percussive slice.

        """
        from scipy.ndimage.filters import median_filter
        # compute the harmonic and percussive slices
        harmonic_slice = median_filter(data, self.harmonic_filter)
        percussive_slice = median_filter(data, self.percussive_filter)
        # return the slices
        return harmonic_slice, percussive_slice

    def masks(self, harmonic_slice, percussive_slice):
        """
        Returns the masks given the harmonic and percussive slices.

        Parameters
        ----------
        harmonic_slice : numpy array
            Harmonic slice.
        percussive_slice : numpy array
            Percussive slice.

        Returns
        -------
        harmonic_mask : numpy array
            Harmonic mask.
        percussive_mask : numpy array
            Percussive mask.

        """
        # compute the masks
        if self.masking in (None, 'binary'):
            # return binary masks
            harmonic_mask = harmonic_slice > percussive_slice
            percussive_mask = percussive_slice >= harmonic_slice
        else:
            # return soft masks
            p = float(self.masking)
            harmonic_slice_ = harmonic_slice ** p
            percussive_slice_ = percussive_slice ** p
            slice_sum_ = harmonic_slice_ + percussive_slice_
            harmonic_mask = harmonic_slice_ / slice_sum_
            percussive_mask = percussive_slice_ / slice_sum_
        # return the masks
        return harmonic_mask, percussive_mask

    def process(self, data):
        """
        Returns the harmonic and percussive components of the given data.

        Parameters
        ----------
        data : numpy array
            Data to be split into harmonic and percussive components.

        Returns
        -------
        harmonic components : numpy array
            Harmonic components.
        percussive components : numpy array
            Percussive components.

        """
        from .spectrogram import Spectrogram
        # data must be in the right format
        if isinstance(data, Spectrogram):
            # use the magnitude spectrogram of the Spectrogram
            spectrogram = data.spec
        # compute the harmonic and percussive slices
        slices = self.slices(spectrogram)
        # compute the corresponding masks
        harmonic_mask, percussive_mask = self.masks(*slices)
        # filter the data
        harmonic = spectrogram * harmonic_mask
        percussive = spectrogram * percussive_mask
        # and return it
        return harmonic, percussive

    @staticmethod
    def add_arguments(parser, masking=None, harmonic_filter=None,
                      percussive_filter=None):
        """
        Add harmonic/percussive source separation related arguments to an
        existing parser object.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        masking : float, optional
            Masking; if 'None', binary masking is used.
        harmonic_filter : tuple, optional
            Harmonic filter (frames, bins).
        percussive_filter : tuple, optional
            Percussive filter (frames, bins).

        Returns
        -------
        argparse argument group
            Harmonic/percussive source separation argument parser group.

        Notes
        -----
        Parameters are included in the group only if they are not 'None'.

        """
        # add harmonic/percussive related options to the existing parser
        g = parser.add_argument_group('harmonic/percussive source separation '
                                      'related arguments')
        if masking is not None:
            g.add_argument('--filter_type', action='store', type=float,
                           default=masking,
                           help='masking coefficient [default=%(default).2f]')
        if harmonic_filter is not None:
            g.add_argument('--harmonic_filter', action='store',
                           default=harmonic_filter,
                           help='harmonic filter size (frames, bins) '
                                '[default=%(default)s]')
        if percussive_filter is not None:
            g.add_argument('--percussive_filter', action='store',
                           default=percussive_filter,
                           help='percussive filter size (frames, bins) '
                                '[default=%(default)s]')
        # return the argument group so it can be modified if needed
        return g


# alias
HPSS = HarmonicPercussiveSourceSeparation
