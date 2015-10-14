# encoding: utf-8
"""
This file contains all harmonic/percussive source separation functionality.

"""

import numpy as np

from madmom.processors import Processor

# TODO: keep this as Processors or should it be done as np.ndarray classes?


class HarmonicPercussiveSourceSeparation(Processor):
    """
    HarmonicPercussiveSourceSeparation is a Processor which separates the
    magnitude spectrogram into its harmonic and percussive components with
    median filters.

    "Harmonic/percussive separation using median filtering."
    Derry FitzGerald.
    Proceedings of the 13th International Conference on Digital Audio Effects
    (DAFx-10), Graz, Austria, September 2010.

    """
    MASKING = 'binary'
    HARMONIC_FILTER = (15, 1)
    PERCUSSIVE_FILTER = (1, 15)

    def __init__(self, masking=MASKING, harmonic_filter=HARMONIC_FILTER,
                 percussive_filter=PERCUSSIVE_FILTER):
        """
        Creates a new HarmonicPercussiveSourceSeparation instance.

        The magnitude spectrogram are separated with median filters with the
        given sizes into their harmonic and percussive parts.

        :param masking:           masking (see below)
        :param harmonic_filter:   tuple with harmonic filter size
                                  (frames, bins)
        :param percussive_filter: tuple with percussive filter size
                                  (frames, bins)

        Note: `masking` can be either the literal 'binary' or any float
              coefficient resulting in a soft mask. `None` translates to a
              binary mask, too.

        """
        # set the parameters, so they get used for computation
        self.masking = masking
        self.harmonic_filter = np.asarray(harmonic_filter, dtype=int)
        self.percussive_filter = np.asarray(percussive_filter, dtype=int)

    def slices(self, data):
        """
        Compute the harmonic and percussive slices of the data.

        :param data: magnitude spectrogram [numpy array]
        :return:     tuple (harmonic slice, percussive slice)

        """
        from scipy.ndimage.filters import median_filter
        # compute the harmonic and percussive slices
        harmonic_slice = median_filter(data, self.harmonic_filter)
        percussive_slice = median_filter(data, self.percussive_filter)
        # return the slices
        return harmonic_slice, percussive_slice

    def masks(self, harmonic_slice, percussive_slice):
        """
        Compute the masks given the harmonic and percussive slices.

        :param harmonic_slice:   harmonic slice
        :param percussive_slice: percussive slice
        :return:                 tuple (harmonic mask, percussive mask)

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

    def process(self, spectrogram):
        """
        Compute the harmonic and percussive components of the given data.

        :param spectrogram: Spectrogram instance or numpy array with the
                            magnitude spectrogram
        :return:            tuple (harmonic components, percussive components)

        """
        from .spectrogram import Spectrogram
        # data must be in the right format
        if isinstance(spectrogram, Spectrogram):
            # use the magnitude spectrogram of the Spectrogram
            spectrogram = spectrogram.spec
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

        :param parser:            existing argparse parser object
        :param masking:           masking [float]
                                  (if not set, binary masking is used)
        :param harmonic_filter:   harmonic filter [tuple (frames, bins)]
        :param percussive_filter: percussive filter [tuple (frames, bins)]
        :return:                  harmonic/percussive source separation
                                  argument parser group

        Parameters are included in the group only if they are not 'None'.

        """
        # TODO: split this among the individual classes
        # add filter related options to the existing parser
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
