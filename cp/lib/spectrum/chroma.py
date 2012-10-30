"""
This module contains implementations of chroma vector extraction algorithms
"""

import numpy as np
from .. import freq_scales as fs
from linear_spectrum_transformer import LinearSpectrumTransformer


class SimpleChromaComputer(LinearSpectrumTransformer):

    """
    A simple chroma computer. Each frequency bin of a magnitude spectrum
    is assigned a chroma class, and all it's contents are added to this class.
    No diffusion, just discrete assignment.
    """

    def __init__(self, fft_length, sample_rate, normalise=False):
        """
        Initialises the computation.

        :Parameters:
          - `fft_length`: Specifies the FFT length used to obtain the magnitude
                          spectrum
          - `sample_rate`: Sample rate of the audio
          - `normalise`: Specifies if the chroma vectors shall be normalised,
                         i.e. divided by it's sum
        """
        super(SimpleChromaComputer, self).__init__(normalise)

        mag_spec_length = fft_length / 2 + 1
        max_note = np.floor(fs.hz_to_midi(sample_rate / 2))
        note_freqs = fs.midi_to_hz(np.arange(0, max_note))
        fft_freqs = abs(np.fft.fftfreq(fft_length) * sample_rate)[:mag_spec_length]

        note_to_fft_distances = abs(note_freqs[:, np.newaxis] - fft_freqs)
        note_assignments = np.argmin(note_to_fft_distances, axis=0) % 12

        self.bin_assignments = np.mgrid[:12, :mag_spec_length][0] == note_assignments
