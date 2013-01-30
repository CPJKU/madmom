import numpy as np


class LinearSpectrumTransformer(object):

    """
    Base class for linear spectrum transformations. A linear spectrum
    transformation transforms a spectrum with N bins into another representation
    with M bins using a linear combination of the N original bins (e.g.
    converts the spectrum into a chroma vector).

    To create a concrete transform you need to derive from this class
    and set self.bin_assignments somewhere (e.g. in the constructor), which
    needs to be a numpy array of shape (M, N)

    :Parameters:
        - `normalise`: Sets whether to normalise the spectrum after the
                       transformation, so that it sums up to 1
    """

    def __init__(self, normalise=False):
        self.normalise = normalise
        self.bin_assignments = 1

    def compute(self, spectrum):
        """
        Computes the transformation of the spectrum.

        :Parameters:
            - `spectrum`: input spectrum

        :Returns:
            Transformed spectrum
        """
        transformed_spectrum = np.dot(self.bin_assignments, spectrum.T).T

        if self.normalise:
            if transformed_spectrum.ndim == 1:
                transformed_spectrum /= transformed_spectrum.sum()
            else:
                transformed_spectrum /= transformed_spectrum.sum(axis=1)[:, np.newaxis]

        return transformed_spectrum


class SemitoneSpectrum(LinearSpectrumTransformer):
    """
    Transforms the spectrum into a semitone scale.
    """

    def __init__(self, fft_length, sample_rate, q_factor=25.0, normalise=False):
        """
        Initialises the semitone transformation

        :Parameters:
          - `fft_length`: Specifies the FFT length used to obtain the magnitude
                          spectrum
          - `sample_rate`: Sample rate of the audio
          - `q_factor`: Defines the width of the rectangle filters used to 
                        transform the spectrum
          - `normalise`: Specifies if the semitone vectors shall be normalised,
                         i.e. divided by their sum
        """
        from .. import freq_scales as fs
        super(SemitoneSpectrum, self).__init__(normalise)

        mag_spec_length = fft_length / 2 + 1
        max_note = np.floor(fs.hz_to_midi(sample_rate / 2))
        note_freqs = fs.midi_to_hz(np.arange(0, max_note))
        min_note_freqs = note_freqs - note_freqs / q_factor
        max_note_freqs = note_freqs + note_freqs / q_factor
        fft_freqs = abs(np.fft.fftfreq(fft_length) * sample_rate)[:mag_spec_length]

        note_dist = abs(note_freqs[:, np.newaxis] - fft_freqs)
        min_note_dist = abs(min_note_freqs[:, np.newaxis] - fft_freqs)
        max_note_dist = abs(max_note_freqs[:, np.newaxis] - fft_freqs)

        direct_assignments = note_dist == note_dist.min(axis=1)[:, np.newaxis]
        range_assignments = (note_dist < min_note_dist) & (note_dist < max_note_dist)

        self.bin_assignments = (direct_assignments | range_assignments).astype(float)


class SimpleChromaComputer(LinearSpectrumTransformer):
    """
    A simple chroma computer. Each frequency bin of a magnitude spectrum
    is assigned a chroma class, and all it's contents are added to this class.
    No diffusion, just discrete assignment.
    """

    def __init__(self, fft_length, sample_rate, normalise=False):
        from .. import freq_scales as fs
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
