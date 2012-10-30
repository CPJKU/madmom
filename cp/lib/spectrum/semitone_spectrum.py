import numpy as np
from linear_spectrum_transformer import LinearSpectrumTransformer
from .. import freq_scales as fs


class SemitoneSpectrum(LinearSpectrumTransformer):

    def __init__(self, fft_length, sample_rate, q_factor=25.0, normalise=False):

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
