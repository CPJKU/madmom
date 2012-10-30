import numpy as np


class LocallyNormalisedOnsetFeature:

    def __init__(self, spectrum_size, norm_window_size=100,
                 log_factor=5000, log_shift=1):

        self.log_factor = 5000
        self.log_shift = 1
        self.maxima = [0.0] * norm_window_size
        self.prev_spectrum = np.zeros(spectrum_size)
        self.diff = np.empty(spectrum_size)  # preallocate space

    def compute(self, spectrum):

        assert spectrum.ndim == 1, "Only single spectra allowed"

        np.maximum(0.0, spectrum - self.prev_spectrum, out=self.diff)
        self.diff *= self.log_factor
        self.diff += self.log_shift
        np.log(self.diff, out=self.diff)

        self.prev_spectrum = spectrum

        del self.maxima[0]
        self.maxima.append(self.diff.max())
        norm_factor = max(self.maxima)

        self.diff /= norm_factor
        return self.diff
