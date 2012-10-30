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
