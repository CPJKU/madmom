import numpy as np
from online import LocallyNormalisedOnsetFeature


def locally_normalised_onset_feature(spectrogram, norm_window_length,
                                     log_factor=5000, log_shift=1):

    # TODO: reimplement this function to make it faster

    lnof = LocallyNormalisedOnsetFeature(spectrogram.shape[1], norm_window_length,
                                         log_factor, log_shift)

    results = np.empty(spectrogram.shape)

    for i, spec in enumerate(spectrogram):
        results[i] = lnof.compute(spec)

    return results


def simple_onset_binwise(spectrogram, log_factor=20.0, log_shift=1.0, normalise=True):
    assert spectrogram.ndim == 2
    assert spectrogram.shape[0] > 1

    res = np.diff(np.log(spectrogram * log_factor + log_shift), axis=0)

    if normalise:
        # after normalisation, the values of the difference are between -1 and 1
        max_val = np.log(log_factor / log_shift + 1)  # log(fact+shift) - log(shift)
        res /= max_val

    return res


def simple_onset(spectrogram, log_factor=20.0, log_shift=1.0, only_positive=False, normalise=True):
    onset_func = simple_onset_binwise(spectrogram, log_factor, log_shift, normalise)

    if only_positive:
        onset_func = np.maximum(0.0, onset_func)

    onset_func = onset_func.sum(1)

    if normalise:
        # after normalisation, the values are between -1 and 1
        onset_func /= spectrogram.shape[1]

    return onset_func
