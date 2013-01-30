"""
This module contains functions computing simple low level audio features
"""

__docformat__ = "restructuredtext en"

import numpy as np
import sys


def root_mean_square(signal):
    """
    Computes the root mean square of the signal. This can be used as
    a measurement of power.

    :Parameters:
      - `signal`: Input audio signal

    :Returns:
        Root mean square of the passed signal
    """
    return np.sqrt(np.dot(signal, signal) / len(signal))


def sound_pressure_level(signal, p_ref=1.0):
    """
    Computes the sound pressure level of a signal.

    From en.wikipedia.org/wiki/Sound_pressure:
    Sound pressure level (SPL) or sound level is a logarithmic measure of the
    effective sound pressure of a sound relative to a reference value.
    It is measured in decibels (dB) above a standard reference level.

    :Parameters:
      - `signal`: Audio data (np.array, list, ...)
      - `p_ref`: Reference level. If microphone is not calibrated, use 1.0

    :Returns:
        Sound pressure level of the passed audio signal in dB
    """

    rms = root_mean_square(signal)

    if rms > 0.0:
        spl = 20.0 * np.log10(rms / p_ref)

    else:
        spl = -sys.float_info.max

    return spl
