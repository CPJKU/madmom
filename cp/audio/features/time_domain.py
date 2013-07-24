# encoding: utf-8
"""
Copyright (c) 2013 Sebastian Böck <sebastian.boeck@jku.at>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
This module contains functions computing simple low level time-domain audio features
"""

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
