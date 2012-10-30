"""
This module contains some simple transformations between scales used to
describe the periodicity of a tone.
"""

__docformat__ = "restructuredtext en"

import numpy as np


def hz_to_erb(f):
    """
    Convert Hz to the ERB scale

    Information about the ERB scale can be found at
    https://ccrma.stanford.edu/~jos/bbt/Equivalent_Rectangular_Bandwidth.html

    :Parameters:
      - `f`: Input frequencies in Hz

    :Returns:
        ERB-scaled frequencies

    """
    return 21.4 * np.log10(1 + 4.37 * f / 1000)


def erb_to_hz(e):
    """
    Convert ERB scaled frequencies to Hz

    Information about the ERB scale can be found at
    https://ccrma.stanford.edu/~jos/bbt/Equivalent_Rectangular_Bandwidth.html

    :Parameters:
      - `e`: ERB scaled input frequencies

    :Returns:
        Frequencies in Hz
    """
    return (10 ** (e / 21.4) - 1) * 1000 / 4.37


def midi_to_hz(m, a4=440.0):
    """
    Returns the frequencies of the corresponding MIDI note ids

    For details take a look at http://www.phys.unsw.edu.au/jw/notes.html

    :Parameters:
      - `m`: MIDI note ids
      - `a4`: Frequency of the concert pitch

    :Returns:
        Frequencies corresponding the the MIDI note ids
    """
    return 2 ** ((m - 69.0) / 12) * a4


def hz_to_midi(f, a4=440.0):
    """
    Returns the MIDI note ids corresponding to frequencies

    For details take a look at http://www.phys.unsw.edu.au/jw/notes.html

    Note that if this function does not necessarily return a valid
    MIDI note id, you may need to round it to the nearest integer.

    :Parameters:
      - `f`: Input frequencies
      - `a4`: Frequency of the concert pitch

    :Returns:
        MIDI note ids corresponding to the frequencies
    """
    return (12 * np.log2(f / a4)) + 69
