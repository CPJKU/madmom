#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all parser functionality used by other modules.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse
import multiprocessing as mp

# get the default values from the corresponding modules
from ..audio.signal import NORM, ATT, FPS, FRAME_SIZE
from ..audio.spectrogram import RATIO, DIFF_FRAMES, MUL, ADD
from ..audio.filters import FMIN, FMAX, BANDS_PER_OCTAVE, NORM_FILTERS


def audio(parser, online=None, norm=NORM, att=ATT, fps=FPS, window=FRAME_SIZE):
    """
    Add audio related arguments to an existing parser object.

    :param parser: existing argparse parser object
    :param online: online mode
    :param norm:   normalize the signal
    :param att:    attenuate the signal [dB]
    :param fps:    frames per second
    :param window: window / frame size
    :return:       audio argument parser group object

    """
    # if audio gets normalized, switch to offline mode
    if norm:
        online = False
    # add wav options to the existing parser
    g = parser.add_argument_group('audio arguments')
    if online is not None:
        g.add_argument('--online', dest='online', action='store_true',
                       default=online,
                       help='operate in online mode [default=%(default)s]')
    if norm is not None:
        g.add_argument('--norm', action='store_true', default=norm,
                       help='normalize the audio signal '
                            '(switches to offline mode)')
    if att is not None:
        g.add_argument('--att', action='store', type=float, default=att,
                       help='attenuate the audio signal [dB]')
    if fps is not None:
        g.add_argument('--fps', action='store', type=int, default=fps,
                       help='frames per second [default=%(default)i]')
    if window is not None:
        g.add_argument('--window', action='store', type=int, default=window,
                       help='frame length [samples, default=%(default)i]')
    # return the argument group so it can be modified if needed
    return g




def filtering(parser, default=None, fmin=FMIN, fmax=FMAX,
              bands=BANDS_PER_OCTAVE, norm_filters=NORM_FILTERS):
    """
    Add filter related arguments to an existing parser object.

    :param parser:       existing argparse parser object
    :param default:      set the default (adds a switch to negate)
    :param fmin:         the minimum frequency
    :param fmax:         the maximum frequency
    :param bands:        number of filter bands per octave
    :param norm_filters: normalize the area of the filter
    :return:             filtering argument parser group object

    """
    # add filter related options to the existing parser
    g = parser.add_argument_group('filtered magnitude spectrogram arguments')
    if default is False:
        g.add_argument('--filter', action='store_true', default=default,
                       help='filter the magnitude spectrogram with a '
                            'filterbank (apply values below)')
    elif default is True:
        g.add_argument('--no_filter', action='store_false', default=default,
                       dest='filter',
                       help='do not filter the magnitude spectrogram with a '
                            'filterbank (ignore values below)')
    if bands is not None:
        g.add_argument('--bands', action='store', type=int, default=bands,
                       help='filter bands per octave [default=%(default)i]')
    if fmin is not None:
        g.add_argument('--fmin', action='store', type=float, default=fmin,
                       help='minimum frequency of filter in Hz [default='
                            '%(default)i]')
    if fmax is not None:
        g.add_argument('--fmax', action='store', type=float, default=fmax,
                       help='maximum frequency of filter in Hz [default='
                            '%(default)i]')
    if norm_filters is False:
        # switch to turn it on
        g.add_argument('--norm_filters', action='store_true',
                       default=norm_filters,
                       help='normalize filters to have equal area')
    if norm_filters is True:
        g.add_argument('--no_norm_filters', dest='norm_filters',
                       action='store_false', default=norm_filters,
                       help='do not equalize filters to have equal area')
    # return the argument group so it can be modified if needed
    return g


# def log(parser, default=None, mul=MUL, add=ADD):
#     """
#     Add logarithmic magnitude related arguments to an existing parser object.
# 
#     :param parser:  existing argparse parser object
#     :param default: set the default (adds a switch to negate)
#     :param mul:     multiply the magnitude spectrogram with given value
#     :param add:     add the given value to the magnitude spectrogram
#     :return:        logarithmic argument parser group object
# 
#     """
#     # add log related options to the existing parser
#     g = parser.add_argument_group('logarithmic magnitude arguments')
#     if default is False:
#         g.add_argument('--log', action='store_true', default=default,
#                        help='logarithmic magnitude [default=linear]')
#     elif default is True:
#         g.add_argument('--no_log', dest='log', action='store_false',
#                        default=default,
#                        help='no logarithmic magnitude [default=logarithmic]')
#     if mul is not None:
#         g.add_argument('--mul', action='store', type=float, default=mul,
#                        help='multiplier (before taking the log) '
#                             '[default=%(default)i]')
#     if add is not None:
#         g.add_argument('--add', action='store', type=float, default=add,
#                        help='value added (before taking the log) '
#                             '[default=%(default)i]')
#     # return the argument group so it can be modified if needed
#     return g




def save_load(parser):
    """
    Add options to save/load activations to an existing parser object.

    :param parser: existing argparse parser object
    :return:       input/output argument parser group object

    """
    # add onset detection related options to the existing parser
    g = parser.add_argument_group('save/load the activations')
    # add options for saving and loading the activations
    g.add_argument('-s', dest='save', action='store_true', default=False,
                   help='save the activations to file')
    g.add_argument('-l', dest='load', action='store_true', default=False,
                   help='load the activations from file')
    g.add_argument('--sep', action='store', default=None,
                   help='separator for saving/loading the activation '
                        'function [default: None, i.e. numpy binary format]')
    # return the argument group so it can be modified if needed
    return g


def midi(parser, length=None, velocity=None):
    """
    Add MIDI related arguments to an existing parser object.

    :param parser:   existing argparse parser object
    :param length:   default length of the notes
    :param velocity: default velocity of the notes
    :return:         MIDI argument parser group object

    """
    # add MIDI related options to the existing parser
    g = parser.add_argument_group('MIDI arguments')
    g.add_argument('--midi', action='store_true', help='save as MIDI')
    if length is not None:
        g.add_argument('--note_length', action='store', type=float,
                       default=length,
                       help='set the note length [default=%(default).2f]')
    if velocity is not None:
        g.add_argument('--note_velocity', action='store', type=int,
                       default=velocity,
                       help='set the note velocity [default=%(default)i]')
    # return the argument group so it can be modified if needed
    return g


def io(parser):
    """
    Add input / output related arguments to an existing parser object.

    :param parser: existing argparse parser object

    """
    import sys
    # general options
    parser.add_argument('input', type=argparse.FileType('r'),
                        help='input file (.wav or saved activation function)')
    parser.add_argument('output', nargs='?', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help='output file [default: STDOUT]')
    parser.add_argument('-v', dest='verbose', action='count',
                        help='increase verbosity level')
