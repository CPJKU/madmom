#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all parser functionality used by other modules.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse
import multiprocessing

# get the default values from the corresponding modules
from ..audio.signal import NORM, ATT, FPS, FRAME_SIZE
from ..audio.spectrogram import RATIO, DIFF_FRAMES, MUL, ADD
from ..audio.filterbank import FMIN, FMAX, BANDS_PER_OCTAVE, NORM_FILTERS
from ..features.onsets import (THRESHOLD, SMOOTH, COMBINE, DELAY, MAX_BINS,
                               PRE_AVG, POST_AVG, PRE_MAX, POST_MAX)
from ..features.beats import THRESHOLD as BT, SMOOTH as BS, MIN_BPM, MAX_BPM


def audio(parser, online=None, norm=NORM, att=ATT, fps=FPS, window=FRAME_SIZE):
    """
    Add audio related arguments to an existing parser object.

    :param parser: existing argparse parser object
    :param online: online mode
    :param norm:   normalize the signal
    :param att:    attenuate the signal [dB]
    :param fps:    frames per second
    :param window: window / frame size
    :return:       the modified parser object

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


def spec(parser, ratio=RATIO, diff_frames=DIFF_FRAMES):
    """
    Add spectrogram related arguments to an existing parser object.

    :param parser:      existing argparse parser object
    :param ratio:       calculate the difference to the frame which window
                        overlaps to this ratio
    :param diff_frames: calculate the difference to the N-th previous frame
    :return:            the modified parser object

    """
    # add spec related options to the existing parser
    # spectrogram options
    g = parser.add_argument_group('spectrogram arguments')
    g.add_argument('--ratio', action='store', type=float, default=ratio,
                   help='window magnitude ratio to calc number of diff '
                        'frames [default=%(default).1f]')
    g.add_argument('--diff_frames', action='store', type=int,
                   default=diff_frames,
                   help='number of diff frames [default=%(default)s]')
    # return the argument group so it can be modified if needed
    return g


def filtering(parser, filtering=None, fmin=FMIN, fmax=FMAX,
              bands=BANDS_PER_OCTAVE, norm_filters=NORM_FILTERS):
    """
    Add filter related arguments to an existing parser object.

    :param parser:       existing argparse parser object
    :param filtering:    add a switch for the whole filter group
    :param fmin:         the minimum frequency
    :param fmax:         the maximum frequency
    :param bands:        number of filter bands per octave
    :param norm_filters: normalize the area of the filter
    :return:             the modified parser object

    """
    # add filter related options to the existing parser
    g = parser.add_argument_group('filtered magnitude spectrogram arguments')
    if filtering is not None:
        g.add_argument('--filter', action='store_true', default=False,
                       help='filter the magnitude spectrogram with a '
                            'filterbank [default=%(default)s]')
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


def log(parser, log=None, mul=MUL, add=ADD):
    """
    Add logarithmic magnitude related arguments to an existing parser object.

    :param parser: existing argparse parser object
    :param log:    add a switch for the whole group
    :param mul:    multiply the magnitude spectrogram with given value
    :param add:    add the given value to the magnitude spectrogram
    :return:       the modified parser object

    """
    # add log related options to the existing parser
    g = parser.add_argument_group('logarithic magnitude arguments')
    if log is not None:
        g.add_argument('--log', action='store_true', default=log,
                       help='logarithmic magnitude [default=%(default)s]')
    if mul is not None:
        g.add_argument('--mul', action='store', type=float, default=mul,
                       help='multiplier (before taking the log) '
                            '[default=%(default)i]')
    if add is not None:
        g.add_argument('--add', action='store', type=float, default=add,
                       help='value added (before taking the log) '
                            '[default=%(default)i]')
    # return the argument group so it can be modified if needed
    return g


def spectral_odf(parser, method='superflux', methods=None, max_bins=MAX_BINS):
    """
    Add spectral ODF related arguments to an existing parser object.

    :param parser:   existing argparse parser object
    :param method:   default ODF method
    :param methods:  list of ODF methods
    :param max_bins: number of bins for the maximum filter (for SuperFlux)
    :return:         the modified parser object

    """
    # add spec related options to the existing parser
    # spectrogram options
    g = parser.add_argument_group('spectral onset detection arguments')
    superflux = False
    if methods is not None:
        g.add_argument('-o', dest='odf', default=method,
                       help='use one of these onset detection functions (%s) '
                            '[default=%s]' % (methods, method))
        if 'superflux' in methods:
            superflux = True
    # add SuperFlux arguments
    if superflux or method == 'superflux':
        g.add_argument('--max_bins', action='store', type=int,
                       default=max_bins,
                       help='bins used for maximum filtering [default='
                            '%(default)i]')
    # return the argument group so it can be modified if needed
    return g


def onset(parser, threshold=THRESHOLD, smooth=SMOOTH, combine=COMBINE,
          delay=DELAY, pre_avg=PRE_AVG, post_avg=POST_AVG, pre_max=PRE_MAX,
          post_max=POST_MAX):
    """
    Add onset detection related arguments to an existing parser object.

    :param parser:    existing argparse parser object
    :param threshold: threshold for peak-picking
    :param smooth:    smooth the onset activations over N seconds
    :param combine:   only report one onset within N seconds
    :param delay:     report onsets N seconds delayed
    :param pre_avg:   use N seconds past information for moving average
    :param post_avg:  use N seconds future information for moving average
    :param pre_max:   use N seconds past information for moving maximum
    :param post_max:  use N seconds future information for moving maximum
    :return:          the modified parser object

    """
    # add onset detection related options to the existing parser
    g = parser.add_argument_group('onset detection arguments')
    g.add_argument('-t', dest='threshold', action='store', type=float,
                   default=threshold,
                   help='detection threshold [default=%(default).2f]')
    g.add_argument('--smooth', action='store', type=float, default=smooth,
                   help='smooth the onset activations over N seconds '
                        '[default=%(default).2f]')
    g.add_argument('--combine', action='store', type=float, default=combine,
                   help='combine onsets within N seconds '
                        '[default=%(default).2f]')
    g.add_argument('--pre_avg', action='store', type=float, default=pre_avg,
                   help='build average over N previous seconds '
                        '[default=%(default).2f]')
    g.add_argument('--post_avg', action='store', type=float, default=post_avg,
                   help='build average over N following seconds '
                        '[default=%(default).2f]')
    g.add_argument('--pre_max', action='store', type=float, default=pre_max,
                   help='search maximum over N previous seconds '
                        '[default=%(default).2f]')
    g.add_argument('--post_max', action='store', type=float, default=post_max,
                   help='search maximum over N following seconds '
                        '[default=%(default).2f]')
    g.add_argument('--delay', action='store', type=float, default=delay,
                   help='report the onsets N seconds delayed '
                        '[default=%(default)i]')
    # return the argument group so it can be modified if needed
    return g


def beat(parser, threshold=BT, smooth=BS, min_bpm=MIN_BPM, max_bpm=MAX_BPM):
    """
    Add beat tracking related arguments to an existing parser object.

    :param parser:    existing argparse parser object
    :param threshold: threshold the beat activation function
    :param smooth:    smooth the beat activations over N seconds
    :param min_bpm:   minimum tempo [bpm]
    :param max_bpm:   maximum tempo [bpm]
    :return:          the modified parser object

    """
    # add onset detection related options to the existing parser
    g = parser.add_argument_group('beat detection arguments')
    g.add_argument('-t', dest='threshold', action='store', type=float,
                   default=threshold,
                   help='detection threshold [default=%(default).2f]')
    g.add_argument('--smooth', action='store', type=float, default=smooth,
                   help='smooth the beat activations over N seconds '
                        '[default=%(default).2f]')
    g.add_argument('--min_bpm', action='store', type=float, default=min_bpm,
                   help='minimum tempo [bpm, default=%(default).2f]')
    g.add_argument('--max_bpm', action='store', type=float, default=max_bpm,
                   help='maximum tempo [bpm, default=%(default).2f]')
    # return the argument group so it can be modified if needed
    return g


def note(parser, threshold=THRESHOLD, smooth=SMOOTH, combine=COMBINE,
         delay=DELAY, pre_avg=PRE_AVG, post_avg=POST_AVG, pre_max=PRE_MAX,
         post_max=POST_MAX):
    """
    Add note transcription related arguments to an existing parser object.

    :param parser:    existing argparse parser object
    :param threshold: threshold for peak-picking
    :param smooth:    smooth the note activations over N seconds
    :param combine:   only report one note within N seconds and pitch
    :param delay:     report notes N seconds delayed
    :param pre_avg:   use N seconds past information for moving average
    :param post_avg:  use N seconds future information for moving average
    :param pre_max:   use N seconds past information for moving maximum
    :param post_max:  use N seconds future information for moving maximum
    :return:          the modified parser object

    """
    # add note transcription detection related options to the existing parser
    g = parser.add_argument_group('note transcription arguments')
    g.add_argument('-t', dest='threshold', action='store', type=float,
                   default=threshold,
                   help='detection threshold [default=%.2f]' % threshold)
    g.add_argument('--smooth', action='store', type=float, default=smooth,
                   help='smooth the note activations over N seconds '
                        '[default=%.2f]' % smooth)
    g.add_argument('--combine', action='store', type=float, default=combine,
                   help='combine notes within N seconds (per pitch)'
                        '[default=%.2f]' % combine)
    g.add_argument('--pre_avg', action='store', type=float, default=pre_avg,
                   help='build average over N previous seconds '
                        '[default=%.2f]' % pre_avg)
    g.add_argument('--post_avg', action='store', type=float, default=post_avg,
                   help='build average over N following seconds '
                        '[default=%.2f]' % post_avg)
    g.add_argument('--pre_max', action='store', type=float, default=pre_max,
                   help='search maximum over N previous seconds '
                        '[default=%.2f]' % pre_max)
    g.add_argument('--post_max', action='store', type=float, default=post_max,
                   help='search maximum over N following seconds '
                        '[default=%.2f]' % post_max)
    g.add_argument('--delay', action='store', type=float, default=delay,
                   help='report the notes N seconds delayed '
                        '[default=%i]' % delay)
    # return the argument group so it can be modified if needed
    return g


def io(parser):
    """
    Add options to save/load activations to an existing parser object.

    :param parser: existing argparse parser object
    :return:       the modified parser object

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
                        'function [default: numpy binary format]')


def nn(parser, nn_files=None, threads=multiprocessing.cpu_count()):
    """
    Add neural network testing options to an existing parser object.

    :param parser:   existing argparse parser object
    :param nn_files: list of NN files
    :param threads:  number of threads to run in parallel
    :return:         the modified parser object

    """
    # add neural network related options to the existing parser
    g = parser.add_argument_group('neural network arguments')
    g.add_argument('--nn_files', action='append', type=str, default=nn_files,
                   help='average the predictions of these pre-trained neural '
                        'networks (multiple files can be given, one file per '
                        'argument)')
    g.add_argument('--threads', action='store', type=int, default=threads,
                   help='number of parallel threads [default=%(default)s]')


def mirex(parser):
    """
    Add MIREX related input / output related arguments to an existing parser
    object.

    :param parser: existing argparse parser object
    :return:       the modified parser object

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


def parser():
    """
    Example parser.

    :return: the parsed arguments

    """
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""This is just an example parser.""")
    # general options
    p.add_argument('files', metavar='files', nargs='+',
                   help='files to be processed')
    p.add_argument('-v', dest='verbose', action='count',
                   help='increase verbosity level')
    # add other argument groups
    audio(p)
    filtering(p)
    log(p)
    spectral_odf(p)
    o = onset(p)
    o.add_argument('--not_needed', action='store_true', default=True,
                   help='we usually do not need this option')
    # version
    p.add_argument('--version', action='version', version='%(prog)s 0.1')
    # parse arguments
    args = p.parse_args()
    # return args
    return args
