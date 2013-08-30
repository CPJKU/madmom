#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all parser functionality used by other modules.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse

# get the default values from the corresponding modules
import cp.audio.audio as a
import cp.audio.spectrogram as s
import cp.audio.filterbank as f
import cp.audio.onset_detection as od


def add_audio_arguments(parser, online=a.ONLINE, norm=a.NORM, att=a.ATT, fps=a.FPS, window=a.FRAME_SIZE):
    """
    Add audio related arguments to an existing parser object.

    :param parser: existing argparse parser object
    :param online: online mode
    :param norm:   normalize the signal
    :param att:    attenuate the signal by N dB
    :param fps:    frames per second
    :param window: window / frame size
    :return:       the modified parser object

    """
    # if audio gets normalized, switch to offline mode
    if norm:
        online = False
    # add wav options to the existing parser
    group = parser.add_argument_group('audio arguments')
    if online is not None:
        group.add_argument('--online', dest='online', action='store_true', default=online, help='operate in online mode [default=%s]' % online)
    if norm is not None:
        group.add_argument('--norm', action='store_true', default=norm, help='normalize the audio signal (switches to offline mode)')
    if att is not None:
        group.add_argument('--att', action='store', type=float, default=att, help='attenuate the audio signal [dB]')
    if fps is not None:
        group.add_argument('--fps', action='store', type=int, default=fps, help='frames per second [default=%i]' % fps)
    if window is not None:
        group.add_argument('--window', action='store', type=int, default=window, help='frame length [samples, default=%i]' % window)
    # return the argument group so it can be modified if needed
    return group


def add_spec_arguments(parser, ratio=s.RATIO, diff_frames=s.DIFF_FRAMES):
    """
    Add spectrogram related arguments to an existing parser object.

    :param parser:      existing argparse parser object
    :param ratio:       calculate the difference to the frame which window overlaps to this ratio
    :param diff_frames: calculate the difference to the N-th previous frame
    :return:            the modified parser object

    """
    # add spec related options to the existing parser
    # spectrogram options
    group = parser.add_argument_group('spectrogram arguments')
    group.add_argument('--diff_frames', action='store', type=int, default=diff_frames, help='diff frames [default=%s]' % diff_frames)
    group.add_argument('--ratio', action='store', type=float, default=ratio, help='window magnitude ratio to calc number of diff frames [default=%.1f]' % ratio)
    # return the argument group so it can be modified if needed
    return group


def add_filter_arguments(parser, filtering=None, fmin=f.FMIN, fmax=f.FMAX,
                         bands=f.BANDS_PER_OCTAVE, norm_filter=f.NORM_FILTER):
    """
    Add filter related arguments to an existing parser object.

    :param parser:    existing argparse parser object
    :param filtering: add a switch for the whole filter group
    :param bands:     number of filter bands per octave
    :param fmin:      the minimum frequency
    :param fmax:      the maximum frequency
    :param equal:     equal (normalize) the area of the filter
    :return:          the modified parser object

    """
    # add filter related options to the existing parser
    group = parser.add_argument_group('filter arguments')
    if filtering is not None:
        group.add_argument('--filter', action='store_true', default=False, help='filter the magnitude spectrogram with a filterbank [default=False]')
    if bands is not None:
        group.add_argument('--bands', action='store', type=int, default=bands, help='number of bands per octave [default=%i]' % bands)
    if fmin is not None:
        group.add_argument('--fmin', action='store', type=float, default=fmin, help='minimum frequency of filter in Hz [default=%i]' % fmin)
    if fmax is not None:
        group.add_argument('--fmax', action='store', type=float, default=fmax, help='maximum frequency of filter in Hz [default=%i]' % fmax)
    if norm_filter is not None:
        group.add_argument('--norm_filter', action='store_true', default=norm_filter, help='equalize filters to have equal area [default=%s]' % norm_filter)
    # return the argument group so it can be modified if needed
    return group


def add_log_arguments(parser, log=None, mul=s.MUL, add=s.ADD):
    """
    Add logarithmic magnitude related arguments to an existing parser object.

    :param parser: existing argparse parser object
    :param log:    add a switch for the whole group
    :param mul:    multiply the magnitude spectrogram with given value
    :param add:    add the given value to the magnitude spectrogram
    :return:       the modified parser object

    """
    # add log related options to the existing parser
    group = parser.add_argument_group('logarithic magnitude arguments')
    if log is not None:
        group.add_argument('--log', action='store_true', default=log, help='logarithmic magnitude [default=%s]' % log)
    if mul is not None:
        group.add_argument('--mul', action='store', type=float, default=mul, help='multiplier (before taking the log) [default=%i]' % mul)
    if add is not None:
        group.add_argument('--add', action='store', type=float, default=add, help='value added (before taking the log) [default=%i]' % add)
    # return the argument group so it can be modified if needed
    return group


def add_spectral_odf_arguments(parser, method='superflux', methods=None, max_bins=od.MAX_BINS):
    """
    Add spectral ODF related arguments to an existing parser object.

    :param parser:      existing argparse parser object
    :param method:      default ODF method
    :param methods:     list of ODF methods
    :param max_bins:    number of bins for the maximum filter (for SuperFlux)
    :return:            the modified parser object

    """
    # add spec related options to the existing parser
    # spectrogram options
    group = parser.add_argument_group('spectral onset detection function arguments')
    if methods is not None:
        group.add_argument('-o', dest='odf', default=method, help='use one of these onset detection functions (%s) [default=%s]' % (methods, method))
    #if 'superflux' in methods or method == 'superflux':
    group.add_argument('--max_bins', action='store', type=int, default=max_bins, help='bins used for maximum filtering [default=%i]' % max_bins)
    # return the argument group so it can be modified if needed
    return group


def add_onset_arguments(parser, threshold=od.THRESHOLD, smooth=od.SMOOTH, combine=od.COMBINE, delay=od.DELAY,
                        pre_avg=od.PRE_AVG, post_avg=od.POST_AVG, pre_max=od.PRE_MAX, post_max=od.POST_MAX):
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
    group = parser.add_argument_group('onset detection arguments')
    group.add_argument('-s', dest='save', action='store_true', default=False, help='save the activations of the onset detection function')
    group.add_argument('-l', dest='load', action='store_true', default=False, help='load the activations of the onset detection function')
    group.add_argument('-t', dest='threshold', action='store', type=float, default=threshold, help='detection threshold [default=%.2f]' % threshold)
    group.add_argument('--smooth', action='store', type=float, default=smooth, help='smooth the onset activations over N seconds [default=%.2f]' % smooth)
    group.add_argument('--combine', action='store', type=float, default=combine, help='combine onsets within N seconds [default=%.2f]' % combine)
    group.add_argument('--pre_avg', action='store', type=float, default=pre_avg, help='build average over N previous seconds [default=%.2f]' % pre_avg)
    group.add_argument('--post_avg', action='store', type=float, default=post_avg, help='build average over N following seconds [default=%.2f]' % post_avg)
    group.add_argument('--pre_max', action='store', type=float, default=pre_max, help='search maximum over N previous seconds [default=%.2f]' % pre_max)
    group.add_argument('--post_max', action='store', type=float, default=post_max, help='search maximum over N following seconds [default=%.2f]' % post_max)
    group.add_argument('--delay', action='store', type=float, default=delay, help='report the onsets N seconds delayed [default=%i]' % delay)
    group.add_argument('--sep', action='store', default='', help='separator for saving/loading the onset detection functions [default=\'\' (numpy binary format)]')
    # return the argument group so it can be modified if needed
    return group


def add_mirex_io(parser):
    """
    Add MIREX related input / output related arguments to an existing parser object.

    :param parser: existing argparse parser object
    :return:       the modified parser object

    """
    import sys
    # general options
    parser.add_argument('input', type=argparse.FileType('r'), help='input file (.wav or saved onset activations)')
    parser.add_argument('output', nargs='?', type=argparse.FileType('w'), default=sys.stdout, help='output file [default: STDOUT]')
    parser.add_argument('-v', dest='verbose', action='count', help='increase verbosity level')


def parser():
    # define parser
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="""This is just an example parser.""")
    # general options
    p.add_argument('files', metavar='files', nargs='+', help='files to be processed')
    p.add_argument('-v', dest='verbose', action='count', help='increase verbosity level')
    # add other argument groups
    add_audio_arguments(p)
    add_filter_arguments(p)
    add_log_arguments(p)
    add_spectral_odf_arguments(p)
    onset = add_onset_arguments(p)
    onset.add_argument('--not_needed', action='store_true', default=True, help='we usually do not need this option')
    # version
    p.add_argument('--version', action='version', version='%(prog)s 1.0')
    # parse arguments
    args = p.parse_args()
    # return args
    return args
