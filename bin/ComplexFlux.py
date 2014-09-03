#!/usr/bin/env python
# encoding: utf-8
"""
ComplexFlux onset detection algorithm.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
import argparse

import madmom.utils

from madmom.audio.signal import Signal, FramedSignal
from madmom.audio.filters import Filterbank
from madmom.audio.spectrogram import LogFiltSpec
from madmom.features.onsets import SpectralOnsetDetection, OnsetDetection


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects all onsets in the
    given input file and writes them to the output file with the SuperFlux
    algorithm with additional tremolo suppression as introduced in:

    "Local group delay based vibrato and tremolo suppression for onset
     detection"
    Sebastian Böck and Gerhard Widmer.
    Proceedings of the 13th International Society for Music Information
    Retrieval Conference (ISMIR), 2013.

    ''')
    # input/output options
    madmom.utils.io_arguments(p)
    # add other argument groups
    Signal.add_arguments(p)
    FramedSignal.add_arguments(p, fps=200, online=False)
    Filterbank.add_arguments(p, bands=24, norm_filters=False)
    LogFiltSpec.add_arguments(p, log=True, mul=1, add=1)
    g = SpectralOnsetDetection.add_arguments(p)
    g.add_argument('--temporal_filter', action='store', type=float,
                   default=SpectralOnsetDetection.TEMPORAL_FILTER,
                   help='use temporal maximum filtering over N seconds '
                        '[default=%(default).3f]')
    OnsetDetection.add_arguments(p, threshold=0.25, pre_max=0.01, post_max=0.05,
                                 pre_avg=0.15, post_avg=0)
    # version
    p.add_argument('--version', action='version', version='ComplexFlux')
    # parse arguments
    args = p.parse_args()
    # switch to offline mode
    if args.norm:
        args.online = False
    # translate online/offline mode
    if args.online:
        args.origin = 'online'
        args.post_max = 0
        args.post_avg = 0
    else:
        args.origin = 'offline'
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """ComplexFlux"""

    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # instantiate OnsetDetection object from activations
        o = SpectralOnsetDetection.from_activations(args.input, fps=args.fps,
                                                    sep=args.sep)
    else:
        # create a logarithmically filtered Spectrogram object
        s = LogFiltSpec(args.input, mono=True, norm=args.norm, att=args.att,
                        frame_size=args.frame_size, origin=args.origin,
                        fps=args.fps, bands_per_octave=args.bands,
                        fmin=args.fmin, fmax=args.fmax, mul=args.mul,
                        add=args.add, norm_filters=args.norm_filters,
                        ratio=args.ratio, diff_frames=args.diff_frames)
        # create a SpectralOnsetDetection detection object
        o = SpectralOnsetDetection.from_data(s, fps=args.fps)
        o.max_bins = args.max_bins
        # process with the detection function
        o.complex_flux(temporal_filter=args.temporal_filter)

    # save onset activations or detect onsets
    if args.save:
        # save activations
        o.activations.save(args.output, sep=args.sep)
    else:
        # detect the onsets
        o.detect(args.threshold, combine=args.combine, delay=args.delay,
                 smooth=args.smooth, pre_avg=args.pre_avg,
                 post_avg=args.post_avg, pre_max=args.pre_max,
                 post_max=args.post_max, online=args.online)
        # write the onsets to output
        o.write(args.output)

if __name__ == '__main__':
    main()
