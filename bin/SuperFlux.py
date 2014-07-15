#!/usr/bin/env python
# encoding: utf-8
"""
SuperFlux onset detection algorithm.

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
    algorithm introduced in:

    "Maximum Filter Vibrato Suppression for Onset Detection"
    by Sebastian Böck and Gerhard Widmer
    Proceedings of the 16th International Conference on Digital Audio Effects
    (DAFx-13), Maynooth, Ireland, September 2013

    ''')
    # input/output options
    madmom.utils.io_arguments(p)
    # add other argument groups
    Signal.add_arguments(p)
    FramedSignal.add_arguments(p, fps=200, online=False)
    Filterbank.add_arguments(p, bands=24, norm_filters=False)
    LogFiltSpec.add_arguments(p, log=True, mul=1, add=1)
    SpectralOnsetDetection.add_arguments(p)
    OnsetDetection.add_arguments(p)
    # version
    p.add_argument('--version', action='version', version='SuperFlux.2013')
    # parse arguments
    args = p.parse_args()
    # switch to offline mode
    if args.norm:
        args.online = False
    # translate online/offline mode
    if args.online:
        args.origin = 'online'
    else:
        args.origin = 'offline'
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """SuperFlux.2013"""

    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # instantiate OnsetDetection object from activations
        o = OnsetDetection.from_activations(args.input, fps=args.fps,
                                            sep=args.sep)
    else:
        # create a Signal object
        s = Signal(args.input, mono=True, norm=args.norm, att=args.att)
        # create a SpectralOnsetDetection detection object
        o = SpectralOnsetDetection(s, max_bins=args.max_bins)
        # do signal processing
        o.pre_process(frame_size=args.frame_size, origin=args.origin,
                      fps=args.fps, bands_per_octave=args.bands,
                      mul=args.mul, add=args.add,
                      norm_filters=args.norm_filters)
        # process with the detection function
        o.superflux()

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
