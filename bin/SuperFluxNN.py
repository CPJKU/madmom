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
from madmom.features.onsets import (NNSpectralOnsetDetection,
                                    SpectralOnsetDetection)


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects all onsets in the
    given input file and writes them to the output file with the algorithm
    introduced in:

    "Enhanced peak picking for onset detection with recurrent neural networks"
    Sebastian Böck, Jan Schlüter and Gerhard Widmer
    Proceedings of the 6th International Workshop on Machine Learning and
    Music (MML), 2013.

    Please note that this implementation uses 100 frames per second (instead
    of 200), because it is faster and produces highly comparable results.

    ''')
    # input/output options
    madmom.utils.io_arguments(p)
    # add other argument groups
    Signal.add_arguments(p)
    FramedSignal.add_arguments(p, fps=100, online=False)
    Filterbank.add_arguments(p, bands=24, norm_filters=False)
    LogFiltSpec.add_arguments(p, log=True, mul=1, add=1)
    SpectralOnsetDetection.add_arguments(p)
    NNSpectralOnsetDetection.add_arguments(p)
    # version
    p.add_argument('--version', action='version', version='SuperFluxNN')
    # parse arguments
    args = p.parse_args()
    # switch to offline mode
    if args.norm:
        args.online = False
    # translate online/offline mode
    if args.online:
        args.origin = 'online'
        args.smooth = 0
    else:
        args.origin = 'offline'
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """SuperFluxNN"""

    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # instantiate OnsetDetection object from activations
        o = NNSpectralOnsetDetection.from_activations(args.input, sep=args.sep)
    else:
        # create a logarithmically filtered Spectrogram object
        s = LogFiltSpec(args.input, mono=True, norm=args.norm, att=args.att,
                        frame_size=args.frame_size, origin=args.origin,
                        fps=args.fps, bands_per_octave=args.bands,
                        fmin=args.fmin, fmax=args.fmax, mul=args.mul,
                        add=args.add, norm_filters=args.norm_filters,
                        ratio=args.ratio, diff_frames=args.diff_frames)
        # create a SpectralOnsetDetection detection object
        o = NNSpectralOnsetDetection.from_data(s, fps=args.fps)
        o.max_bins = args.max_bins
        # process with the detection function
        o.superflux()

    # save onset activations or detect onsets
    if args.save:
        # save activations
        o.activations.save(args.output, sep=args.sep)
    else:
        # detect the onsets
        o.detect(args.threshold, combine=args.combine, delay=args.delay,
                 smooth=args.smooth, online=args.online)
        # write the onsets to output
        o.write(args.output)

if __name__ == '__main__':
    main()
