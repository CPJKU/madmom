#!/usr/bin/env python
# encoding: utf-8
"""
SuperFlux onset detection algorithm.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
import argparse

from madmom.utils import write_events, io_arguments
from madmom import SequentialProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import SpectrogramProcessor
from madmom.features.onsets import SpectralOnsetDetectionProcessor
from madmom.features.peak_picking import PeakPickingProcessor
from madmom.features import Activations


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
    Sebastian Böck and Gerhard Widmer
    Proceedings of the 16th International Conference on Digital Audio Effects
    (DAFx-13), 2013.

    ''')
    # input/output options
    io_arguments(p)
    # add other argument groups
    SignalProcessor.add_arguments(p, att=0, norm=False)
    FramedSignalProcessor.add_arguments(p, fps=200, online=False)
    SpectrogramProcessor.add_filter_arguments(p, bands=24, fmin=30, fmax=17000,
                                              norm_filters=False)
    SpectrogramProcessor.add_log_arguments(p, log=True, mul=1, add=1)
    SpectrogramProcessor.add_diff_arguments(p, diff_ratio=0.5, diff_max_bins=3)
    PeakPickingProcessor.add_arguments(p, threshold=1.1, pre_max=0.01,
                                       post_max=0.05, pre_avg=0.15, post_avg=0,
                                       combine=0.03, delay=0)
    Activations.add_arguments(p)
    # version
    p.add_argument('--version', action='version', version='SuperFlux.2014')
    # parse arguments
    args = p.parse_args()
    # switch to offline mode
    if args.norm:
        args.online = False
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """SuperFlux.2014"""

    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # load the activations
        act = Activations.load(args.input, fps=args.fps, sep=args.sep)
    else:
        # create processors
        p1 = SignalProcessor(**vars(args))
        p2 = FramedSignalProcessor(**vars(args))
        p3 = SpectrogramProcessor(**vars(args))
        p4 = SpectralOnsetDetectionProcessor(odf='superflux', **vars(args))
        # sequentially process everything
        act = SequentialProcessor([p1, p2, p3, p4]).process(args.input)

    # save onset activations or detect onsets
    if args.save:
        # save activations
        Activations(act, fps=args.fps).save(args.output, sep=args.sep)
    else:
        # detect the onsets and write them to file/stdout
        onsets = PeakPickingProcessor(**vars(args)).process(act)
        write_events(onsets, args.output)

if __name__ == '__main__':
    main()
