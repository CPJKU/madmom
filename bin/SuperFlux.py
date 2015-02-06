#!/usr/bin/env python
# encoding: utf-8
"""
SuperFlux onset detection algorithm.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from madmom.utils import io_arguments
from madmom.features import ActivationsProcessor
from madmom.features.onsets import SpectralOnsetProcessor as SuperFlux


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse

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
    # add arguments
    io_arguments(p)
    ActivationsProcessor.add_arguments(p)
    SuperFlux.add_signal_arguments(p, norm=False, att=0)
    SuperFlux.add_framing_arguments(p, fps=200, online=False)
    SuperFlux.add_filter_arguments(p, bands=24, fmin=30, fmax=17000,
                                   norm_filters=False)
    SuperFlux.add_log_arguments(p, log=True, mul=1, add=1)
    SuperFlux.add_diff_arguments(p, diff_ratio=0.5, diff_max_bins=3)
    SuperFlux.add_peak_picking_arguments(p, threshold=1.1, pre_max=0.01,
                                         post_max=0.05, pre_avg=0.15,
                                         post_avg=0, combine=0.03, delay=0)
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

    # create an processor
    processor = SuperFlux(**vars(args))
    # swap in/out processors if needed
    if args.load:
        processor.in_processor = ActivationsProcessor(mode='r', **vars(args))
    if args.save:
        processor.out_processor = ActivationsProcessor(mode='w', **vars(args))

    # process everything
    processor.process(args.input, args.output)

if __name__ == '__main__':
    main()
