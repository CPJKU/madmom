#!/usr/bin/env python
# encoding: utf-8
"""
SuperFlux with neural network based peak picking onset detection algorithm.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse

from madmom.utils import io_arguments
from madmom.features.onsets import (
    SpectralOnsetDetectionProcessor as SuperFlux, NNPeakPickingProcessor)


def main():
    """SuperFluxNN"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all onsets in an audio file with the SuperFlux
    algorithm with neural network based peak-picking as described in:

    "Enhanced peak picking for onset detection with recurrent neural networks"
    Sebastian Böck, Jan Schlüter and Gerhard Widmer
    Proceedings of the 6th International Workshop on Machine Learning and
    Music (MML), 2013.

    Please note that this implementation uses 100 frames per second (instead
    of 200), because it is faster and produces highly comparable results.

    ''')
    # add arguments
    io_arguments(p, suffix='.onsets.txt')
    SuperFlux.add_activation_arguments(p)
    SuperFlux.add_signal_arguments(p, norm=False, att=0)
    SuperFlux.add_framing_arguments(p, fps=100, online=False)
    SuperFlux.add_filter_arguments(p, bands=24, fmin=30, fmax=17000,
                                   norm_filters=False)
    SuperFlux.add_log_arguments(p, log=True, mul=1, add=1)
    SuperFlux.add_diff_arguments(p, diff_ratio=0.5, diff_max_bins=3)
    NNPeakPickingProcessor.add_arguments(p)
    # version
    p.add_argument('--version', action='version', version='SuperFluxNN')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args

    # create a processor
    processor = SuperFlux(peak_picking_method='nn', **vars(args))
    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
