#!/usr/bin/env python
# encoding: utf-8
"""
ComplexFlux onset detection algorithm.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse

from madmom.utils import io_arguments
from madmom.features.onsets import SpectralOnsetDetection as ComlpexFlux


def main():
    """ComplexFlux"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all onsets in an audio file with the SuperFlux
    algorithm with additional tremolo suppression as described in:

    "Local group delay based vibrato and tremolo suppression for onset
     detection"
    Sebastian Böck and Gerhard Widmer.
    Proceedings of the 13th International Society for Music Information
    Retrieval Conference (ISMIR), 2013.

    ''')
    # version
    p.add_argument('--version', action='version', version='ComplexFlux.2014')
    # add arguments
    io_arguments(p, suffix='.onsets.txt')
    ComlpexFlux.add_activation_arguments(p)
    ComlpexFlux.add_signal_arguments(p, norm=False, att=0)
    ComlpexFlux.add_framing_arguments(p, fps=200, online=False)
    ComlpexFlux.add_filter_arguments(p, bands=24, fmin=30, fmax=17000,
                                     norm_filters=False)
    ComlpexFlux.add_log_arguments(p, log=True, mul=1, add=1)
    ComlpexFlux.add_diff_arguments(p, diff_ratio=0.5, diff_max_bins=3)
    ComlpexFlux.add_peak_picking_arguments(p, threshold=0.25, pre_max=0.01,
                                           post_max=0.05, pre_avg=0.15,
                                           post_avg=0, combine=0.03, delay=0)
    # parse arguments
    args = p.parse_args()
    # switch to offline mode
    if args.norm:
        args.online = False
    # print arguments
    if args.verbose:
        print args

    # create a processor
    processor = ComlpexFlux(onset_method='complex_flux', **vars(args))
    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
