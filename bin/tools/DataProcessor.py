#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse
import numpy as np

from madmom import IOProcessor, io_arguments
from madmom.utils import io_arguments
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import (SpectrogramProcessor,
                                      StackSpectrogramProcessor)


def writer(data, outfile):
    """
    Wrapper around np.save which swaps the arguments to the correct position.

    :param data:    data to be saved
    :param outfile: output file

    """
    np.save(outfile, data)


def main():
    """DataProcessor"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software processes audio files in the given manner and saves them as
    numpy files.

    ''')
    # version
    p.add_argument('--version', action='version', version='DataProcessor')
    # add arguments
    io_arguments(p, suffix='.npy')
    SignalProcessor.add_arguments(p, norm=False, att=0)
    FramedSignalProcessor.add_arguments(p, frame_size=[1024, 2048, 4096],
                                        fps=100, online=False)
    SpectrogramProcessor.add_filter_arguments(p, bands=12, fmin=30, fmax=17000,
                                              norm_filters=True,
                                              duplicate_filters=False)
    SpectrogramProcessor.add_log_arguments(p, log=True, mul=1, add=1)
    SpectrogramProcessor.add_diff_arguments(p, diff_ratio=0.5, diff_max_bins=1)
    StackSpectrogramProcessor.add_arguments(p)
    # parse arguments
    args = p.parse_args()
    # switch to offline mode
    if args.norm:
        args.online = False
    # print arguments
    if args.verbose:
        print args

    # TODO: add default settings for onsets/beats/etc.?

    # create a processor
    sig = SignalProcessor(num_channels=1, **vars(args))
    stack = StackSpectrogramProcessor(**vars(args))
    processor = IOProcessor([sig, stack], writer)
    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
