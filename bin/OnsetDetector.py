#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse
import glob

from madmom.utils import write_events, io_arguments
from madmom import SequentialProcessor, IOProcessor
from madmom.audio.signal import SignalProcessor
from madmom.audio.spectrogram import StackSpectrogramProcessor
from madmom.ml.rnn import RNNProcessor
from madmom.features.peak_picking import PeakPickingProcessor
from madmom.features import ActivationsProcessor
from madmom import MODELS_PATH

# set the path to saved neural networks and generate a list of NN files
NN_FILES = glob.glob("%s/onsets_brnn_[1-8].npz" % MODELS_PATH)


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects all onsets in the
    given input (file) and writes them to the output (file).
    ''')

    # input/output options
    io_arguments(p)
    # add other argument groups
    SignalProcessor.add_arguments(p, att=0, norm=False)
    # rnn onset detection arguments
    RNNProcessor.add_arguments(p, nn_files=NN_FILES)
    PeakPickingProcessor.add_arguments(p, threshold=0.35, smooth=0.07,
                                       combine=0.03, delay=0)
    ActivationsProcessor.add_arguments(p)
    # version
    p.add_argument('--version', action='version', version='OnsetDetector.2013')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """OnsetDetector.2013"""

    # parse arguments
    args = parser()
    # set the frame rate
    args.fps = 100

    # load or create onset activations
    if args.load:
        # load the activations
        act = ActivationsProcessor(mode='r', **vars(args))
        in_processor = SequentialProcessor([act])
    else:
        # signal handling processor
        sig = SignalProcessor(**vars(args))
        # parallel specs + stacking processor
        stack = StackSpectrogramProcessor(frame_sizes=[1024, 2048, 4096],
                                          fps=args.fps, online=False, bands=6,
                                          norm_filters=True, mul=5, add=1,
                                          diff_ratio=0.25)
        # multiple RNN processor
        rnn = RNNProcessor(nn_files=args.nn_files,
                           num_threads=args.num_threads)
        # sequentially process everything
        in_processor = SequentialProcessor([sig, stack, rnn])

    # save onset activations or detect onsets
    if args.save:
        # save activations
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        # peak-picking & output processor
        pp = PeakPickingProcessor(pre_max=0.01, post_max=0.01, **vars(args))
        in_processor.append(pp)
        out_processor = write_events

    # process everything
    IOProcessor(in_processor, out_processor).process(args.input, args.output)


if __name__ == '__main__':
    main()
