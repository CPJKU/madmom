#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

from madmom.utils import io_arguments
from madmom.features import ActivationsProcessor
from madmom.features.onsets import RNNOnsetDetection


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
    given input (file) and writes them to the output (file) with the algorithm
    introduced in:

    "Online Real-time Onset Detection with Recurrent Neural Networks"
    Sebastian Böck, Andreas Arzt, Florian Krebs and Markus Schedl
    Proceedings of the 15th International Conference on Digital Audio Effects
    (DAFx-12), 2012.

    The paper includes an error in Section 2.2.1, 2nd paragraph:
    The targets of the training examples have been annotated 1 frame shifted to
    the future, thus the results given in Table 2 are off by 10ms. Given the
    fact that the delayed reporting (as outlined in Section 2.3) is not
    needed, an extra shift of 5ms needs to be added to the mean errors given in
    Table 2.

    This implementation takes care of this error is is modified in this way:
    - a logarithmic frequency spacing is used for the spectrograms instead of
      the Bark scale
    - targets are annotated at the next frame for neural network training
    - post processing reports the onset instantaneously instead of delayed.

    ''')
    # add arguments
    io_arguments(p)
    ActivationsProcessor.add_arguments(p)
    RNNOnsetDetection.add_arguments(p, online=True)
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
    """OnsetDetectorLL.2013"""

    # parse arguments
    args = parser()

    # create an processor
    processor = RNNOnsetDetection(online=True, **vars(args))
    # swap in/out processors if needed
    if args.load:
        processor.in_processor = ActivationsProcessor(mode='r', **vars(args))
    if args.save:
        processor.out_processor = ActivationsProcessor(mode='w', **vars(args))

    # process everything
    processor.process(args.input, args.output)


if __name__ == '__main__':
    main()
