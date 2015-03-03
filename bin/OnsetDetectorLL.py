#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import argparse

from madmom.utils import io_arguments
from madmom.features.onsets import RNNOnsetDetection


def main():
    """OnsetDetectorLL.2013"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The software detects all onsets in an audio file with the algorithm
    described in:

    "Online Real-time Onset Detection with Recurrent Neural Networks"
    Sebastian Böck, Andreas Arzt, Florian Krebs and Markus Schedl
    Proceedings of the 15th International Conference on Digital Audio Effects
    (DAFx), 2012.

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
    # version
    p.add_argument('--version', action='version', version='OnsetDetector.2013')
    # add arguments
    io_arguments(p)
    RNNOnsetDetection.add_arguments(p, online=True)
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args

    # create a processor
    processor = RNNOnsetDetection(online=True, **vars(args))
    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()
