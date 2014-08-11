#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) Sebastian Böck <sebastian.boeck@jku.at>

Redistribution in any form is not permitted!

"""

from madmom.audio.signal import Signal
from madmom.features.beats import MMBeatTracking


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse
    import madmom.utils

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects all beats in the
    given input (file) and writes them to the output (file) according to the
    method described in:

    "A multi-model approach to beat tracking considering heterogeneous music
     styles"
    Sebastian Böck, Florian Krebs and Gerhard Widmer
    Proceedings of the 15th International Society for Music Information
    Retrieval Conference (ISMIR 2014), Taipeh, Taiwan, November 2014

    ''')

    # input/output options
    madmom.utils.io_arguments(p)
    # signal arguments
    Signal.add_arguments(p, norm=False)
    # beat tracking arguments
    MMBeatTracking.add_arguments(p)
    # version
    p.add_argument('--version', action='version', version='MMBeatTracker')
    # parse arguments
    args = p.parse_args()
    if args.dbn_file in ['None', '']:
        args.dbn_file = None
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """MMBeatTracker"""

    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # load activations
        b = MMBeatTracking.from_activations(args.input, fps=100)
        b.dbn_file = args.dbn_file
        # set the number of threads, since the detection works multi-threaded
        b.num_threads = args.num_threads
    else:
        # exit if no NN files are given
        if not args.nn_files:
            raise SystemExit('no NN model(s) given')

        # create a Signal object
        s = Signal(args.input, mono=True, norm=args.norm, att=args.att)
        # create an RNNBeatTracking object
        b = MMBeatTracking(s, nn_files=args.nn_files,
                           nn_ref_files=args.nn_ref_files,
                           dbn_file=args.dbn_file,
                           num_threads=args.num_threads)

    # save beat activations or detect beats
    if args.save:
        # save activations
        b.activations.save(args.output, sep=args.sep)
    else:
        # detect the beats
        b.detect(num_beat_states=args.num_beat_states,
                 tempo_change_probability=args.tempo_change_probability,
                 observation_lambda=args.observation_lambda,
                 min_bpm=args.min_bpm, max_bpm=args.max_bpm,
                 correct=args.correct,
                 norm_observations=args.norm_observations)
        # save detections
        b.write(args.output)

if __name__ == '__main__':
    main()
