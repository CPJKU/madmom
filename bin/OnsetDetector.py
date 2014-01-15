#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2012-2013 Sebastian Böck <sebastian.boeck@jku.at>

Redistribution in any form is not permitted!
"""

import os
import glob
import numpy as np
import itertools as it
import multiprocessing as mp

from madmom.audio.wav import Wav
from madmom.audio.spectrogram import LogFiltSpec
from madmom.features.onsets import Onset
from madmom.ml.rnn import RecurrentNeuralNetwork

# set the path to saved neural networks and generate lists of NN files
NN_PATH = '%s/../madmom/ml/data' % (os.path.dirname(__file__))
NN_FILES = glob.glob("%s/onsets_brnn*npz" % NN_PATH)

# TODO: this information should be included/extracted in/from the NN files
FPS = 100
BANDS_PER_OCTAVE = 6
MUL = 5
ADD = 1
FMIN = 27.5
FMAX = 18000
RATIO = 0.25
NORM_FILTERS = True


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments
    """
    import argparse
    import madmom.utils.params

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects all onsets in the
    given input (file) and writes them to the output (file).
    ''')
    # mirex options
    madmom.utils.params.mirex(p)
    # add other argument groups
    madmom.utils.params.nn(p)
    madmom.utils.params.audio(p, fps=None, norm=False, online=None, window=None)
    madmom.utils.params.onset(p, threshold=0.35, combine=0.03, smooth=0.07,
                              pre_avg=0, post_avg=0, pre_max=1. / FPS,
                              post_max=1. / FPS)
    madmom.utils.params.io(p)
    # version
    p.add_argument('--version', action='version', version='OnsetDetector.2013')
    # parse arguments
    args = p.parse_args()
    # set some defaults
    args.fps = FPS
    args.online = False
    if args.nn_files is None:
        args.nn_files = NN_FILES
    args.threads = min(len(args.nn_files), args.threads)
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def process((nn_file, data)):
    """
    Loads a RNN model from the given file (first tuple item) and passes the
    given numpy array of data through it (second tuple item).

    """
    return RecurrentNeuralNetwork(nn_file).activate(data)


def main():
    """OnsetDetector.2013"""

    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # load activations
        o = Onset(args.input, args.fps, args.online, args.sep)
    else:
        # exit if no NN files are given
        if not args.nn_files:
            raise SystemExit('no NN model(s) given')

        # create a Wav object
        w = Wav(args.input, mono=True, norm=args.norm, att=args.att)
        # 1st spec
        s = LogFiltSpec(w, frame_size=1024, fps=FPS,
                        bands_per_octave=BANDS_PER_OCTAVE, mul=MUL, add=ADD,
                        norm_filters=NORM_FILTERS)
        data = np.hstack((s.spec, s.pos_diff))
        # 2nd spec
        s = LogFiltSpec(w, frame_size=2048, fps=FPS,
                        bands_per_octave=BANDS_PER_OCTAVE, mul=MUL, add=ADD,
                        norm_filters=NORM_FILTERS)
        data = np.hstack((data, s.spec, s.pos_diff))
        # 3rd spec
        s = LogFiltSpec(w, frame_size=4096, fps=FPS,
                        bands_per_octave=BANDS_PER_OCTAVE, mul=MUL, add=ADD,
                        norm_filters=NORM_FILTERS)
        # stack the data
        data = np.hstack((data, s.spec, s.pos_diff))

        # init a pool of workers (if needed)
        mp_map = mp.Pool(args.threads).map if args.threads != 1 else map

        # compute predictions with all saved neural networks (in parallel)
        activations = mp_map(process, it.izip(args.nn_files, it.repeat(data)))
        # average activations
        nn_files = len(args.nn_files)
        act = sum(activations) / nn_files if nn_files > 1 else activations[0]

        # create an Onset object with the activations
        o = Onset(act.ravel(), args.fps, args.online)

    # save onset activations or detect onsets
    if args.save:
        # save activations
        o.save_activations(args.output, sep=args.sep)
    else:
        # detect the onsets
        o.detect(args.threshold, combine=args.combine, delay=args.delay,
                 smooth=args.smooth, pre_avg=args.pre_avg,
                 post_avg=args.post_avg, pre_max=args.pre_max,
                 post_max=args.post_max)
        # write the onsets to output
        o.write(args.output)

if __name__ == '__main__':
    main()
