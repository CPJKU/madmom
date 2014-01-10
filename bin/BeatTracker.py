#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2012-2013 Sebastian Böck <sebastian.boeck@jku.at>

Redistribution in any form is not permitted!
"""

import os
import glob
import numpy as np

from madmom.audio.wav import Wav
from madmom.audio.spectrogram import LogFiltSpec
from madmom.features.beats import Beat
from madmom.ml.rnn import RecurrentNeuralNetwork

# set the path to saved neural networks and generate lists of NN files
NN_PATH = '%s/../madmom/ml/data' % (os.path.dirname(__file__))
NN_FILES = glob.glob("%s/beats_blstm*npz" % NN_PATH)

# TODO: this information should be included/extracted in/from the NN files
FPS = 100
BANDS_PER_OCTAVE = 3
MUL = 1
ADD = 1
FMIN = 30
FMAX = 17000
RATIO = 0.5
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
    If invoked without any parameters, the software detects all beats in the
    given input (file) and writes them to the output (file).
    ''')
    # mirex options
    madmom.utils.params.mirex(p)
    # add other argument groups
    p.add_argument('--nn_files', action='append', type=str, default=NN_FILES,
                   help='use these pre-trained neural networks '
                        '(multiple files can be given, one per argument)')
    madmom.utils.params.audio(p, fps=None, norm=False, online=None,
                              window=None)
    b = madmom.utils.params.beat(p)
    b.add_argument('--look_ahead', action='store', type=float, default=4,
                   help='look ahead N seconds [default=4]')
    madmom.utils.params.io(p)
    # version
    p.add_argument('--version', action='version', version='BeatTracker.2013')
    # parse arguments
    args = p.parse_args()
    # set some defaults
    args.fps = FPS
    args.online = False
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """BeatTracker.2013"""

    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # load activations
        b = Beat(args.input, args.fps, args.online, args.sep)
    else:
        # exit if no NN files are given
        if not args.nn_files:
            raise SystemExit('no NN models given')
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
        data = np.hstack((data, s.spec, s.pos_diff))
        # test the data against all saved neural nets
        act = None
        for nn_file in args.nn_files:
            if act is None:
                act = RecurrentNeuralNetwork(nn_file).activate(data)
            else:
                act += RecurrentNeuralNetwork(nn_file).activate(data)
        # normalize activations
        if len(args.nn_files) > 1:
            act /= len(args.nn_files)
        # create an Beat object with the activations
        b = Beat(act.ravel(), args.fps, args.online)

    # save beat activations or detect beats
    if args.save:
        # save activations
        b.save_activations(args.output, sep=args.sep)
    else:
        # track the beats
        b.track(args.threshold, smooth=args.smooth, look_ahead=args.look_ahead,
                min_bpm=args.min_bpm, max_bpm=args.max_bpm)
        # write the beats to output
        b.write(args.output)

if __name__ == '__main__':
    main()
