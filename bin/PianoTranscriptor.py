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
from madmom.features.notes import NoteTranscription
from madmom.ml.rnn import RecurrentNeuralNetwork

# set the path to saved neural networks and generate lists of NN files
NN_PATH = '%s/../madmom/ml/data' % (os.path.dirname(__file__))
NN_FILES = glob.glob("%s/notes*npz" % NN_PATH)

# TODO: this information should be included/extracted in/from the NN files
FPS = 100
BANDS_PER_OCTAVE = 12
MUL = 5
ADD = 1
FMIN = 27.5
FMAX = 18000
RATIO = 0.5
NORM_FILTERS = True


def parser():
    import argparse
    import madmom.utils.params

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    If invoked without any parameters, the software detects all notes in
    the given input (file) and writes them to the output (file).
    ''')
    # mirex options
    madmom.utils.params.mirex(p)
    # add other argument groups
    p.add_argument('--nn_files', action='append', type=str, default=NN_FILES,
                   help='use these pre-trained neural networks '
                        '(multiple files can be given, one per argument)')
    madmom.utils.params.audio(p, norm=False)
    madmom.utils.params.onset(p, threshold=0.35, combine=0.03, smooth=0.07,
                             pre_avg=0, post_avg=0, pre_max=1. / FPS,
                             post_max=1. / FPS)
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
    """PianoTranscriptor.2012"""

    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # load activations
        n = NoteTranscription(args.input, args.fps, args.online, args.sep)
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
        #np.save('/Users/sb/data/tmp/tmp', data)
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

        # create an Note object with the activations
        n = NoteTranscription(act, args.fps, args.online)

    # save note activations or detect beats
    if args.save:
        # save activations
        n.save_activations(args.output, sep=args.sep)
    else:
        # track the beats
        n.detect(args.threshold, combine=args.combine, delay=args.delay,
                 smooth=args.smooth, pre_avg=args.pre_avg,
                 post_avg=args.post_avg, pre_max=args.pre_max,
                 post_max=args.post_max)
        # write the beats to output
        n.write(args.output)

if __name__ == '__main__':
    main()
