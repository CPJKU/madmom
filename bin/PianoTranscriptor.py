#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) Sebastian Böck <sebastian.boeck@jku.at>

Redistribution in any form is not permitted!

"""

from madmom.audio.signal import Signal
from madmom.features.notes import RNNNoteTranscription
import madmom.utils.midi as midi


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
    If invoked without any parameters, the software detects all notes in
    the given input (file) and writes them to the output (file).

    "Polyphonic Piano Note Transcription with Recurrent Neural Networks"
    Sebastian Böck and Markus Schedl.
    Proceedings of the 37th International Conference on Acoustics, Speech and
    Signal Processing (ICASSP), 2012.

    ''')
    # input/output options
    madmom.utils.io_arguments(p)
    # signal arguments
    Signal.add_arguments(p, norm=False)
    # rnn note transcription arguments
    RNNNoteTranscription.add_arguments(p)
    # midi arguments
    midi.MIDIFile.add_arguments(p, length=0.6, velocity=100)
    # version
    p.add_argument('--version', action='version',
                   version='PianoTranscriptor.2014')
    # parse arguments
    args = p.parse_args()
    # print arguments
    if args.verbose:
        print args
    # return
    return args


def main():
    """PianoTranscriptor.2014"""

    # parse arguments
    args = parser()

    # load or create onset activations
    if args.load:
        # load activations
        n = RNNNoteTranscription.from_activations(args.input, fps=100)
    else:
        # exit if no NN files are given
        if not args.nn_files:
            raise SystemExit('no NN model(s) given')

        # create a Signal object
        s = Signal(args.input, mono=True, norm=args.norm, att=args.att)
        # create a RNNBeatDetection object from the signal and given NN files
        n = RNNNoteTranscription(s, nn_files=args.nn_files,
                                 num_threads=args.num_threads)

    # save note activations or detect the notes
    if args.save:
        # save activations
        n.activations.save(args.output)
    else:
        # write the notes to output
        if args.midi:
            import numpy as np
            # expand the array to have a length and velocity
            notes = np.hstack((n.detections, np.ones_like(n.detections)))
            # set dummy offset
            notes[:, 2] = notes[:, 0] + args.note_length
            # set dummy velocity
            notes[:, 3] *= args.note_velocity
            m = midi.MIDIFile(notes)
            m.write(args.output)
        else:
            n.write(args.output)

if __name__ == '__main__':
    main()
