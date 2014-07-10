#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) Sebastian Böck <sebastian.boeck@jku.at>

Redistribution in any form is not permitted!

"""

from madmom.audio.signal import Signal
from madmom.features.notes import NoteTranscription


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
    If invoked without any parameters, the software detects all notes in
    the given input (file) and writes them to the output (file).
    ''')
    # input/output options
    madmom.utils.params.io(p)
    # add note transcription parameters
    NoteTranscription.add_arguments(p)

    # add other argument groups
    madmom.utils.params.audio(p, fps=None, norm=False, online=None,
                              window=None)
    madmom.utils.params.midi(p, length=0.6, velocity=100)
    madmom.utils.params.save_load(p)
    # version
    p.add_argument('--version', action='version',
                   version='PianoTranscriptor.2014')
    # parse arguments
    args = p.parse_args()
    # set some defaults
    args.threads = min(len(args.nn_files), max(1, args.threads))
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
        n = NoteTranscription(args.input, **vars(args))
    else:
        # exit if no NN files are given
        if not args.nn_files:
            raise SystemExit('no NN model(s) given')

        # create a Signal object
        w = Signal(args.input, mono=True, norm=args.norm, att=args.att)
        # create an Note object with the activations
        n = NoteTranscription(w, **vars(args))

    # save note activations or detect the notes
    if args.save:
        # save activations
        n.save_activations(args.output)
    else:
        # write the notes to output
        if args.midi:
            import numpy as np
            import madmom.utils.midi as midi
            notes = np.asarray(n.detections)
            # expand the array
            notes = np.hstack((notes, np.ones_like(notes)))
            # set dummy offset
            notes[:, 2] = notes[:, 0] + args.note_length
            # set dummy velocity
            notes[:, 3] *= args.note_velocity
            m = midi.MIDIFile(notes)
            m.write(args.output)
        else:
            n.save_detections(args.output)

if __name__ == '__main__':
    main()
