#!/usr/bin/env python
# encoding: utf-8
"""
This file contains note transcription related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import glob
import numpy as np

from madmom import MODELS_PATH
from madmom.utils import open

from . import Activations, RNNEventDetection
from .onsets import peak_picking


def load_notes(filename):
    """
    Load the target notes from a file.

    :param filename: input file name or file handle
    :return:         numpy array with notes

    """
    with open(filename, 'rb') as f:
        return np.loadtxt(f)


class RNNNoteTranscription(RNNEventDetection):
    """
    Note transcription with RNNs.

    "Polyphonic Piano Note Transcription with Recurrent Neural Networks"
    Sebastian Böck and Markus Schedl.
    Proceedings of the 37th International Conference on Acoustics, Speech and
    Signal Processing (ICASSP), 2012.

    """
    # define NN files
    NN_FILES = glob.glob("%s/notes_brnn*npz" % MODELS_PATH)

    # TODO: this information should be included/extracted in/from the NN files
    FPS = 100
    BANDS_PER_OCTAVE = 12
    MUL = 5
    ADD = 1

    # TODO: These do not seem to be used anywhere
    FMIN = 27.5
    FMAX = 18000
    RATIO = 0.5
    NORM_FILTERS = True

    # default values for note peak-picking
    THRESHOLD = 0.35
    SMOOTH = 0.09
    PRE_AVG = 0
    POST_AVG = 0
    PRE_MAX = 1. / FPS
    POST_MAX = 1. / FPS
    # default values for note reporting
    COMBINE = 0.05
    DELAY = 0

    def pre_process(self):
        """
        Pre-process the signal to obtain a data representation suitable for RNN
        processing.

        :return: pre-processed data

        """
        spr = super(RNNNoteTranscription, self)
        spr.pre_process(frame_sizes=[1024, 2048, 4096], bands_per_octave=12,
                        mul=5, ratio=0.5)
        # return data
        return self._data

    def process(self):
        """
        Test the data with the defined RNNs.

        :return: note activations

        """
        # process the data
        super(RNNNoteTranscription, self).process()
        # reshape the activations
        self._activations = self._activations.reshape(-1, 88)
        # and return them
        return self._activations

    def detect(self, threshold=THRESHOLD, smooth=SMOOTH, combine=COMBINE,
               delay=DELAY):
        """
        Detect the notes with the given peak-picking parameters.

        :param threshold: threshold for note detection
        :param smooth:    smooth activations over N seconds
        :param combine:   combine note onsets within N seconds
        :param delay:     report note onsets N seconds delayed
        :return:          detected notes

        """
        # convert timing information to frames
        smooth = int(round(self.fps * smooth))
        # detect notes
        detections = peak_picking(self.activations, threshold, smooth)
        # convert to seconds / MIDI note numbers
        onsets = detections[0].astype(np.float) / self.fps
        midi_notes = detections[1] + 21
        # shift if necessary
        if delay != 0:
            onsets += delay
        # combine multiple notes
        if combine > 0:
            detections = []
            # iterate over each detected note separately
            for note in np.unique(midi_notes):
                # get all note detections
                note_onsets = onsets[midi_notes == note]
                # always use the first note
                detections.append((note_onsets[0], note))
                # filter all notes which occur within `combine` seconds
                combined_note_onsets = note_onsets[1:][np.diff(note_onsets) >
                                                       combine]
                # zip them with the MIDI note number and add them to the list
                detections.extend(zip(combined_note_onsets,
                                      [note] * len(combined_note_onsets)))
        else:
            # just zip all detected notes
            detections = zip(onsets, midi_notes)
        # sort the detections and save as numpy array
        self._detections = np.asarray(sorted(detections))
        # also return them
        return self._detections

    def write(self, filename, sep='\t'):
        """
        Write the detected notes to a file.

        :param filename: output file name or file handle
        :param sep:    separator for the fields [default='\t']

        """
        from madmom.utils import open
        # write the detected notes to the output
        with open(filename, 'wb') as f:
            for note in self.detections:
                f.write(sep.join([str(x) for x in note]) + '\n')

    @classmethod
    def add_arguments(cls, parser, nn_files=NN_FILES, threshold=THRESHOLD,
                      smooth=SMOOTH, combine=COMBINE, delay=DELAY,
                      pre_avg=PRE_AVG, post_avg=POST_AVG,
                      pre_max=PRE_MAX, post_max=POST_MAX):
        """
        Add note transcription related arguments to an existing parser object.

        :param parser:    existing argparse parser object
        :param nn_files:  list with files of NN models
        :param threshold: threshold for peak-picking
        :param smooth:    smooth the note activations over N seconds
        :param combine:   only report one note within N seconds and pitch
        :param delay:     report notes N seconds delayed
        :param pre_avg:   use N seconds past information for moving average
        :param post_avg:  use N seconds future information for moving average
        :param pre_max:   use N seconds past information for moving maximum
        :param post_max:  use N seconds future information for moving maximum
        :return:          note argument parser group object

        """
        # add Activations parser
        Activations.add_arguments(parser)
        # add RNNEventDetection arguments
        RNNEventDetection.add_arguments(parser, nn_files=nn_files)
        # add note transcription related options to the existing parser
        g = parser.add_argument_group('note transcription arguments')
        g.add_argument('-t', dest='threshold', action='store', type=float,
                       default=threshold, help='detection threshold '
                       '[default=%(default)s]')
        g.add_argument('--smooth', action='store', type=float, default=smooth,
                       help='smooth the note activations over N seconds '
                       '[default=%(default).2f]')
        g.add_argument('--combine', action='store', type=float,
                       default=combine, help='combine notes within N seconds '
                       '(per pitch) [default=%(default).2f]')
        g.add_argument('--pre_avg', action='store', type=float,
                       default=pre_avg, help='build average over N previous '
                       'seconds [default=%(default).2f]')
        g.add_argument('--post_avg', action='store', type=float,
                       default=post_avg, help='build average over N following '
                       'seconds [default=%(default).2f]')
        g.add_argument('--pre_max', action='store', type=float,
                       default=pre_max, help='search maximum over N previous '
                       'seconds [default=%(default).2f]')
        g.add_argument('--post_max', action='store', type=float,
                       default=post_max, help='search maximum over N '
                       'following seconds [default=%(default).2f]')
        g.add_argument('--delay', action='store', type=float, default=delay,
                       help='report the notes N seconds delayed '
                       '[default=%(default)i]')
        # return the argument group so it can be modified if needed
        return g
