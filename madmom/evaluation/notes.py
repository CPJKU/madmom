#!/usr/bin/env python
# encoding: utf-8
"""
This file contains note evaluation functionality.

"""

import numpy as np

from ..utils import suppress_warnings
from . import (evaluation_io, calc_errors, MultiClassEvaluation,
               MultiClassEvaluation as NoteSumEvaluation,
               MeanEvaluation as NoteMeanEvaluation)
from .onsets import onset_evaluation


@suppress_warnings
def load_notes(filename, delimiter=None):
    """
    Load a list of notes from file.

    :param filename:  name of the file
    :param delimiter: string used to separate values
    :return:          array with events

    Expected file format: onset_time, MIDI_note, [duration, [velocity]]

    """
    return np.loadtxt(filename, delimiter=delimiter, ndmin=2)


def remove_duplicate_rows(data):
    """
    Remove duplicate rows of a numpy array.

    :param data: 2D numpy array
    :return:     array with duplicate rows removed

    """
    # found at: http://pastebin.com/Ad6EgNjB
    order = np.lexsort(data.T)
    data = data[order]
    diff = np.diff(data, axis=0)
    unique = np.ones(len(data), 'bool')
    unique[1:] = (diff != 0).any(axis=1)
    return data[unique]


def note_evaluation(detections, annotations, window):
    """
    Determine the true/false positive/negative detections.

    :param detections:  array with detected notes
                        [[onset, MIDI note, duration, velocity]]
    :param annotations: array with annotated notes (same format as detections)
    :param window:      detection window [seconds]
    :return:            tuple of tp, fp, tn, fn numpy arrays

    tp: array with true positive detections
    fp: array with false positive detections
    tn: array with true negative detections (this one is empty!)
    fn: array with false negative detections

    Note: the true negative array is empty, because we are not interested in
          this class, since it is magnitudes as big as the note class.

    """
    # TODO: extend to also evaluate the duration and velocity of notes
    #       until then only use the first two columns (onsets + pitch)
    detections = remove_duplicate_rows(detections[:, :2])
    annotations = remove_duplicate_rows(annotations[:, :2])
    # init TP, FP, TN and FN lists
    tp = np.zeros((0, 2))
    fp = np.zeros((0, 2))
    tn = np.zeros((0, 2))
    fn = np.zeros((0, 2))
    # get a list of all notes
    notes = np.unique(np.concatenate((detections[:, 1],
                                      annotations[:, 1]))).tolist()
    # iterate over all notes
    for note in notes:
        # perform normal onset detection on ech note
        det = detections[detections[:, 1] == note]
        ann = annotations[annotations[:, 1] == note]
        tp_, fp_, _, fn_ = onset_evaluation(det[:, 0], ann[:, 0], window)
        # convert returned arrays to lists and append the detections and
        # annotations to the correct lists
        tp = np.vstack((tp, det[np.in1d(det[:, 0], tp_)]))
        fp = np.vstack((fp, det[np.in1d(det[:, 0], fp_)]))
        fn = np.vstack((fn, ann[np.in1d(ann[:, 0], fn_)]))
    # check calculation
    assert len(tp) + len(fp) == len(detections), 'bad TP / FP calculation'
    assert len(tp) + len(fn) == len(annotations), 'bad FN calculation'
    # return the arrays
    return tp, fp, tn, fn

# default evaluation values
WINDOW = 0.025


# for note evaluation with Precision, Recall, F-measure use the Evaluation
# class and just define the evaluation function
# TODO: extend to also report the measures without octave errors
class NoteEvaluation(MultiClassEvaluation):
    """
    Evaluation class for measuring Precision, Recall and F-measure of notes.

    """

    def __init__(self, detections, annotations, window=WINDOW, **kwargs):
        # evaluate
        numbers = note_evaluation(detections, annotations, window)
        # tp, fp, tn, fn = numbers
        super(NoteEvaluation, self).__init__(*numbers)
        # Note: just use the first column to calculate the errors, but append
        #       together with the note number so that we can calculate detailed
        #       error statistics
        self._errors = np.zeros((0, 2))
        for i, error in enumerate(calc_errors(self.tp[:, 0],
                                              annotations[:, 0])):
            self._errors = np.vstack((self._errors, [error, self.tp[i][1]]))


def add_parser(parser):
    """
    Add a note evaluation sub-parser to an existing parser.

    :param parser: existing argparse parser
    :return:       note evaluation sub-parser and evaluation parameter group

    """
    import argparse
    # add tempo evaluation sub-parser to the existing parser
    p = parser.add_parser(
        'notes', help='note evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
    This program evaluates pairs of files containing the note annotations and
    detections. Suffixes can be given to filter them from the list of files.

    Each line represents a note and must have the following format with values
    being separated by whitespace [brackets indicate optional values]:
    `onset_time MIDI_note [duration [velocity]]`

    Lines starting with # are treated as comments and are ignored.

    ''')
    # set defaults
    p.set_defaults(eval=NoteEvaluation,
                   sum_eval=NoteSumEvaluation,
                   mean_eval=NoteMeanEvaluation,
                   load_fn=load_notes)
    # file I/O
    evaluation_io(p, ann_suffix='.notes', det_suffix='.notes.txt')
    # evaluation parameters
    g = p.add_argument_group('note evaluation arguments')
    g.add_argument('-w', dest='window', action='store', type=float,
                   default=0.025,
                   help='evaluation window (+/- the given size) '
                        '[seconds, default=%(default)s]')
    g.add_argument('--delay', action='store', type=float, default=0.,
                   help='add given delay to all detections [seconds]')
    # return the sub-parser and evaluation argument group
    return g, g
