# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains note evaluation functionality.

"""

from __future__ import absolute_import, division, print_function

import warnings
import numpy as np

from . import (evaluation_io, MultiClassEvaluation, SumEvaluation,
               MeanEvaluation)
from .onsets import onset_evaluation, OnsetEvaluation
from ..io import load_notes


# default note evaluation values
WINDOW = 0.025


def remove_duplicate_notes(data):
    """
    Remove duplicate rows from the array.

    Parameters
    ----------
    data : numpy array
        Data.

    Returns
    -------
    numpy array
        Data array with duplicate rows removed.

    Notes
    -----
    This function removes only exact duplicates.

    """
    if data.size == 0:
        return data
    # found here: http://stackoverflow.com/questions/2828059/
    # find the unique rows
    order = np.ascontiguousarray(data).view(
        np.dtype((np.void, data.dtype.itemsize * data.shape[1])))
    unique = np.unique(order, return_index=True)[1]
    # only use the unique rows
    data = data[unique]
    # sort them by the first column and return them
    return data[data[:, 0].argsort()]


# note onset evaluation function
def note_onset_evaluation(detections, annotations, window=WINDOW):
    """
    Determine the true/false positive/negative note onset detections.

    Parameters
    ----------
    detections : numpy array
        Detected notes.
    annotations : numpy array
        Annotated ground truth notes.
    window : float, optional
        Evaluation window [seconds].

    Returns
    -------
    tp : numpy array, shape (num_tp, 2)
        True positive detections.
    fp : numpy array, shape (num_fp, 2)
        False positive detections.
    tn : numpy array, shape (0, 2)
        True negative detections (empty, see notes).
    fn : numpy array, shape (num_fn, 2)
        False negative detections.
    errors : numpy array, shape (num_tp, 2)
        Errors of the true positive detections wrt. the annotations.

    Notes
    -----
    The expected note row format is:

    'note_time' 'MIDI_note' ['duration' ['MIDI_velocity']]

    The returned true negative array is empty, because we are not interested
    in this class, since it is magnitudes bigger than true positives array.

    """
    # make sure the arrays have the correct types and dimensions
    detections = np.asarray(detections, dtype=np.float)
    annotations = np.asarray(annotations, dtype=np.float)
    # check dimensions
    if detections.ndim != 2 or annotations.ndim != 2:
        raise ValueError('detections and annotations must be 2D arrays')

    # init TP, FP, TN and FN lists
    tp = np.zeros((0, 2))
    fp = np.zeros((0, 2))
    tn = np.zeros((0, 2))  # this will not be altered
    fn = np.zeros((0, 2))
    errors = np.zeros((0, 2))
    # if neither detections nor annotations are given
    if detections.size == 0 and annotations.size == 0:
        # return the arrays as is
        return tp, fp, tn, fn, errors
    # if only detections are given
    elif annotations.size == 0:
        # all detections are FP
        return tp, detections, tn, fn, errors
    # if only annotations are given
    elif detections.size == 0:
        # all annotations are FN
        return tp, tp, tn, annotations, errors

    # TODO: extend to also evaluate the duration and velocity of notes
    # for onset evaluation use only the onset time and midi note number
    detections = detections[:, :2]
    annotations = annotations[:, :2]

    # get a list of all notes detected / annotated
    notes = np.unique(np.concatenate((detections[:, 1],
                                      annotations[:, 1]))).tolist()
    # iterate over all notes
    for note in notes:
        # perform normal onset detection on each note
        det = detections[detections[:, 1] == note]
        ann = annotations[annotations[:, 1] == note]
        tp_, fp_, _, fn_, err_ = onset_evaluation(det[:, 0], ann[:, 0], window)
        # convert returned arrays to lists and append the detections and
        # annotations to the correct lists
        tp = np.vstack((tp, det[np.in1d(det[:, 0], tp_)]))
        fp = np.vstack((fp, det[np.in1d(det[:, 0], fp_)]))
        fn = np.vstack((fn, ann[np.in1d(ann[:, 0], fn_)]))
        # append the note number to the errors
        err_ = np.vstack((np.array(err_),
                          np.repeat(np.asarray([note]), len(err_)))).T
        errors = np.vstack((errors, err_))
    # check calculations
    if len(tp) + len(fp) != len(detections):
        raise AssertionError('bad TP / FP calculation')
    if len(tp) + len(fn) != len(annotations):
        raise AssertionError('bad FN calculation')
    if len(tp) != len(errors):
        raise AssertionError('bad errors calculation')
    # sort the arrays
    # Note: The errors must have the same sorting order as the TPs, so they
    #       must be done first (before the TPs get sorted)
    errors = errors[tp[:, 0].argsort()]
    tp = tp[tp[:, 0].argsort()]
    fp = fp[fp[:, 0].argsort()]
    fn = fn[fn[:, 0].argsort()]
    # return the arrays
    return tp, fp, tn, fn, errors


# for note evaluation with Precision, Recall, F-measure use the Evaluation
# class and just define the evaluation function
# TODO: extend to also report the measures without octave errors
class NoteEvaluation(MultiClassEvaluation):
    """
    Evaluation class for measuring Precision, Recall and F-measure of notes.

    Parameters
    ----------
    detections : str, list or numpy array
        Detected notes.
    annotations : str, list or numpy array
        Annotated ground truth notes.
    window : float, optional
        F-measure evaluation window [seconds]
    delay : float, optional
        Delay the detections `delay` seconds for evaluation.

    """

    def __init__(self, detections, annotations, window=WINDOW, delay=0,
                 **kwargs):
        # convert to numpy array
        detections = np.array(detections, dtype=np.float, ndmin=2)
        annotations = np.array(annotations, dtype=np.float, ndmin=2)
        # shift the detections if needed
        if delay != 0:
            detections[:, 0] += delay
        # evaluate
        numbers = note_onset_evaluation(detections, annotations, window)
        tp, fp, tn, fn, errors = numbers
        super(NoteEvaluation, self).__init__(tp, fp, tn, fn, **kwargs)
        self.errors = errors
        # save them for the individual note evaluation
        self.detections = detections
        self.annotations = annotations
        self.window = window

    @property
    def mean_error(self):
        """Mean of the errors."""
        warnings.warn('mean_error is given for all notes, this will change!')
        if len(self.errors) == 0:
            return np.nan
        return np.mean(self.errors[:, 0])

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        warnings.warn('std_error is given for all notes, this will change!')
        if len(self.errors) == 0:
            return np.nan
        return np.std(self.errors[:, 0])

    def tostring(self, notes=False, **kwargs):
        """

        Parameters
        ----------
        notes : bool, optional
            Display detailed output for all individual notes.

        Returns
        -------
        str
            Evaluation metrics formatted as a human readable string.

        """
        ret = ''
        if self.name is not None:
            ret += '%s\n  ' % self.name
        # add statistics for the individual note
        if notes:
            # determine which notes are present
            notes = []
            if self.tp.any():
                notes = np.append(notes, np.unique(self.tp[:, 1]))
            if self.fp.any():
                notes = np.append(notes, np.unique(self.fp[:, 1]))
            if self.tn.any():
                notes = np.append(notes, np.unique(self.tn[:, 1]))
            if self.fn.any():
                notes = np.append(notes, np.unique(self.fn[:, 1]))
            # evaluate them individually
            for note in sorted(np.unique(notes)):
                # detections and annotations for this note (only onset times)
                det = self.detections[self.detections[:, 1] == note][:, 0]
                ann = self.annotations[self.annotations[:, 1] == note][:, 0]
                name = 'MIDI note %s' % note
                e = OnsetEvaluation(det, ann, self.window, name=name)
                # append to the output string
                ret += '  %s\n' % e.tostring(notes=False)
        # normal formatting
        ret += 'Notes: %5d TP: %5d FP: %4d FN: %4d ' \
               'Precision: %.3f Recall: %.3f F-measure: %.3f ' \
               'Acc: %.3f mean: %5.1f ms std: %5.1f ms' % \
               (self.num_annotations, self.num_tp, self.num_fp, self.num_fn,
                self.precision, self.recall, self.fmeasure, self.accuracy,
                self.mean_error * 1000., self.std_error * 1000.)
        # return
        return ret


class NoteSumEvaluation(SumEvaluation, NoteEvaluation):
    """
    Class for summing note evaluations.

    """

    @property
    def errors(self):
        """Errors of the true positive detections wrt. the ground truth."""
        if not self.eval_objects:
            # return empty array
            return np.zeros((0, 2))
        return np.concatenate([e.errors for e in self.eval_objects])


class NoteMeanEvaluation(MeanEvaluation, NoteSumEvaluation):
    """
    Class for averaging note evaluations.

    """

    @property
    def mean_error(self):
        """Mean of the errors."""
        warnings.warn('mean_error is given for all notes, this will change!')
        return np.nanmean([e.mean_error for e in self.eval_objects])

    @property
    def std_error(self):
        """Standard deviation of the errors."""
        warnings.warn('std_error is given for all notes, this will change!')
        return np.nanmean([e.std_error for e in self.eval_objects])

    def tostring(self, **kwargs):
        """
        Format the evaluation metrics as a human readable string.

        Returns
        -------
        str
            Evaluation metrics formatted as a human readable string.

        """
        # format with floats instead of integers
        ret = ''
        if self.name is not None:
            ret += '%s\n  ' % self.name
        ret += 'Notes: %5.2f TP: %5.2f FP: %5.2f FN: %5.2f ' \
               'Precision: %.3f Recall: %.3f F-measure: %.3f ' \
               'Acc: %.3f mean: %5.1f ms std: %5.1f ms' % \
               (self.num_annotations, self.num_tp, self.num_fp, self.num_fn,
                self.precision, self.recall, self.fmeasure, self.accuracy,
                self.mean_error * 1000., self.std_error * 1000.)
        return ret


def add_parser(parser):
    """
    Add a note evaluation sub-parser to an existing parser.

    Parameters
    ----------
    parser : argparse parser instance
        Existing argparse parser object.

    Returns
    -------
    sub_parser : argparse sub-parser instance
        Note evaluation sub-parser.
    parser_group : argparse argument group
        Note evaluation argument group.

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
    p.set_defaults(eval=NoteEvaluation, sum_eval=NoteSumEvaluation,
                   mean_eval=NoteMeanEvaluation, load_fn=load_notes)
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
    return p, g
