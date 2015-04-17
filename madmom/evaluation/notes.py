#!/usr/bin/env python
# encoding: utf-8
"""
This file contains note evaluation functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np


from . import calc_errors, Evaluation, MultiClassEvaluation, MeanEvaluation
from .onsets import onset_evaluation


def load_notes(filename, delimiter=None):
    """
    Load a list of notes from file.

    :param filename:  name of the file
    :param delimiter: string used to separate values
    :return:          array with events

    Expected file format: onset_time, MIDI_note, [duration, [velocity]]

    """
    return np.loadtxt(filename, delimiter=delimiter)


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
    tp = []
    fp = []
    tn = []
    fn = []
    # get a list of all notes
    notes = np.unique(np.concatenate((detections[:, 1],
                                      annotations[:, 1]))).tolist()
    # iterate over all notes
    for note in notes:
        # perform normal onset detection on ech note
        det = detections[detections[:, 1] == note]
        tar = annotations[annotations[:, 1] == note]
        tp_, fp_, _, fn_ = onset_evaluation(det[:, 0], tar[:, 0], window)
        # convert returned arrays to lists and append the detections and
        # annotations to the correct lists
        tp.extend(det[np.in1d(det[:, 0], tp_)].tolist())
        fp.extend(det[np.in1d(det[:, 0], fp_)].tolist())
        fn.extend(tar[np.in1d(tar[:, 0], fn_)].tolist())
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
class NoteEvaluation(Evaluation):
    """
    Evaluation class for measuring Precision, Recall and F-measure of notes.

    """
    def __init__(self, detections, annotations, window=WINDOW):
        super(NoteEvaluation, self).__init__()
        self.detections = detections
        self.annotations = annotations
        # evaluate
        numbers = note_evaluation(detections, annotations, window)
        # tp, fp, tn, fn = numbers
        super(NoteEvaluation, self).__init__(*numbers)

    @property
    def errors(self):
        """
        Absolute errors of all true positive detections relative to the closest
        annotations.

        """
        if self._errors is None:
            if self.num_tp == 0:
                # FIXME: what is the error in case of no TPs
                self._errors = []
            else:
                # just use the first column to calculate the errors
                # FIXME: do this for all notes individually
                self._errors = calc_errors(self.tp[:, 0],
                                           self.annotations[:, 0]).tolist()
        return self._errors


def parser():
    """
    Create a parser and parse the arguments.

    :return: the parsed arguments

    """
    import argparse
    from . import evaluation_io
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    This script evaluates pairs of files containing the note annotations and
    detections. Suffixes can be given to filter them from the list of files.

    Each line represents a note and must have the following format with values
    being separated by tabs [brackets indicate optional values]:
    `onset_time MIDI_note [duration [velocity]]`

    Lines starting with # are treated as comments and are ignored.

    """)
    # files used for evaluation
    evaluation_io(p, ann_suffix='.notes', det_suffix='.notes.txt')
    # parameters for evaluation
    g = p.add_argument_group('evaluation arguments')
    g.add_argument('-w', dest='window', action='store', type=float,
                   default=0.025,
                   help='evaluation window (+/- the given size) '
                        '[seconds, default=%(default)s]')
    g.add_argument('--delay', action='store', type=float, default=0.,
                   help='add given delay to all detections [seconds]')
    # parse the arguments
    args = p.parse_args()
    # print the args
    if args.verbose >= 2:
        print args
    # return
    return args


def main():
    """
    Simple note evaluation.

    """
    from madmom.utils import search_files, match_file

    # parse arguments
    args = parser()

    # get detection and annotation files
    if args.det_dir is None:
        args.det_dir = args.files
    if args.ann_dir is None:
        args.ann_dir = args.files
    det_files = search_files(args.det_dir, args.det_suffix)
    ann_files = search_files(args.ann_dir, args.ann_suffix)
    # quit if no files are found
    if len(ann_files) == 0:
        print "no files to evaluate. exiting."
        exit()

    # sum and mean evaluation for all files
    sum_eval = MultiClassEvaluation()
    mean_eval = MeanEvaluation()
    # evaluate all files
    for ann_file in ann_files:
        # load the annotations
        annotations = load_notes(ann_file)
        # get the matching detection files
        matches = match_file(ann_file, det_files,
                             args.ann_suffix, args.det_suffix)
        if len(matches) > 1:
            # exit if multiple detections were found
            raise SystemExit("multiple detections for %s found." % ann_file)
        elif len(matches) == 0:
            # print a warning if no detections were found
            import warnings
            warnings.warn(" can't find detections for %s." % ann_file)
            # but continue and assume no detections
            detections = np.zeros((0, 0))
        else:
            # load the detections
            detections = load_notes(matches[0])
        # shift the detections if needed
        if args.delay != 0:
            detections[:, 0] += args.delay
        # evaluate
        e = NoteEvaluation(detections, annotations, window=args.window)
        # print stats for the file
        if args.verbose:
            print ann_file
            if args.verbose >= 2:
                print e.print_errors('  ', args.tex, True)
            else:
                print e.print_errors('  ', args.tex, False)
        # add this file's evaluation to the global evaluation
        sum_eval += e
        mean_eval.append(e)
    # print summary
    print sum_eval.print_errors('sum for %i file(s):\n  ' % len(mean_eval))
    print mean_eval.print_errors('mean for %i file(s):\n  ' % len(mean_eval))

if __name__ == '__main__':
    main()
