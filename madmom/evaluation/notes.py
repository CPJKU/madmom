#!/usr/bin/env python
# encoding: utf-8
"""
This file contains note evaluation functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np

from .simple import Evaluation, SumEvaluation, MeanEvaluation


def load_notes(filename, delimiter=None):
    """
    Load a list of notes from file.

    :param filename:  name of the file
    :param delimiter: string used to separate values [default=any whitespace]
    :return:          array with events

    Expected file format: onset_time, MIDI_note, [length, [velocity]]

    """
    return np.loadtxt(filename, delimiter=delimiter)


# evaluation function for note detection
def count_errors(detections, targets, window):
    """
    Count the true and false detections of the given detections and targets.

    :param detections: 2D array of events [[seconds, label]]
    :param targets: 2D array of events [[seconds, label]]
    :param window: detection window [seconds]
    :return: tuple of tp, fp, tn, fn numpy arrays

    tp: 2D numpy array with true positive detections
    fp: 2D numpy array with false positive detections
    tn: 2D numpy array with true negative detections (this one is empty!)
    fn: 2D numpy array with false negative detections

    """
    from ..evaluation.helpers import calc_absolute_errors
    # no detections
    if detections.size == 0:
        # all targets are FNs
        return np.empty(0), np.empty(0), np.empty(0), targets
    # calc the absolute errors of detections wrt. targets
    errors = calc_absolute_errors(detections, targets)
    # true positive detections
    tp = detections[errors <= window]
    # the remaining detections are FP
    fp = detections[errors > window]
    # calc the absolute errors of detections wrt. targets
    errors = np.asarray(calc_absolute_errors(targets, detections))
    fn = targets[errors > window]
    # return the arrays
    return tp, fp, np.empty(0), fn


# default evaluation values
WINDOW = 0.025


# for note evaluation with Presicion, Recall, F-measure use the Evaluation
# class and just define the evaluation function
# TODO: extend to also report the measures without octave errors
class NoteEvaluation(Evaluation):
    """
    Simple class for measuring Precision, Recall and F-measure of notes.

    """
    def __init__(self, detections, targets, window=WINDOW):
        super(NoteEvaluation, self).__init__(detections, targets, count_errors, window=window)


class SumNoteEvaluation(SumEvaluation):
    pass


class MeanNoteEvaluation(MeanEvaluation):
    pass


def parser():
    import argparse

    # define parser
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    If invoked without any parameters the script evaluates pairs of files
    with the targets (.onsets) and detection (.onsets.txt) as simple text
    files with one onset timestamp per line according to the rules given in

    "Evaluating the Online Capabilities of Onset Detection Methods"
    by Sebastian Böck, Florian Krebs and Markus Schedl
    in Proceedings of the 13th International Society for
    Music Information Retrieval Conference (ISMIR 2012)

    """)
    p.add_argument('files', metavar='files', nargs='+', help='path or files to be evaluated (list of files being filtered according to -d and -t arguments)')
    # extensions used for evaluation
    p.add_argument('-d', dest='detections', action='store', default='.onsets.txt', help='extensions of the detections [default: .onsets.txt]')
    p.add_argument('-t', dest='targets', action='store', default='.onsets', help='extensions of the targets [default: .onsets]')
    # parameters for evaluation
    p.add_argument('-w', dest='window', action='store', default=0.05, type=float, help='evaluation window [seconds, default=0.05]')
    p.add_argument('--delay', action='store', default=0., type=float, help='add given delay to all detections [seconds]')
    p.add_argument('--tex', action='store_true', help='format errors for use in .tex files')
    # verbose
    p.add_argument('-v', dest='verbose', action='count', help='increase verbosity level')
    # parse the arguments
    args = p.parse_args()
    # print the args
    if args.verbose >= 2:
        print args
    # return
    return args


def main():
    from ..utils.helpers import files

    # parse the arguments
    args = parser()

    # TODO: find a better way to determine the corresponding detection/target
    # files from a given list/path of files

    # filter target files
    tar_files = files(args.files, args.targets)
    # filter detection files
    det_files = files(args.files, args.detections)
    # must be the same number FIXME: find better solution which checks the names
    assert len(tar_files) == len(det_files), "different number of targets (%i) and detections (%i)" % (len(tar_files), len(det_files))

    # sum counter for all files
    sum_counter = SumNoteEvaluation()
    mean_counter = MeanNoteEvaluation()
    # evaluate all files
    for i in range(len(det_files)):
        detections = load_notes(det_files[i])
        targets = load_notes(tar_files[i])
        # shift the detections if needed
        if args.delay != 0:
            detections += args.delay
        # evaluate the onsets
        oe = NoteEvaluation(detections, targets, args.window)
        # print stats for each file
        if args.verbose:
            print det_files[i]
            oe.print_errors(args.tex)
        # add to sum counter
        sum_counter += oe
        mean_counter += oe
    # print summary
    print 'sum for %i files; detection window %.1f ms (+- %.1f ms)' % (len(det_files), args.window * 2000, args.window * 1000)
    sum_counter.print_errors(args.tex)
    print 'mean for %i files; detection window %.1f ms (+- %.1f ms)' % (len(det_files), args.window * 2000, args.window * 1000)
    mean_counter.print_errors(args.tex)

if __name__ == '__main__':
    main()
