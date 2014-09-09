#!/usr/bin/env python
# encoding: utf-8
"""
This file contains onset evaluation functionality.

It is described in:

"Evaluating the Online Capabilities of Onset Detection Methods"
Sebastian Böck, Florian Krebs and Markus Schedl
Proceedings of the 13th International Society for Music Information Retrieval
Conference (ISMIR), 2012.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np

from . import calc_errors, Evaluation, SumEvaluation, MeanEvaluation


# evaluation function for onset detection
def onset_evaluation(detections, annotations, window):
    """
    Determine the true/false positive/negative detections.

    :param detections:  array with detected onsets [seconds]
    :param annotations: array with annotated onsets [seconds]
    :param window:      detection window [seconds]
    :return:            tuple of tp, fp, tn, fn numpy arrays

    tp: array with true positive detections
    fp: array with false positive detections
    tn: array with true negative detections (this one is empty!)
    fn: array with false negative detections

    Note: The true negative array is empty, because we are not interested in
          this class, since it is ~20 times as big as the onset class.

    """
    # convert numpy array to lists if needed
    if isinstance(detections, np.ndarray):
        detections = detections.tolist()
    if isinstance(annotations, np.ndarray):
        annotations = annotations.tolist()
    # sort the detections and annotations
    det = sorted(detections)
    ann = sorted(annotations)
    # cache variables
    det_length = len(detections)
    ann_length = len(annotations)
    det_index = 0
    ann_index = 0
    # init TP, FP, TN and FN lists
    tp = []
    fp = []
    tn = []
    fn = []
    while det_index < det_length and ann_index < ann_length:
        # fetch the first detection
        d = det[det_index]
        # fetch the first annotation
        t = ann[ann_index]
        # compare them
        if abs(d - t) <= window:
            # TP detection
            tp.append(d)
            # increase the detection and annotation index
            det_index += 1
            ann_index += 1
        elif d < t:
            # FP detection
            fp.append(d)
            # increase the detection index
            det_index += 1
            # do not increase the annotation index
        elif d > t:
            # we missed a annotation: FN
            fn.append(t)
            # do not increase the detection index
            # increase the annotation index
            ann_index += 1
    # the remaining detections are FP
    fp.extend(det[det_index:])
    # the remaining annotations are FN
    fn.extend(ann[ann_index:])
    # check calculation
    assert len(tp) + len(fp) == len(detections), 'bad TP / FP calculation'
    assert len(tp) + len(fn) == len(annotations), 'bad FN calculation'
    # return the arrays
    return tp, fp, tn, fn


# default values
WINDOW = 0.025
COMBINE = 0.03


# for onset evaluation with Precision, Recall, F-measure use the Evaluation
# class and just define the evaluation and error functions
class OnsetEvaluation(Evaluation):
    """
    Simple class for measuring Precision, Recall and F-measure.

    """
    def __init__(self, detections, annotations, window=WINDOW):
        # convert the detections and annotations
        detections = np.asarray(sorted(detections), dtype=np.float)
        annotations = np.asarray(sorted(annotations), dtype=np.float)
        # evaluate
        numbers = onset_evaluation(detections, annotations, window)
        # tp, fp, tn, fn = numbers
        super(OnsetEvaluation, self).__init__(*numbers)
        # calculate errors
        self._errors = calc_errors(self.tp, annotations).tolist()


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
    This script evaluates pairs of files containing the onset annotations and
    detections. Suffixes can be given to filter them from the list of files.

    Each line represents an onset and must have the following format:
    `onset_time`.

    Lines starting with # are treated as comments and are ignored.

    """)
    # files used for evaluation
    evaluation_io(p, ann_suffix='.onsets', det_suffix='.onsets.txt')
    # parameters for evaluation
    g = p.add_argument_group('evaluation arguments')
    g.add_argument('-w', dest='window', action='store', type=float,
                   default=WINDOW,
                   help='evaluation window (+/- the given size) '
                        '[seconds, default=%(default).3f]')
    g.add_argument('-c', dest='combine', action='store', type=float,
                   default=COMBINE,
                   help='combine annotation events within this range '
                        '[seconds, default=%(default).3f]')
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
    Simple onset evaluation.

    """
    from madmom.utils import files, match_file, load_events, combine_events

    # parse arguments
    args = parser()

    # get detection and annotation files
    det_files = files(args.files, args.det_suffix)
    ann_files = files(args.files, args.ann_suffix)
    # quit if no files are found
    if len(det_files) == 0:
        print "no files to evaluate. exiting."
        exit()

    # sum and mean evaluation for all files
    sum_eval = SumEvaluation()
    mean_eval = MeanEvaluation()
    # evaluate all files
    for det_file in det_files:
        # load the detections
        detections = load_events(det_file)
        # shift the detections if needed
        if args.delay != 0:
            detections += args.delay
        # get the matching annotation files
        matches = match_file(det_file, ann_files, args.det_suffix,
                             args.ann_suffix)
        # quit if any file does not have a matching annotation file
        if len(matches) == 0:
            print " can't find an annotation file for %s. exiting." % det_file
            exit()
        # do a mean evaluation with all matched annotation files
        me = MeanEvaluation()
        for ann_file in matches:
            # load the annotations
            annotations = load_events(ann_file)
            # combine the annotations if needed
            if args.combine > 0:
                annotations = combine_events(annotations, args.combine)
            # add the OnsetEvaluation to mean evaluation
            me.append(OnsetEvaluation(detections, annotations,
                                      window=args.window))
            # process the next annotation file
        # print stats for each file
        if args.verbose:
            print det_file
            print me.print_errors('  ', args.tex)
        # add the resulting sum counter
        sum_eval += me
        mean_eval.append(me)
        # process the next detection file
    # print summary
    print 'sum for %i files:' % (len(det_files))
    print sum_eval.print_errors('  ', args.tex)
    print 'mean for %i files:' % (len(det_files))
    print mean_eval.print_errors('  ', args.tex)

if __name__ == '__main__':
    main()
