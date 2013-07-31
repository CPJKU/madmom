#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (c) 2012 Sebastian Böck <sebastian.boeck@jku.at>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy as np

from helpers import load_events, combine_events
from simple import Evaluation, SumEvaluation, MeanEvaluation


# evaluation function for onset detection
def count_errors(detections, targets, window):
    """
    Count the true and false detections of the given detections and targets.

    :param detections: a list of events [seconds]
    :param targets: a list of events [seconds]
    :param window: detection window [seconds]
    :return: tuple of tp, fp, tn, fn numpy arrays

    tp: list with true positive detections
    fp: list with false positive detections
    tn: list with true negative detections (this one is empty!)
    fn: list with false negative detections

    """
    from helpers import calc_absolute_errors
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


# for onset evaluation with Presicion, Recall, F-measure use the Evaluation
# class and just define the evaluation function
class OnsetEvaluation(Evaluation):
    """
    Simple class for measuring Precision, Recall and F-measure.

    """
    def __init__(self, detections, targets, window=0.025):
        super(OnsetEvaluation, self).__init__(detections, targets, count_errors, window=window)


class SumOnsetEvaluation(SumEvaluation):
    pass


class MeanOnsetEvaluation(MeanEvaluation):
    pass


def main():
    import os.path
    import argparse
    import glob
    import fnmatch

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
    p.add_argument('-v', dest='verbose', action='store_true', help='be verbose')
    # extensions used for evaluation
    p.add_argument('-d', dest='detections', action='store', default='.onsets.txt', help='extensions of the detections [default: .onsets.txt]')
    p.add_argument('-t', dest='targets', action='store', default='.onsets', help='extensions of the targets [default: .onsets]')
    # parameters for evaluation
    p.add_argument('-w', dest='window', action='store', default=50, type=float, help='evaluation window [in milliseconds]')
    p.add_argument('-c', dest='combine', action='store', default=30, type=float, help='combine target events within this range [in milliseconds]')
    p.add_argument('--delay', action='store', default=0., type=float, help='add given delay to all detections [in milliseconds]')
    p.add_argument('--tex', action='store_true', help='format errors for use is .tex files')
    # version
    p.add_argument('--version', action='version', version='%(prog)s 1.0 (2012-10-01)')
    # parse the arguments
    args = p.parse_args()

    # convert the detection, combine, and delay values to seconds
    args.window /= 2000.  # also halve the size of the detection window
    args.combine /= 1000.
    args.delay /= 1000.

    # determine the files to process
    files = []
    for f in args.files:
        # check what we have (file/path)
        if os.path.isdir(f):
            # use all files in the given path
            files = glob.glob(f + '/*')
        else:
            # file was given, append to list
            files.append(f)
    # sort files
    files.sort()

    # TODO: find a better way to determine the corresponding detection/target files from a given list/path of files
    # filter target files
    tar_files = fnmatch.filter(files, "*%s" % args.targets)
    # filter detection files
    det_files = fnmatch.filter(files, "*%s" % args.detections)
    # must be the same number
    assert len(tar_files) == len(det_files), "different number of targets (%i) and detections (%i)" % (len(tar_files), len(det_files))

    # sum counter for all files
    sum_counter = SumOnsetEvaluation()
    mean_counter = MeanOnsetEvaluation()
    # evaluate all files
    for i in range(len(det_files)):
        detections = load_events(det_files[i])
        targets = load_events(tar_files[i])
        # combine the targets if needed
        if args.combine > 0:
            targets = combine_events(targets, args.combine)
        # shift the detections if needed
        if args.delay != 0:
            detections += args.delay
        # evaluate the onsets
        oe = OnsetEvaluation(detections, targets, args.window)
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
