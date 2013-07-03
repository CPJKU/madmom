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

from helpers import load_events, combine_events

import numpy as np  # needed only for mean and std.dev in Counter.print_errors()


class Counter(object):
    """
    Simple class for counting errors.

    """
    def __init__(self):
        """
        Creates a new Counter object instance.

        """
        # for simple events like onsets or beats
        self.num = 0    # number of targets
        self.tp = 0     # number of true positives
        self.fp = 0     # number of false positives
        self.fn = 0     # number of false negatives
        self.dev = []   # array for deviations

    # for adding 2 Counters
    def __add__(self, other):
        if isinstance(other, Counter):
            self.num += other.num
            self.tp += other.tp
            self.fp += other.fp
            self.fn += other.fn
            self.dev.extend(other.dev)
            return self
        else:
            return NotImplemented

    @property
    def precision(self):
        """Precision."""
        try:
            return self.tp / float(self.tp + self.fp)
        except ZeroDivisionError:
            return 0.

    @property
    def recall(self):
        """Recall."""
        try:
            return self.tp / float(self.tp + self.fn)
        except ZeroDivisionError:
            return 0.

    @property
    def fmeasure(self):
        """F-measure."""
        try:
            return 2. * self.precision * self.recall / (self.precision + self.recall)
        except ZeroDivisionError:
            return 0.

    @property
    def accuracy(self):
        """Accuracy."""
        try:
            return self.tp / float(self.fp + self.fn + self.tp)
        except ZeroDivisionError:
            return 0.

    @property
    def true_positive_rate(self):
        """True positive rate."""
        try:
            return self.tp / float(self.num)
        except ZeroDivisionError:
            return 0.

    @property
    def false_positive_rate(self):
        """False positive rate."""
        try:
            return self.fp / float(self.fp + self.tp)
        except ZeroDivisionError:
            return 0.

    @property
    def false_negative_rate(self):
        """False negative rate."""
        try:
            return self.fn / float(self.fn + self.tp)
        except ZeroDivisionError:
            return 0.

    def print_errors(self, tex=False):
        """
        Print errors.

        :param tex: output format to be used in .tex files [default=False]

        """
        # print the errors
        print '  targets: %5d correct: %5d fp: %4d fn: %4d p=%.3f r=%.3f f=%.3f' % (self.num, self.tp, self.fp, self.fn, self.precision, self.recall, self.fmeasure)
        print '  tp: %.1f%% fp: %.1f%% acc: %.1f%% mean: %.1f ms std: %.1f ms' % (self.true_positive_rate * 100., self.false_positive_rate * 100., self.accuracy * 100., np.mean(self.dev) * 1000., np.std(self.dev) * 1000.)
        if tex:
            print "%i events & Precision & Recall & F-measure & True Positves & False Positives & Accuracy & Delay\\\\" % (self.num)
            print "tex & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f %.1f\$\\pm\$%.1f\\,ms\\\\" % (self.precision, self.recall, self.fmeasure, self.true_positive_rate, self.false_positive_rate, self.accuracy, np.mean(self.dev) * 1000., np.std(self.dev) * 1000.)


def evaluate(detections, targets, window, delay=0):
    """
    Evaluate the errors for the given detections and targets.

    :param detections: a list of events [seconds]
    :param targets: a list of events [seconds]
    :param window: detection window [seconds]
    :param delay: add delay to all detections [seconds, default=0]
    :return: a Counter object instance

    """
    # sort the detections and targets
    detections.sort()
    targets.sort()
    # counter for evaluation
    counter = Counter()
    counter.num = len(targets)
    # evaluate
    det_length = len(detections)
    tar_length = len(targets)
    det_index = 0
    tar_index = 0
    while det_index < det_length and tar_index < tar_length:
        # TODO: right now the first detection is compared to the first target
        # but we should compare the closets to get correct mean/std.dev values
        # besides that the evaluation is correct
        # fetch the first detection
        det = detections[det_index]
        # fetch the first target
        tar = targets[tar_index]
        # shift with delay
        if abs(det + delay - tar) <= window:
            # TP detection
            counter.tp += 1
            # save the deviation
            counter.dev.append(det + delay - tar)
            # increase the detection and target index
            det_index += 1
            tar_index += 1
        elif det + delay < tar:
            # FP detection
            counter.fp += 1
            # increase the detection index
            det_index += 1
            # do not increase the target index
        elif det + delay > tar:
            # we missed a target, thus FN
            counter.fn += 1
            # do not increase the detection index
            # increase the target index
            tar_index += 1
    # the remaining detections are FP
    counter.fp += det_length - det_index
    # the remaining targets are FN
    counter.fn += tar_length - tar_index
    assert counter.tp == counter.num - counter.fn, "too stupid to count correctly"
    # return the counter
    return counter


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
    sum_counter = Counter()
    # evaluate all files
    for i in range(len(det_files)):
        detections = load_events(det_files[i])
        targets = load_events(tar_files[i])
        if args.combine > 0:
            targets = combine_events(targets, args.combine)
        counter = evaluate(detections, targets, args.window, args.delay)
        # print stats for each file
        if args.verbose:
            print det_files[i]
            counter.print_errors(args.tex)
        # add to sum counter
        sum_counter += counter
    # print summary
    print 'summary for %i files; detection window %.1f ms (+- %.1f ms)' % (len(det_files), args.window * 2000, args.window * 1000)
    sum_counter.print_errors(args.tex)

if __name__ == '__main__':
    main()
