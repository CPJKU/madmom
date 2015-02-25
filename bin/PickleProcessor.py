#!/usr/bin/env python
# encoding: utf-8
"""
@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import os
import cPickle
import multiprocessing as mp


def process_single(processor, input, output):
    """
    Process a single file with the given Processor.

    :param processor: pickled Processor
    :param input:     input audio file
    :param output:    output file

    """
    processor.process(input, output)


class ParallelProcess(mp.Process):
    """
    Parallel Processing class.

    """

    def __init__(self, task_queue):
        """
        Create a ParallelProcess, which processes the tasks.

        :param task_queue: queue with tasks

        """
        mp.Process.__init__(self)
        self.task_queue = task_queue

    def run(self):
        """
        Process all task from the task queue.

        """
        while True:
            # get the task tuple
            processor, input_file, output_file = self.task_queue.get()
            # process the Processor with the data
            processor.process(input_file, output_file)
            # signal that it is done
            self.task_queue.task_done()


def process_batch(processor, files, output_dir=None, output_suffix=None,
                  num_threads=mp.cpu_count()):
    """
    Process a list of files with the given Processor in batch mode.

    :param processor:     pickled Processor
    :param files:         audio files [list]
    :param output_dir:    output directory
    :param output_suffix: output suffix
    :param num_threads:   number of parallel threads

    Note: Either `output_dir` or `output_suffix` must be set.

    """
    # either output_dir or output_suffix must be given
    if output_dir is None and output_suffix is None:
        raise ValueError('either output directory or suffix must be given')
    # make sure the directory exists
    if output_dir is not None:
        try:
            # create output directory
            os.mkdir(output_dir)
        except OSError:
            # directory exists already
            pass

    # create task queue
    tasks = mp.JoinableQueue()
    # create working threads
    processes = [ParallelProcess(tasks) for _ in range(num_threads)]
    for p in processes:
        p.daemon = True
        p.start()

    # process all the files
    for input_file in files:
        # set the output file name
        if output_dir is not None:
            output_file = "%s/%s" % (output_dir, os.path.basename(input_file))
        else:
            output_file = input_file
        # append the suffix if needed
        if output_suffix is not None:
            output_file += output_suffix
        # put processing tasks in the queue
        tasks.put((processor, input_file, output_file))
    # wait for all processing tasks to finish
    tasks.join()


def main():
    """Generic Pickle Processor."""

    import argparse
    import sys

    # define parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    This script runs previously pickled Processors in either batch or single
    file mode.

    To obtain a pickled Processor, call the 'dump(outfile)' method.

    ''')
    # general options
    parser.add_argument('processor', type=argparse.FileType('r'),
                        help='pickled processor')
    parser.add_argument('--version', action='version',
                        version='PickleProcessor v0.1')

    # add subparsers
    sub_parsers = parser.add_subparsers()

    # single file processing
    sp = sub_parsers.add_parser('single', help='single file processing')
    sp.set_defaults(func=process_single)
    sp.add_argument('input', type=argparse.FileType('r'),
                    help='input audio file')
    sp.add_argument('output', nargs='?',
                    type=argparse.FileType('w'), default=sys.stdout,
                    help='output file [default: STDOUT]')

    # batch file processing
    sp = sub_parsers.add_parser('batch', help='batch file processing')
    sp.set_defaults(func=process_batch)
    sp.add_argument('files', nargs='+', help='files to be processed')
    sp.add_argument('-o', dest='output_dir', default=None,
                    help='output directory [default=%(default)s]')
    sp.add_argument('-s', dest='output_suffix', default='.txt',
                    help='suffix appended to the files [default=%(default)s]')
    sp.add_argument('-j', dest='num_threads', type=int, default=mp.cpu_count(),
                    help='number of parallel threads [default=%(default)s]')

    # parse arguments and call the processing function
    args = parser.parse_args()
    kwargs = vars(args)
    function = kwargs.pop('func')
    processor = cPickle.load(kwargs.pop('processor'))
    function(processor, **kwargs)


if __name__ == '__main__':
    main()
