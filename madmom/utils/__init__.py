# encoding: utf-8
"""
Utility package.

"""

import os
import sys
import glob
import fnmatch
import contextlib
import __builtin__
import argparse
import multiprocessing as mp

import numpy as np


# overwrite the built-in open() to transparently apply some magic file handling
@contextlib.contextmanager
def open(filename, mode='r'):
    """
    Context manager which yields an open file or handle with the given mode
    and closes it if needed afterwards.

    :param filename: file name or open file handle
    :param mode:     mode in which to open the file
    :return:         an open file handle

    """
    # check if we need to open the file
    if isinstance(filename, basestring):
        f = fid = __builtin__.open(filename, mode)
    else:
        f = filename
        fid = None
    # TODO: include automatic (un-)zipping here?
    # yield an open file handle
    yield f
    # close the file if needed
    if fid:
        fid.close()


def search_files(path, suffix=None):
    """
    Returns a list of files in path matching the given suffix or filters
    the given list to include only those matching the given suffix.

    :param path:   path or list of files to be searched / filtered
    :param suffix: only return files with this suffix [string, list]
    :return:       list of files

    """
    # determine the files
    if type(path) == list:
        # a list of files or paths is given
        file_list = []
        # recursively call the function
        for f in path:
            file_list.extend(search_files(f, suffix))
    elif os.path.isdir(path):
        # use all files in the given path
        if suffix is None:
            file_list = glob.glob("%s/*" % path)
        elif isinstance(suffix, list):
            file_list = []
            for e in suffix:
                file_list.extend(glob.glob("%s/*%s" % (path, e)))
        else:
            file_list = glob.glob("%s/*%s" % (path, suffix))
    elif os.path.isfile(path):
        file_list = []
        # no matching needed
        if suffix is None:
            file_list = [path]
        # a list of suffices is given
        elif isinstance(suffix, list):
            for e in suffix:
                if path.endswith(e):
                    file_list = [path]
        # a single suffix is given
        elif path.endswith(suffix):
            file_list = [path]
    else:
        raise IOError("%s does not exist." % path)
    # sort files
    file_list.sort()
    # return file list
    return file_list


def strip_suffix(filename, suffix=None):
    """
    Strip of the suffix of the given filename or string.

    :param filename: filename or string to process
    :param suffix:   suffix to be stripped off
    :return:         filename or string without suffix

    """
    if suffix is not None and filename.endswith(suffix):
        return filename[:-len(suffix)]
    return filename


def match_file(filename, match_list, suffix=None, match_suffix=None):
    """
    Match a filename or string against a list of other filenames or strings.

    :param filename:     filename or string to be matched
    :param match_list:   match to this list of filenames or strings
    :param suffix:       ignore this suffix of the filename when matching
    :param match_suffix: only match files with this suffix
    :return:             list of matched files

    """
    # get the base name without the path
    basename = os.path.basename(strip_suffix(filename, suffix))
    # init return list
    matches = []
    # look for files with the same base name in the files_list
    if match_suffix is not None:
        pattern = "*%s*%s" % (basename, match_suffix)
    else:
        pattern = "*%s" % basename
    for match in fnmatch.filter(match_list, pattern):
        # base names must match exactly
        if basename == os.path.basename(strip_suffix(match, match_suffix)):
            matches.append(match)
    # return the matches
    return matches


def load_events(filename):
    """
    Load a list of events from a text file, one floating point number per line.

    :param filename: name of the file or file handle
    :return:         numpy array of events

    Note: Comments (i.e. lines tarting with '#') are ignored.

    """
    with open(filename, 'rb') as f:
        # read in the events, one per line
        # 1st column is the event's time, the rest is ignored
        return np.fromiter((float(line.split(None, 1)[0]) for line in f
                            if not line.startswith('#')), dtype=np.float)


def write_events(events, filename):
    """
    Write a list of events to a text file, one floating point number per line.

    :param events:   list of events [seconds]
    :param filename: output file name or file handle

    """
    # write the events to the output
    if filename is not None:
        with open(filename, 'wb') as f:
            f.writelines('%g\n' % e for e in events)
    # also return them
    return events


def combine_events(events, delta):
    """
    Combine all events within a certain range.

    :param events: list of events [seconds]
    :param delta:  combination length [seconds]
    :return:       list of combined events

    """
    # add a small value to delta, otherwise we end up in floating point hell
    delta += 1e-12
    # return immediately if possible
    if len(events) <= 1:
        return events
    # create working copy
    events = np.array(events, copy=True)
    # set start position
    idx = 0
    # get first event
    left = events[idx]
    # iterate over all remaining events
    for right in events[1:]:
        if right - left <= delta:
            # combine the two events
            left = events[idx] = 0.5 * (right + left)
        else:
            # move forward
            idx += 1
            left = events[idx] = right
    # return the combined events
    return events[:idx + 1]


def quantize_events(events, fps, length=None, shift=None):
    """
    Quantize the events with the given resolution.

    :param events: sequence of events [seconds]
    :param fps:    quantize with N frames per second
    :param length: length of the returned array [frames]
    :param shift:  shift the events by N seconds before quantisation
    :return:       a quantized numpy array

    """
    # shift all events if needed
    if shift:
        events = np.asarray(events) + shift
    # determine the length for the quantized array
    if length is None:
        # set the length to be long enough to cover all events
        length = int(round(np.max(events) * float(fps))) + 1
    else:
        # else filter all events which do not fit in the array
        # since we apply rounding later, we need to subtract half a bin
        events = events[:np.searchsorted(events, float(length - 0.5) / fps)]
    # init array
    quantized = np.zeros(length)
    # set the events
    for event in events:
        idx = int(round(event * float(fps)))
        quantized[idx] = 1
    # return the quantized array
    return quantized


# argparse action to set and overwrite default lists
class OverrideDefaultListAction(argparse.Action):
    """
    OverrideDefaultListAction

    An argparse action that works similarly to the regular 'append' action.
    The default value is deleted when a new value is specified. The 'append'
    action would append the new value to the default.

    """
    def __init__(self, *args, **kwargs):
        super(OverrideDefaultListAction, self).__init__(*args, **kwargs)
        self.set_to_default = True

    def __call__(self, parser, namespace, value, option_string=None):
        if self.set_to_default:
            setattr(namespace, self.dest, [])
            self.set_to_default = False
        cur_values = getattr(namespace, self.dest)
        cur_values.append(value)


# functions for processing file(s) with a Processor
def process_single(processor, input, output, **kwargs):
    """
    Process a single file with the given Processor.

    :param processor: pickled Processor
    :param input:     input audio file
    :param output:    output file

    """
    processor.process(input, output)


class ParallelProcess(mp.Process):
    """
    Parallel Process class.

    """
    def __init__(self, task_queue):
        """
        Create a ParallelProcess, which processes tasks.

        :param task_queue: queue with tasks

        """
        mp.Process.__init__(self)
        self.task_queue = task_queue

    def run(self):
        """
        Process all tasks from the task queue.

        """
        while True:
            # get the task tuple
            processor, input_file, output_file = self.task_queue.get()
            # process the Processor with the data
            processor.process(input_file, output_file)
            # signal that it is done
            self.task_queue.task_done()


def process_batch(processor, files, output_dir=None, output_suffix=None,
                  num_workers=mp.cpu_count(), **kwargs):
    """
    Process a list of files with the given Processor in batch mode.

    :param processor:     pickled Processor
    :param files:         audio files [list]
    :param output_dir:    output directory
    :param output_suffix: output suffix
    :param num_workers:   number of parallel working threads

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
    processes = [ParallelProcess(tasks) for _ in range(num_workers)]
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


# function for pickling a processor
def pickle_processor(processor, outfile, **kwargs):
    """
    Pickle the Processor to file.

    :param processor: the Processor
    :param outfile:   file where to pickle it

    """
    processor.dump(outfile)


# generic input/output arguments for scripts
def io_arguments(parser):
    """
    Add input / output related arguments to an existing parser.

    :param parser: existing argparse parser

    """
    # add general options
    parser.add_argument('-v', dest='verbose', action='count',
                        help='increase verbosity level')
    # add subparsers
    sub_parsers = parser.add_subparsers(title='processing options')
    # pickle processor options
    sp = sub_parsers.add_parser('pickle', help='pickle processor')
    sp.set_defaults(func=pickle_processor)
    sp.add_argument('outfile', type=str, help='file to pickle the processor')
    # single file processing options
    sp = sub_parsers.add_parser('single', help='single file processing')
    sp.set_defaults(func=process_single)
    sp.add_argument('input', type=argparse.FileType('r'),
                    help='input audio file')
    sp.add_argument('output', nargs='?',
                    type=argparse.FileType('w'), default=sys.stdout,
                    help='output file [default: STDOUT]')
    sp.add_argument('-j', dest='num_threads', type=int, default=mp.cpu_count(),
                    help='number of parallel threads [default=%(default)s]')
    # batch file processing options
    sp = sub_parsers.add_parser('batch', help='batch file processing')
    sp.set_defaults(func=process_batch)
    sp.add_argument('files', nargs='+', help='files to be processed')
    sp.add_argument('-o', dest='output_dir', default=None,
                    help='output directory [default=%(default)s]')
    sp.add_argument('-s', dest='output_suffix', default='.txt',
                    help='extension appended to the files '
                         '[default=%(default)s]')
    sp.add_argument('-j', dest='num_workers', type=int, default=mp.cpu_count(),
                    help='number of parallel workers [default=%(default)s]')

# finally import the submodules
from . import midi
