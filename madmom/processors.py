# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This file contains all processor related functionality.

All features should be implemented as classes which inherit from Processor
(or provide a XYZProcessor(Processor) variant). This way, multiple Processor
objects can be chained/combined to achieve the wanted functionality.

"""

from __future__ import absolute_import, division, print_function

import os
import sys
import abc
import argparse
import multiprocessing as mp

from collections import MutableSequence


class Processor(object):
    """
    Abstract base class for processing data.

    """
    __metaclass__ = abc.ABCMeta

    @classmethod
    def load(cls, infile):
        """
        Instantiate a new Processor from a file.

        This method un-pickles a saved Processor object. Subclasses should
        overwrite this method with a better performing solution if speed is an
        issue.

        :param infile: file name or file handle
        :return:       Processor instance

        """
        import pickle
        # close the open file if needed and use its name
        if not isinstance(infile, str):
            infile.close()
            infile = infile.name
        # instantiate a new Processor and return it
        return pickle.load(open(infile, 'rb'))

    def dump(self, outfile):
        """
        Save the Processor to a file.

        This method pickles a Processor object and saves it. Subclasses should
        overwrite this method with a better performing solution if speed is an
        issue.

        :param outfile: output file name or file handle

        """
        import pickle
        import warnings
        warnings.warn('The resulting file is considered a model file, please '
                      'see the LICENSE file for details!')
        # close the open file if needed and use its name
        if not isinstance(outfile, str):
            outfile.close()
            outfile = outfile.name
        # dump the Processor to the given file
        pickle.dump(self, open(outfile, 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    @abc.abstractmethod
    def process(self, data):
        """
        Process the data.

        This method must be implemented by the derived class and should
        process the given data and return the processed output.

        :param data: data to be processed
        :return:     processed data

        """
        return data

    def __call__(self, *args):
        """This magic method makes a Processor instance callable."""
        return self.process(*args)


class OutputProcessor(Processor):
    """
    Class for processing data and/or feeding it into some sort of output.

    """

    @abc.abstractmethod
    def process(self, data, output):
        """
        Processes the data and feeds it to output.

        :param data:   data to be processed (e.g. written to file)
        :param output: output file name or file handle
        :return:       also return the processed data

        """
        # pylint: disable=arguments-differ

        # also return the data
        return data


# functions for processing file(s) with a Processor
def _process(process_tuple):
    """
    Function to process a Processor object (first tuple item) with the given
    data (second tuple item). The processed data is returned and if a third
    tuple item is given, the processed data is also outputted to the given
    output file or file handle.

    Instead of a Processor also a function accepting a single positional
    argument (data) or two positional arguments (data, output) and returning
    the processed data can be given.

    :param process_tuple: tuple (Processor/function, data, [output])
    :return:              processed data

    Note: This must be a top-level function to be pickle-able.

    """
    if process_tuple[0] is None:
        # do not process the data, if the first item (i.e. Processor) is None
        return process_tuple[1]
    else:
        # just call whatever we got here (every Processor is callable)
        return process_tuple[0](*process_tuple[1:])


class SequentialProcessor(MutableSequence, Processor):
    """
    Processor class for sequential processing of data.

    """
    def __init__(self, processors):
        """
        Instantiates a SequentialProcessor object.

        :param processors: list of Processor objects

        """
        self.processors = []
        # iterate over all given processors and save them
        for processor in processors:
            # wrap lists and tuples as a SequentialProcessor
            if isinstance(processor, (list, tuple)):
                processor = SequentialProcessor(processor)
            # save the processors
            self.processors.append(processor)

    def __getitem__(self, index):
        """
        Get the Processor at the given processing chain position.

        :param index: position inside the processing chain

        """
        return self.processors[index]

    def __setitem__(self, index, processor):
        """
        Set the Processor at the given processing chain position.

        :param index:     position inside the processing chain
        :param processor: Processor to set

        """
        self.processors[index] = processor

    def __delitem__(self, index):
        """
        Delete the Processor at the given processing chain position.

        :param index: position inside the processing chain

        """
        del self.processors[index]

    def __len__(self):
        """
        Length of the processing chain.

        """
        return len(self.processors)

    def insert(self, index, processor):
        """
        Insert a Processor at the given processing chain position.

        :param index:     position inside the processing chain
        :param processor: Processor to insert

        """
        self.processors.insert(index, processor)

    def append(self, other):
        """
        Append another Processor to the processing chain.

        :param other: the Processor to be appended.

        """
        self.processors.append(other)

    def extend(self, other):
        """
        Extend the processing chain with a list of Processors.

        :param other: list with Processors to be appended.

        """
        self.processors.extend(other)

    def process(self, data):
        """
        Process the data sequentially with the defined processing chain.

        :param data: data to be processed
        :return:     processed data

        """
        # sequentially process the data
        for processor in self.processors:
            data = _process((processor, data))
        return data


# inherit from SequentialProcessor because of append() and extend()
class ParallelProcessor(SequentialProcessor):
    """
    Processor class for parallel processing of data.

    """
    NUM_THREADS = 1

    def __init__(self, processors, num_threads=NUM_THREADS):
        """
        Instantiates a ParallelProcessor object.

        :param processors:  list with processing objects
        :param num_threads: number of parallel working threads

        Note: We use `**kwargs` instead of `num_threads` to be able to pass
              an arbitrary number of processors which should get processed in
              parallel as the first arguments.
              Tuples or lists are wrapped as a `SequentialProcessor`.

        """
        # save the processing chain
        super(ParallelProcessor, self).__init__(processors)
        # number of threads
        if num_threads is None:
            num_threads = 1
        # Note: we must define the map function here, otherwise it leaks both
        #       memory and file descriptors if we init the pool in the process
        #       method. This also means that we must use only 1 thread if we
        #       want to pickle the Processor, because map is pickle-able,
        #       whereas mp.Pool().map is not.
        self.map = map
        if min(len(processors), max(1, num_threads)) != 1:
            self.map = mp.Pool(num_threads).map

    def process(self, data):
        """
        Process the data in parallel.

        :param data: data to be processed
        :return:     list with processed data

        """
        import itertools as it
        # process data in parallel and return a list with processed data
        return list(self.map(_process, zip(self.processors, it.repeat(data))))

    @classmethod
    def add_arguments(cls, parser, num_threads=NUM_THREADS):
        """
        Add parallel processing options to an existing parser object.

        :param parser:      existing argparse parser object
        :param num_threads: number of threads to run in parallel [int]
        :return:            parallel processing argument parser group

        Note: A value of 0 or negative numbers for `num_threads` suppresses the
              inclusion of the parallel option and 'None' is returned instead
              of the parsing group.
              Setting `num_threads` to 'None' sets the number to the number of
              CPU cores.

        """
        if num_threads is None:
            num_threads = cls.NUM_THREADS
        # do not include the group
        if num_threads <= 0:
            return None
        # add parallel processing options
        g = parser.add_argument_group('parallel processing arguments')
        g.add_argument('-j', '--threads', dest='num_threads',
                       action='store', type=int, default=num_threads,
                       help='number of parallel threads [default=%(default)s]')
        # return the argument group so it can be modified if needed
        return g


class IOProcessor(OutputProcessor):
    """
    Input/Output Processor which processes the input data with the input
    Processor and feeds everything into the given output Processor.

    The output Processor is the only one ever called with two arguments
    ('data', 'output').

    All Processors defined in the input chain are sequentially called with the
    'data' argument only.

    """

    def __init__(self, in_processor, out_processor=None):
        """
        Creates a IOProcessor instance.

        :param in_processor:  Processor or list or function
        :param out_processor: OutputProcessor or function

        Note: `in_processor` can be a Processor (or subclass thereof) or a
              function accepting a single argument (data) or a list thereof
              which gets wrapped as a SequentialProcessor.

              `out_processor` can be a OutputProcessor or a function
              accepting two arguments (data, output). If a tuple or list is
              given, it is wrapped in an `IOProcessor` itself with the last
              element regarded as the `out_processor`.

        """
        # TODO: check the input and output processors!?
        #       as input a Processor, SequentialProcessor, ParallelProcessor
        #       or a function with only one argument should be accepted
        #       as output a OutputProcessor, IOProcessor or function with two
        #       arguments should be accepted
        # wrap the input processor in a SequentialProcessor if needed
        if isinstance(in_processor, (list, tuple)):
            self.in_processor = SequentialProcessor(in_processor)
        else:
            self.in_processor = in_processor
        # wrap the output processor in an IOProcessor if needed
        if isinstance(out_processor, list):
            if len(out_processor) >= 2:
                # use the last processor as output and all others as input
                self.out_processor = IOProcessor(out_processor[:-1],
                                                 out_processor[-1])
            if len(out_processor) == 1:
                self.out_processor = out_processor[0]
        else:
            self.out_processor = out_processor

    def __getitem__(self, index):
        """
        Get the Processor at the given position.

        :param index: processor position

        Note: Index '0' refers to the `in_processor`,  index '1' to the
              `out_processor`.

        """
        if index == 0:
            return self.in_processor
        elif index == 1:
            return self.out_processor
        else:
            raise IndexError('Only `in_processor` at index 0 and '
                             '`out_processor` at index 1 are defined.')

    def process(self, data, output=None):
        """
        Processes the data with the input Processor and outputs everything into
        the output Processor.

        :param data:   input data or file to be loaded
                       [numpy array or file name or file handle]
        :param output: output file [file name or file handle]
        :return:       Activations instance

        """
        # process the data by the input processor
        data = _process((self.in_processor, data, ))
        # process the data by the output processor and return it
        return _process((self.out_processor, data, output))


def process_single(processor, infile, outfile, **kwargs):
    """
    Process a single file with the given Processor.

    :param processor: pickled Processor
    :param infile:    input file or file handle
    :param outfile:   outfile file or file handle
    :param kwargs:    additional keyword arguments will be ignored

    """
    # pylint: disable=unused-argument

    processor.process(infile, outfile)


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


# function to batch process multiple files with a processor
def process_batch(processor, files, output_dir=None, output_suffix=None,
                  strip_ext=True, num_workers=mp.cpu_count(), **kwargs):
    """
    Process a list of files with the given Processor in batch mode.

    :param processor:     pickled Processor
    :param files:         audio files [list]
    :param output_dir:    output directory
    :param output_suffix: output suffix
    :param strip_ext:     strip off the extension from the files [bool]
    :param num_workers:   number of parallel working threads [int]
    :param kwargs:        additional keyword arguments will be ignored

    Note: Either `output_dir` or `output_suffix` must be set.

    """
    # pylint: disable=unused-argument

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
        # strip off the extension
        if strip_ext:
            output_file = os.path.splitext(output_file)[0]
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
    :param kwargs:    additional keyword arguments will be ignored

    """
    # pylint: disable=unused-argument

    processor.dump(outfile)


# generic input/output arguments for scripts
def io_arguments(parser, output_suffix='.txt', pickle=True):
    """
    Add input / output related arguments to an existing parser.

    :param parser:        existing argparse parser
    :param output_suffix: suffix appended to the output files
    :param pickle:        add 'pickle' subparser [bool]

    """
    # add general options
    parser.add_argument('-v', dest='verbose', action='count',
                        help='increase verbosity level')
    # add subparsers
    sub_parsers = parser.add_subparsers(title='processing options')
    if pickle:
        # pickle processor options
        sp = sub_parsers.add_parser('pickle', help='pickle processor')
        sp.set_defaults(func=pickle_processor)
        # Note: requiring '-o' is a simple safety measure to not overwrite
        #       existing audio files after using the processor in 'batch' mode
        sp.add_argument('-o', dest='outfile', type=argparse.FileType('w'),
                        help='file to pickle the processor to')
    # single file processing options
    sp = sub_parsers.add_parser('single', help='single file processing')
    sp.set_defaults(func=process_single)
    sp.add_argument('infile', type=argparse.FileType('r'),
                    help='input audio file')
    # Note: requiring '-o' is a simple safety measure to not overwrite existing
    #       audio files after using the processor in 'batch' mode
    sp.add_argument('-o', dest='outfile', type=argparse.FileType('w'),
                    default=sys.stdout, help='output file [default: STDOUT]')
    sp.add_argument('-j', dest='num_threads', type=int, default=mp.cpu_count(),
                    help='number of parallel threads [default=%(default)s]')
    # batch file processing options
    sp = sub_parsers.add_parser('batch', help='batch file processing')
    sp.set_defaults(func=process_batch)
    sp.add_argument('files', nargs='+', help='files to be processed')
    sp.add_argument('--out', dest='output_dir', default=None,
                    help='output directory [default=%(default)s]')
    sp.add_argument('-s', dest='output_suffix', default=output_suffix,
                    help='suffix appended to the files (dot must be included '
                         'if wanted) [default=%(default)s]')
    sp.add_argument('--ext', dest='strip_ext', action='store_false',
                    default=True,
                    help='keep the extension of the input file [default='
                         'strip it off before appending the output suffix]')
    sp.add_argument('-j', dest='num_workers', type=int, default=mp.cpu_count(),
                    help='number of parallel workers [default=%(default)s]')
    sp.set_defaults(num_threads=1)
