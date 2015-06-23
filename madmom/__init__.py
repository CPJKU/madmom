# encoding: utf-8
"""
This package is used internally by the Department of Computational Perception,
Johannes Kepler University, Linz, Austria (http://www.cp.jku.at) and the
Austrian Research Institute for Artificial Intelligence (OFAI), Vienna, Austria
(http://www.ofai.at).

All features should be implemented as classes which inherit from Processor
(or provide a XYProcessor(Processor) variant). This way, multiple Processor
objects can be chained to achieve the wanted functionality.

Please see the README for further details of this module.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
import argparse
import multiprocessing as mp
import os
import abc
import contextlib
import sys


MODELS_PATH = '%s/models' % (os.path.dirname(__file__))


# decorator to suppress warnings
def suppress_warnings(function):
    def decorator_function(*args, **kwargs):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return function(*args, **kwargs)
    return decorator_function


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
    import __builtin__
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
        import cPickle
        # close the open file if needed and use its name
        if not isinstance(infile, basestring):
            infile.close()
            infile = infile.name
        # instantiate a new Processor and return it
        return cPickle.load(open(infile, 'rb'))

    def dump(self, outfile):
        """
        Save the Processor to a file.

        This method pickles a Processor object and saves it. Subclasses should
        overwrite this method with a better performing solution if speed is an
        issue.

        :param outfile: output file name or file handle

        """
        import cPickle
        import warnings
        warnings.warn('The resulting file is considered a model file, please '
                      'see the LICENSE file for details!')
        # close the open file if needed and use its name
        if not isinstance(outfile, basestring):
            outfile.close()
            outfile = outfile.name
        # dump the Processor to the given file
        cPickle.dump(self, open(outfile, 'wb'),
                     protocol=cPickle.HIGHEST_PROTOCOL)

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
        """This magic method makes an instance callable."""
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
        # also return the data!
        return data


def _process(process_tuple):
    """
    Function to process a Processor object (first tuple item) with the given
    data (second tuple item).

    Instead of a Processor also a function accepting a single positional
    argument (data) and returning the processed data can be given.

    :param process_tuple: tuple (Processor/function, data)
    :return:              processed data

    Note: This must be a top-level function to be pickle-able.

    """
    # just call whatever we got here, since every Processor is callable
    if process_tuple[0] is None:
        # return the data unaltered
        return process_tuple[1]
    elif isinstance(process_tuple[0], Processor):
        # call the process function
        return process_tuple[0].process(process_tuple[1])
    else:
        # just call the function
        return process_tuple[0](process_tuple[1])


class SequentialProcessor(Processor):
    """
    Class for sequential processing of data.

    """
    def __init__(self, processors):
        """
        Instantiates a SequentialProcessor object.

        :param processors: list with Processor objects

        """
        # wrap the processor in a list if needed
        if isinstance(processors, Processor):
            processors = [processors]
        # save the processors
        self.processors = processors

    def process(self, data):
        """
        Process the data sequentially.

        :param data: data to be processed
        :return:     processed data

        """
        # sequentially process the data
        for processor in self.processors:
            data = _process((processor, data))
        return data

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


# inherit from SequentialProcessor because of append() and extend()
class ParallelProcessor(SequentialProcessor):
    """
    Base class for parallel processing of data.

    """
    NUM_THREADS = 1

    def __init__(self, processors, num_threads=NUM_THREADS):
        """
        Instantiates a ParallelProcessor object.

        :param processors:  list with processing objects
        :param num_threads: number of parallel working threads

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
            import multiprocessing as mp
            self.map = mp.Pool(num_threads).map

    def process(self, data):
        """
        Process the data in parallel.

        :param data: data to be processed
        :return:     list with processed data

        """
        import itertools as it
        # process data in parallel and return a list with processed data
        return self.map(_process, it.izip(self.processors, it.repeat(data)))

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
    (data, output).

    All Processors defined in the input chain are sequentially called with
    only the data argument.

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
              accepting two arguments (data, output)

        """
        # TODO: check the input and output processors!?
        #       as input a Processor, SequentialProcessor, ParallelProcessor
        #       or a function with only one argument should be accepted
        #       as output a OutputProcessor, IOProcessor or function with two
        #       arguments should be accepted
        # wrap the input processor in a SequentialProcessor is needed
        if isinstance(in_processor, list):
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
        data = _process((self.in_processor, data))
        # TODO: unify this with _process!?
        # further process it with the output Processor and return it
        if self.out_processor is None:
            # no processing needed, just return the data
            return data
        elif isinstance(self.out_processor, Processor):
            # call the process method
            return self.out_processor.process(data, output)
        else:
            # or simply call the function
            return self.out_processor(data, output)


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
                  strip_ext=True, num_workers=mp.cpu_count(), **kwargs):
    """
    Process a list of files with the given Processor in batch mode.

    :param processor:     pickled Processor
    :param files:         audio files [list]
    :param output_dir:    output directory
    :param output_suffix: output suffix
    :param strip_ext:     strip off the extension from the files [bool]
    :param num_workers:   number of parallel working threads [int]

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

    """
    processor.dump(outfile)


# generic input/output arguments for scripts
def io_arguments(parser, suffix='.txt'):
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
    sp.add_argument('-s', dest='output_suffix', default=suffix,
                    help='suffix appended to the files (dot must be included '
                         'if wanted) [default=%(default)s]')
    sp.add_argument('--ext', dest='strip_ext', action='store_false',
                    default=True,
                    help='keep the extension of the input file [default='
                         'strip it off before appending the output suffix]')
    sp.add_argument('-j', dest='num_workers', type=int, default=mp.cpu_count(),
                    help='number of parallel workers [default=%(default)s]')

# finally import all submodules
from . import audio, features, evaluation, ml, utils
