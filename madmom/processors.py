# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains all processor related functionality.

Notes
-----
All features should be implemented as classes which inherit from Processor
(or provide a XYZProcessor(Processor) variant). This way, multiple Processor
objects can be chained/combined to achieve the wanted functionality.

"""

from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import multiprocessing as mp

import numpy as np

from collections import MutableSequence


class Processor(object):
    """
    Abstract base class for processing data.

    """

    @classmethod
    def load(cls, infile):
        """
        Instantiate a new Processor from a file.

        This method un-pickles a saved Processor object. Subclasses should
        overwrite this method with a better performing solution if speed is an
        issue.

        Parameters
        ----------
        infile : str or file handle
            Pickled processor.

        Returns
        -------
        :class:`Processor` instance
            Processor.

        """
        import pickle
        # close the open file if needed and use its name
        try:
            infile.close()
            infile = infile.name
        except AttributeError:
            pass
        # instantiate a new Processor and return it
        with open(infile, 'rb') as f:
            # Python 2 and 3 behave differently
            try:
                # Python 3
                obj = pickle.load(f, encoding='latin1')
            except TypeError:
                # Python 2 doesn't have/need the encoding
                obj = pickle.load(f)
        # warn if the unpickled Processor is of other type
        if obj.__class__ is not cls:
            import warnings
            warnings.warn("Expected Processor of class '%s' but loaded "
                          "Processor is of class '%s', processing anyways." %
                          (cls.__name__, obj.__class__.__name__))
        return obj

    def dump(self, outfile):
        """
        Save the Processor to a file.

        This method pickles a Processor object and saves it. Subclasses should
        overwrite this method with a better performing solution if speed is an
        issue.

        Parameters
        ----------
        outfile : str or file handle
            Output file for pickling the processor.

        """
        import pickle
        # close the open file if needed and use its name
        try:
            outfile.close()
            outfile = outfile.name
        except AttributeError:
            pass
        # dump the Processor to the given file
        # Note: for Python 2 / 3 compatibility reason use protocol 2
        pickle.dump(self, open(outfile, 'wb'), protocol=2)

    def process(self, data):
        """
        Process the data.

        This method must be implemented by the derived class and should
        process the given data and return the processed output.

        Parameters
        ----------
        data : depends on the implementation of subclass
            Data to be processed.

        Returns
        -------
        depends on the implementation of subclass
            Processed data.

        """
        raise NotImplementedError('must be implemented by subclass.')

    def __call__(self, *args, **kwargs):
        # this magic method makes a Processor callable
        return self.process(*args, **kwargs)


class OutputProcessor(Processor):
    """
    Class for processing data and/or feeding it into some sort of output.

    """

    def process(self, data, output):
        """
        Processes the data and feed it to the output.

        This method must be implemented by the derived class and should
        process the given data and return the processed output.

        Parameters
        ----------
        data : depends on the implementation of subclass
            Data to be processed (e.g. written to file).
        output : str or file handle
            Output file name or file handle.

        Returns
        -------
        depends on the implementation of subclass
            Processed data.

        """
        # pylint: disable=arguments-differ
        raise NotImplementedError('must be implemented by subclass.')


# functions for processing file(s) with a Processor
def _process(process_tuple):
    """
    Function to process a Processor with data.

    The processed data is returned and if applicable also piped to the given
    output.

    Parameters
    ----------
    process_tuple : tuple (Processor/function, data, [output])
        The tuple must contain a Processor object as the first item and the
        data to be processed as the second tuple item. If a third tuple item
        is given, it is used as an output.
        Instead of a Processor also a function accepting a single positional
        argument (data) or two positional arguments (data, output) can be
        given. It must behave exactly as a :class:`Processor`, i.e. return
        the processed data and optionally pipe it to the output.

    Returns
    -------
    depends on the processor
        Processed data.

    Notes
    -----
    This must be a top-level function to be pickle-able.

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

    Parameters
    ----------
    processors : list
         Processor instances to be processed sequentially.

    Notes
    -----
    If the `processors` list contains lists or tuples, these get wrapped as a
    SequentialProcessor itself.

    """

    def __init__(self, processors):
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

        Parameters
        ----------
        index : int
            Position inside the processing chain.

        Returns
        -------
        :class:`Processor`
            Processor at the given position.

        """
        return self.processors[index]

    def __setitem__(self, index, processor):
        """
        Set the Processor at the given processing chain position.

        Parameters
        ----------
        index : int
            Position inside the processing chain.
        processor : :class:`Processor`
            Processor to set.

        """
        self.processors[index] = processor

    def __delitem__(self, index):
        """
        Delete the Processor at the given processing chain position.

        Parameters
        ----------
        index : int
            Position inside the processing chain.

        """
        del self.processors[index]

    def __len__(self):
        """Length of the processing chain."""
        return len(self.processors)

    def insert(self, index, processor):
        """
        Insert a Processor at the given processing chain position.

        Parameters
        ----------
        index : int
            Position inside the processing chain.
        processor : :class:`Processor`
            Processor to insert.

        """
        self.processors.insert(index, processor)

    def append(self, other):
        """
        Append another Processor to the processing chain.

        Parameters
        ----------
        other : :class:`Processor`
            Processor to append to the processing chain.

        """
        self.processors.append(other)

    def extend(self, other):
        """
        Extend the processing chain with a list of Processors.

        Parameters
        ----------
        other : list
            Processors to be appended to the processing chain.

        """
        self.processors.extend(other)

    def process(self, data, **kwargs):
        """
        Process the data sequentially with the defined processing chain.

        Parameters
        ----------
        data : depends on the first processor of the processing chain
            Data to be processed.

        Returns
        -------
        depends on the last processor of the processing chain
            Processed data.

        """
        # sequentially process the data
        for processor in self.processors:
            data = _process((processor, data))
        return data


# inherit from SequentialProcessor because of append() and extend()
class ParallelProcessor(SequentialProcessor):
    """
    Processor class for parallel processing of data.

    Parameters
    ----------
    processors : list
        Processor instances to be processed in parallel.
    num_threads : int, optional
        Number of parallel working threads.

    Notes
    -----
    If the `processors` list contains lists or tuples, these get wrapped as a
    :class:`SequentialProcessor`.

    """
    # pylint: disable=too-many-ancestors

    def __init__(self, processors, num_threads=None):
        # set the processing chain
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
        if min(len(processors), max(1, num_threads)) > 1:
            self.map = mp.Pool(num_threads).map

    def process(self, data):
        """
        Process the data in parallel.

        Parameters
        ----------
        data : depends on the processors
            Data to be processed.

        Returns
        -------
        list
            Processed data.

        """
        import itertools as it
        # process data in parallel and return a list with processed data
        return list(self.map(_process, zip(self.processors, it.repeat(data))))

    @staticmethod
    def add_arguments(parser, num_threads):
        """
        Add parallel processing options to an existing parser object.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        num_threads : int, optional
            Number of parallel working threads.

        Returns
        -------
        argparse argument group
            Parallel processing argument parser group.

        Notes
        -----
        The group is only returned if only if `num_threads` is not 'None'.
        Setting it smaller or equal to 0 sets it the number of CPU cores.

        """
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
    processor and pipes everything into the given output processor.

    All Processors defined in the input chain are sequentially called with the
    'data' argument only. The output Processor is the only one ever called with
    two arguments ('data', 'output').

    Parameters
    ----------
    in_processor : :class:`Processor`, function, tuple or list
        Input processor. Can be a :class:`Processor` (or subclass thereof
        like :class:`SequentialProcessor` or :class:`ParallelProcessor`), a
        function accepting a single argument ('data'). If a tuple or list
        is given, it is wrapped as a :class:`SequentialProcessor`.
    out_processor : :class:`OutputProcessor`, function, tuple or list
        OutputProcessor or function accepting two arguments ('data', 'output').
        If a tuple or list is given, it is wrapped in an :class:`IOProcessor`
        itself with the last element regarded as the `out_processor` and all
        others as `in_processor`.

    """

    def __init__(self, in_processor, out_processor=None):
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

        Parameters
        ----------
        index : int
            Processor position. Index '0' refers to the `in_processor`,
            index '1' to the `out_processor`.

        Returns
        -------
        :class:`Processor`
            Processor at the given position.

        """
        if index == 0:
            return self.in_processor
        elif index == 1:
            return self.out_processor
        else:
            raise IndexError('Only `in_processor` at index 0 and '
                             '`out_processor` at index 1 are defined.')

    def process(self, data, output=None, **kwargs):
        """
        Processes the data with the input processor and pipe everything into
        the output processor, which also pipes it to `output`.

        Parameters
        ----------
        data : depends on the input processors
            Data to be processed.
        output: str or file handle
            Output file (handle).

        Returns
        -------
        depends on the output processors
            Processed data.

        """
        # process the data by the input processor
        data = _process((self.in_processor, data, ))
        # process the data by the output processor and return it
        return _process((self.out_processor, data, output))


# functions and classes to process files with a Processor
def process_single(processor, infile, outfile, **kwargs):
    """
    Process a single file with the given Processor.

    Parameters
    ----------
    processor : :class:`Processor` instance
        Processor to be processed.
    infile : str or file handle
        Input file (handle).
    outfile : str or file handle
        Output file (handle).

    """
    # pylint: disable=unused-argument
    processor(infile, outfile)


class _ParallelProcess(mp.Process):
    """
    Class for processing tasks in a queue.

    Parameters
    ----------
    task_queue :
        Queue with tasks, i.e. tuples ('processor', 'infile', 'outfile')

    Notes
    -----
    Usually, multiple instances are created via :func:`process_batch`.

    """
    def __init__(self, task_queue):
        super(_ParallelProcess, self).__init__()
        self.task_queue = task_queue

    def run(self):
        """Process all tasks from the task queue."""
        from .audio.signal import LoadAudioFileError
        while True:
            # get the task tuple
            processor, infile, outfile = self.task_queue.get()
            try:
                # process the Processor with the data
                processor.process(infile, outfile)
            except LoadAudioFileError as e:
                print(e)
            # signal that it is done
            self.task_queue.task_done()


# function to batch process multiple files with a processor
def process_batch(processor, files, output_dir=None, output_suffix=None,
                  strip_ext=True, num_workers=mp.cpu_count(), shuffle=False,
                  **kwargs):
    """
    Process a list of files with the given Processor in batch mode.

    Parameters
    ----------
    processor : :class:`Processor` instance
        Processor to be processed.
    files : list
        Input file(s) (handles).
    output_dir : str, optional
        Output directory.
    output_suffix : str, optional
        Output suffix (e.g. '.txt' including the dot).
    strip_ext : bool, optional
        Strip off the extension from the input files.
    num_workers : int, optional
        Number of parallel working threads.
    shuffle : bool, optional
        Shuffle the `files` before distributing them to the working threads

    Notes
    -----
    Either `output_dir` and/or `output_suffix` must be set. If `strip_ext` is
    True, the extension of the input file names is stripped off before the
    `output_suffix` is appended to the input file names.

    Use `shuffle` if you experience out of memory errors (can occur for certain
    methods with high memory consumptions if consecutive files are rather
    long).

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
    processes = [_ParallelProcess(tasks) for _ in range(num_workers)]
    for p in processes:
        p.daemon = True
        p.start()

    # shuffle files?
    if shuffle:
        from random import shuffle
        shuffle(files)

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


# processor for buffering data
class BufferProcessor(Processor):
    """
    Buffer for processors which need context to do their processing.

    E.g. SpectrogramDifference needs a context of two frames to be able to
    compute the difference between two consecutive frames.

    Parameters
    ----------
    buffer_length : int
        Length of the buffer (in time steps to be buffered).

    """

    def __init__(self, buffer_length, init=None):
        self.buffer_length = buffer_length
        self._buffer = init

    def process(self, data):
        """
        Buffer the data.

        Parameters
        ----------
        data : numpy array or subclass thereof
            Data to be buffered.

        Returns
        -------
        numpy array or subclass thereof
            Data with buffered context.

        """
        # length of the data
        data_length = len(data)
        # init the buffer with the same ndarray subclass and data type
        if self._buffer is None:
            # TODO: find a better way to concatenate two subclassed ndarrays
            #       which keeps the class/type/dtype intact
            self._buffer = np.repeat(data[:1] * 0,
                                     self.buffer_length + data_length, axis=0)
        # remove `data_length` from buffer at the beginning and append new data
        self._buffer = np.roll(self._buffer, -data_length, axis=0)
        self._buffer[-data_length:] = data
        return self._buffer

    # alias for easier / more intuitive calling
    buffer = process

    @staticmethod
    def add_arguments(parser, buffer_length=None):
        """
        Add buffering related arguments to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        buffer_length : int
            Length of the buffer.

        Returns
        -------
        argparse argument group
            Buffering argument parser group.

        """
        # add buffering options to the existing parser
        g = parser.add_argument_group('buffering arguments')
        if buffer_length is not None:
            g.add_argument('--buffer_length', action='store', type=int,
                           help='length of the buffer')
        return g


# function to process live input
def process_online(processor, stream, outfile, **kwargs):
    """
    Process a stream with the given Processor.

    Parameters
    ----------
    processor : :class:`Processor` instance
        Processor to be processed.
    stream : :class:`.audio.signal.Stream`
        Stream to get the data from. If 'None' a new stream is created with
        the additional keyword arguments.
    outfile : str or file handle
        Output file (handle).
    kwargs : dict, optional
        Keyword arguments passed to :class:`.audio.signal.Stream` if
        `in_stream` is 'None'.

    """
    from madmom.audio.signal import Stream
    # FIXME: If a Stream is given we must check if the arguments match the ones
    #        of the Processor. Maybe just always do the stream creation in here
    #        and infer the needed arguments from the Processor?
    if not isinstance(stream, Stream):
        stream = Stream(**kwargs)
    # start the stream if not running already
    if not stream.is_running():
        stream.start()
    # process all frames with the given processor
    for frame in stream:
        if isinstance(processor, IOProcessor):
            processor(frame, outfile, **kwargs)
        else:
            processor(frame, **kwargs)


# function for pickling a processor
def pickle_processor(processor, outfile, **kwargs):
    """
    Pickle the Processor to a file.

    Parameters
    ----------
    processor : :class:`Processor` instance
        Processor to be pickled.
    outfile : str or file handle
        Output file (handle) where to pickle it.

    """
    # pylint: disable=unused-argument
    processor.dump(outfile)


# generic input/output arguments for scripts
def io_arguments(parser, output_suffix='.txt', pickle=True, online=False):
    """
    Add input / output related arguments to an existing parser.

    Parameters
    ----------
    parser : argparse parser instance
        Existing argparse parser object.
    output_suffix : str, optional
        Suffix appended to the output files.
    pickle : bool, optional
        Add a 'pickle' sub-parser to the parser?

    """
    # default output
    try:
        output = sys.stdout.buffer
    except AttributeError:
        output = sys.stdout
    # add general options
    parser.add_argument('-v', dest='verbose', action='count',
                        help='increase verbosity level')
    # add subparsers
    sub_parsers = parser.add_subparsers(title='processing options')

    # pickle processor options
    if pickle:
        sp = sub_parsers.add_parser('pickle', help='pickle processor')
        sp.set_defaults(func=pickle_processor)
        # Note: requiring '-o' is a simple safety measure to not overwrite
        #       existing audio files after using the processor in 'batch' mode
        sp.add_argument('-o', dest='outfile', type=argparse.FileType('wb'),
                        default=output, help='output file [default: STDOUT]')

    # single file processing options
    sp = sub_parsers.add_parser('single', help='single file processing')
    sp.set_defaults(func=process_single)
    sp.add_argument('infile', type=argparse.FileType('rb'),
                    help='input audio file')
    # Note: requiring '-o' is a simple safety measure to not overwrite existing
    #       audio files after using the processor in 'batch' mode
    sp.add_argument('-o', dest='outfile', type=argparse.FileType('wb'),
                    default=output, help='output file [default: STDOUT]')
    sp.add_argument('-j', dest='num_threads', type=int, default=mp.cpu_count(),
                    help='number of parallel threads [default=%(default)s]')

    # batch file processing options
    sp = sub_parsers.add_parser('batch', help='batch file processing')
    sp.set_defaults(func=process_batch)
    sp.add_argument('files', nargs='+', help='files to be processed')
    sp.add_argument('-o', dest='output_dir', default=None,
                    help='output directory [default=%(default)s]')
    sp.add_argument('-s', dest='output_suffix', default=output_suffix,
                    help='suffix appended to the files (dot must be included '
                         'if wanted) [default=%(default)s]')
    sp.add_argument('--ext', dest='strip_ext', action='store_false',
                    help='keep the extension of the input file [default='
                         'strip it off before appending the output suffix]')
    sp.add_argument('-j', dest='num_workers', type=int, default=mp.cpu_count(),
                    help='number of parallel workers [default=%(default)s]')
    sp.add_argument('--shuffle', action='store_true',
                    help='shuffle files before distributing them to the '
                         'working threads [default=process them in sorted '
                         'order]')
    sp.set_defaults(num_threads=1)

    # online processing options
    if online:
        sp = sub_parsers.add_parser('online', help='online processing')
        sp.set_defaults(func=process_online)
        # Note: requiring '-o' is a simple safety measure to not overwrite
        #       existing audio files after using the processor in 'batch' mode
        sp.add_argument('-o', dest='outfile', type=argparse.FileType('wb'),
                        default=output, help='output file [default: STDOUT]')
        sp.add_argument('--block_size', dest='block_size', type=int,
                        default=1, help='number of frames used for processing')
        sp.set_defaults(sample_rate=44100)
        sp.set_defaults(num_channels=1)
        sp.set_defaults(origin='future')
        sp.set_defaults(num_frames=1)
        sp.set_defaults(stream=None)
        sp.set_defaults(online=True)
