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

import argparse
import itertools as it
import multiprocessing as mp
import os
import sys
from collections import MutableSequence

import numpy as np

from .utils import integer_types


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
        from .io import open_file
        # instantiate a new Processor and return it
        with open_file(infile, 'rb') as f:
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
        from .io import open_file
        # dump the Processor to the given file
        # Note: for Python 2 / 3 compatibility reason use protocol 2
        with open_file(outfile, 'wb') as f:
            pickle.dump(self, f, protocol=2)

    def process(self, data, **kwargs):
        """
        Process the data.

        This method must be implemented by the derived class and should
        process the given data and return the processed output.

        Parameters
        ----------
        data : depends on the implementation of subclass
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the implementation of subclass
            Processed data.

        """
        raise NotImplementedError('Must be implemented by subclass.')

    def __call__(self, *args, **kwargs):
        # this magic method makes a Processor callable
        return self.process(*args, **kwargs)


class OnlineProcessor(Processor):
    """
    Abstract base class for processing data in online mode.

    Derived classes must implement the following methods:

    - process_online(): process the data in online mode,
    - process_offline(): process the data in offline mode.

    """

    def __init__(self, online=False):
        self.online = online

    def process(self, data, **kwargs):
        """
        Process the data either in online or offline mode.

        Parameters
        ----------
        data : depends on the implementation of subclass
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the implementation of subclass
            Processed data.

        Notes
        -----
        This method is used to pass the data to either `process_online` or
        `process_offline`, depending on the `online` setting of the processor.

        """
        if self.online:
            return self.process_online(data, **kwargs)
        return self.process_offline(data, **kwargs)

    def process_online(self, data, reset=True, **kwargs):
        """
        Process the data in online mode.

        This method must be implemented by the derived class and should process
        the given data frame by frame and return the processed output.

        Parameters
        ----------
        data : depends on the implementation of subclass
            Data to be processed.
        reset : bool, optional
            Reset the processor to its initial state before processing.
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the implementation of subclass
            Processed data.

        """
        raise NotImplementedError('Must be implemented by subclass.')

    def process_offline(self, data, **kwargs):
        """
        Process the data in offline mode.

        This method must be implemented by the derived class and should process
        the given data and return the processed output.

        Parameters
        ----------
        data : depends on the implementation of subclass
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the implementation of subclass
            Processed data.

        """
        raise NotImplementedError('Must be implemented by subclass.')

    def reset(self):
        """
        Reset the OnlineProcessor.

        This method must be implemented by the derived class and should reset
        the processor to its initial state.

        """
        raise NotImplementedError('Must be implemented by subclass.')


class OutputProcessor(Processor):
    """
    Class for processing data and/or feeding it into some sort of output.

    """

    def process(self, data, output, **kwargs):
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
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the implementation of subclass
            Processed data.

        """
        # pylint: disable=arguments-differ
        raise NotImplementedError('Must be implemented by subclass.')


# functions for processing file(s) with a Processor
def _process(process_tuple):
    """
    Function to process a Processor with data.

    The processed data is returned and if applicable also piped to the given
    output.

    Parameters
    ----------
    process_tuple : tuple (Processor/function, data[, output], kwargs)

        The tuple must contain a Processor object as the first item and the
        data to be processed as the second tuple item. If a third tuple item
        is given, it is used as an output argument. The last item is passed
        as keyword arguments to the processor's process() method.
        Instead of a Processor also a function accepting a single positional
        argument (data) or two positional arguments (data, output) can be
        given. It must behave exactly as a :class:`Processor`, i.e. return
        the processed data and optionally pipe it to the output. Keyword
        arguments are not passed to the function.

    Returns
    -------
    depends on the processor
        Processed data.

    Notes
    -----
    This must be a top-level function to be pickle-able.

    """
    # do not process the data, if the first item (i.e. Processor) is None
    if process_tuple[0] is None:
        return process_tuple[1]
    # call the Processor with data and kwargs
    elif isinstance(process_tuple[0], Processor):
        return process_tuple[0](*process_tuple[1:-1], **process_tuple[-1])
    # just call whatever we got here (e.g. a function) without kwargs
    return process_tuple[0](*process_tuple[1:-1])


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
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the last processor of the processing chain
            Processed data.

        """
        # sequentially process the data
        for processor in self.processors:
            data = _process((processor, data, kwargs))
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

    def process(self, data, **kwargs):
        """
        Process the data in parallel.

        Parameters
        ----------
        data : depends on the processors
            Data to be processed.
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        list
            Processed data.

        """
        # if only a single processor is given, there's no need to map()
        if len(self.processors) == 1:
            return [_process((self.processors[0], data, kwargs))]
        # process data in parallel and return a list with processed data
        return list(self.map(_process, zip(self.processors, it.repeat(data),
                                           it.repeat(kwargs))))


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
        if isinstance(out_processor, (list, tuple)):
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
        kwargs : dict, optional
            Keyword arguments for processing.

        Returns
        -------
        depends on the output processors
            Processed data.

        """
        # process the data by the input processor
        data = _process((self.in_processor, data, kwargs))
        # process the data by the output processor and return it
        return _process((self.out_processor, data, output, kwargs))


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
    # adjust origin in online mode
    if kwargs.get('online'):
        kwargs['origin'] = 'online'
        kwargs['reset'] = False
    # process the input file
    _process((processor, infile, outfile, kwargs))


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
            processor, infile, outfile, kwargs = self.task_queue.get()
            try:
                # process the Processor with the data
                _process((processor, infile, outfile, kwargs))
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
        tasks.put((processor, input_file, output_file, kwargs))
    # wait for all processing tasks to finish
    tasks.join()


# processor for buffering data
class BufferProcessor(Processor):
    """
    Buffer for processors which need context to do their processing.

    Parameters
    ----------
    buffer_size : int or tuple
        Size of the buffer (time steps, [additional dimensions]).
    init : numpy array, optional
        Init the buffer with this array.
    init_value : float, optional
        If only `buffer_size` is given but no `init`, use this value to
        initialise the buffer.

    Notes
    -----
    If `buffer_size` (or the first item thereof in case of tuple) is 1,
    only the un-buffered current value is returned.

    If context is needed, `buffer_size` must be set to >1.
    E.g. SpectrogramDifference needs a context of two frames to be able to
    compute the difference between two consecutive frames.

    """

    def __init__(self, buffer_size=None, init=None, init_value=0):
        # if init is given, infer buffer_size from it
        if buffer_size is None and init is not None:
            buffer_size = init.shape
        # if buffer_size is int, make a tuple
        elif isinstance(buffer_size, integer_types):
            buffer_size = (buffer_size, )
        # TODO: use np.pad for fancy initialisation (can be done in process())
        # init buffer if needed
        if buffer_size is not None and init is None:
            init = np.ones(buffer_size) * init_value
        # save variables
        self.buffer_size = buffer_size
        self.init = init
        self.data = init

    def reset(self, init=None):
        """
        Reset BufferProcessor to its initial state.

        Parameters
        ----------
        init : numpy array, shape (num_hiddens,), optional
            Reset BufferProcessor to this initial state.

        """
        self.data = init if init is not None else self.init

    def process(self, data, **kwargs):
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
        # expected minimum number of dimensions
        ndmin = len(self.buffer_size)
        # cast the data to have that many dimensions
        if data.ndim < ndmin:
            data = np.array(data, copy=False, subok=True, ndmin=ndmin)
        # length of the data
        data_length = len(data)
        # remove `data_length` from buffer at the beginning and append new data
        self.data = np.roll(self.data, -data_length, axis=0)
        self.data[-data_length:] = data
        # return the complete buffer
        return self.data

    # alias for easier / more intuitive calling
    buffer = process

    def __getitem__(self, index):
        """
        Direct access to the buffer data.

        Parameters
        ----------
        index : int, slice, ndarray,
            Any NumPy indexing method to access the buffer data directly.

        Returns
        -------
        numpy array or subclass thereof
            Requested view of the buffered data.

        """
        return self.data[index]


# function to process live input
def process_online(processor, infile, outfile, **kwargs):
    """
    Process a file or audio stream with the given Processor.

    Parameters
    ----------
    processor : :class:`Processor` instance
        Processor to be processed.
    infile : str or file handle, optional
        Input file (handle). If none is given, the stream present at the
        system's audio inpup is used. Additional keyword arguments can be used
        to influence the frame size and hop size.
    outfile : str or file handle
        Output file (handle).
    kwargs : dict, optional
        Keyword arguments passed to :class:`.audio.signal.Stream` if
        `in_stream` is 'None'.

    Notes
    -----
    Right now there is no way to determine if a processor is online-capable or
    not. Thus, calling any processor with this function may not produce the
    results expected.

    """
    from madmom.audio.signal import Stream, FramedSignal
    # set default values
    kwargs['sample_rate'] = kwargs.get('sample_rate', 44100)
    kwargs['num_channels'] = kwargs.get('num_channels', 1)
    # if no iput file is given, create a Stream with the given arguments
    if infile is None:
        # open a stream and start if not running already
        stream = Stream(**kwargs)
        if not stream.is_running():
            stream.start()
    # use the input file
    else:
        # set parameters for opening the file
        from .audio.signal import FRAME_SIZE, HOP_SIZE, FPS, NUM_CHANNELS
        frame_size = kwargs.get('frame_size', FRAME_SIZE)
        hop_size = kwargs.get('hop_size', HOP_SIZE)
        fps = kwargs.get('fps', FPS)
        num_channels = kwargs.get('num_channels', NUM_CHANNELS)
        # FIXME: overwrite the frame size with the maximum value of all used
        #        processors. This is needed if multiple frame sizes are used
        import warnings
        warnings.warn('make sure that the `frame_size` (%d) is equal to the '
                      'maximum value used by any `FramedSignalProcessor`.' %
                      frame_size)
        # Note: origin must be 'online' and num_frames 'None' to behave exactly
        #       the same as with live input
        stream = FramedSignal(infile, frame_size=frame_size, hop_size=hop_size,
                              fps=fps, origin='online', num_frames=None,
                              num_channels=num_channels)
    # set arguments for online processing
    # Note: pass only certain arguments, because these will be passed to the
    #       processors at every time step (kwargs contains file handles etc.)
    process_args = {'reset': False}  # do not reset stateful processors
    # process everything frame-by-frame
    for frame in stream:
        _process((processor, frame, outfile, process_args))


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
        Add a 'pickle' sub-parser to the parser.
    online : bool, optional
        Add a 'online' sub-parser to the parser.

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
                    help='number of threads [default=%(default)s]')
    # add arguments needed for loading processors
    if online:
        sp.add_argument('--online', action='store_true', default=None,
                        help='use online settings [default: offline]')

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
                    help='number of workers [default=%(default)s]')
    sp.add_argument('--shuffle', action='store_true',
                    help='shuffle files before distributing them to the '
                         'working threads [default=process them in sorted '
                         'order]')
    sp.set_defaults(num_threads=1)

    # online processing options
    if online:
        sp = sub_parsers.add_parser('online', help='online processing')
        sp.set_defaults(func=process_online)
        sp.add_argument('infile', nargs='?', type=argparse.FileType('rb'),
                        default=None, help='input audio file (if no file is '
                                           'given, a stream operating on the '
                                           'system audio input is used)')
        sp.add_argument('-o', dest='outfile', type=argparse.FileType('wb'),
                        default=output, help='output file [default: STDOUT]')
        sp.add_argument('-j', dest='num_threads', type=int, default=1,
                        help='number of threads [default=%(default)s]')
        # set arguments for loading processors
        sp.set_defaults(online=True)      # use online settings/parameters
        sp.set_defaults(num_frames=1)     # process everything frame-by-frame
        sp.set_defaults(origin='stream')  # set origin to get whole frame
