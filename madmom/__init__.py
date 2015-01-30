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
import os
import abc

from madmom.utils import open


MODELS_PATH = '%s/models' % (os.path.dirname(__file__))


class Processor(object):
    """
    Abstract base class for all kind of processing objects.

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
        # instantiate a new object and return it
        return cPickle.load(open(infile))

    def save(self, outfile):
        """
        Save the Processor to a file.

        This method pickles a Processor object and saves it. Subclasses should
        overwrite this method with a better performing solution if speed is an
        issue.

        :param outfile: output file name or file handle

        """
        import cPickle
        # save the Processor object to a file
        cPickle.dump(self, open(outfile, 'rw'))

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


class DummyProcessor(Processor):
    """
    Dummy processor which just passes the data.

    """

    def process(self, data):
        """
        Just passes the data.

        :param data: data to be returned
        :return:     data

        """
        return data


class SequentialProcessor(Processor):
    """
    Base class for sequential processing of data.

    """
    def __init__(self, processors):
        """
        Instantiates a SequentialProcessor object.

        :param processors: list with Processor objects

        """
        self.processors = processors

    def process(self, data):
        """
        Process the data sequentially.

        :param data: data to be processed
        :return:     processed data

        """
        # sequentially process the data
        for processor in self.processors:
            data = processor.process(data)
        return data


def _process(process_tuple):
    """
    Function to process a Processor object (first tuple item) with the given
    data (second tuple item).

    :param process_tuple: tuple (processing, data)
    :return:              processed data

    Note: this must be a top-level function to be pickle-able.

    """
    return process_tuple[0].process(process_tuple[1])


class ParallelProcessor(Processor):
    """
    Base class for parallel processing of data.

    """
    import multiprocessing as mp
    NUM_THREADS = mp.cpu_count()

    def __init__(self, processors, num_threads=NUM_THREADS):
        """
        Instantiates a ParallelProcessor object.

        :param processors:  list with processing objects
        :param num_threads: number of parallel working threads

        """
        # save the processing queue
        self.processors = processors
        # number of threads
        if num_threads is None:
            num_threads = self.NUM_THREADS
        self.num_threads = num_threads

    def process(self, data, num_threads=None):
        """
        Process the data in parallel.

        :param data:        data to be processed
        :param num_threads: number of parallel working threads
        :return:            list with individually processed data

        """
        import multiprocessing as mp
        import itertools as it
        # number of threads
        if num_threads is None:
            num_threads = self.num_threads
        # init a pool of workers (if needed)
        map_ = map
        if min(len(self.processors), max(1, num_threads)) != 1:
            map_ = mp.Pool(num_threads).map
        # process data in parallel and return a list with processed data
        return map_(_process, it.izip(self.processors, it.repeat(data)))

    @classmethod
    def add_arguments(cls, parser, num_threads=NUM_THREADS):
        """
        Add parallel processing options to an existing parser object.

        :param parser:      existing argparse parser object
        :param num_threads: number of threads to run in parallel [int]
        :return:            parallel processing argument parser group

        Parameters are included in the group only if they are not 'None'.

        """
        # add parallel processing options
        g = parser.add_argument_group('parallel processing arguments')
        g.add_argument('-j', '--threads', dest='num_threads', action='store',
                       type=int, default=num_threads,
                       help='number of parallel threads [default=%(default)s]')
        # return the argument group so it can be modified if needed
        return g
