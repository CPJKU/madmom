#!/usr/bin/env python
# encoding: utf-8
"""
This file contains all functionality needed for interaction with RNNLIB.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import numpy as np

import os.path
import shutil
import tempfile

from Queue import Queue
from threading import Thread

# rnnlib binary, please see the README file
RNNLIB = 'rnnlib'


# Activations
class Activations(np.ndarray):
    """
    Class for reading in activations as written by RNNLIB.

    """
    def __new__(cls, filename, labels=None, fps=None):
        # read in the file
        label = 0
        with open(filename, 'r') as f:
            activations = None
            for line in f:
                # read in the header
                if line.startswith('#'):
                    continue
                if line.startswith('LABEL'):
                    labels = [line.split(": ")[1].split()]
                    continue
                if line.startswith('DIMENSION'):
                    dimensions = int(line.split(": ")[1])
                    # init the matrix
                    if labels:
                        activations = np.zeros((dimensions, len(labels)))
                    else:
                        activations = np.zeros(dimensions)
                    continue
                # make sure we have an activations array
                if activations is None:
                    raise AssertionError('no activations initialised')
                # read in the data
                if labels:
                    activations[:, label] = np.fromstring(line, sep=' ')
                    # increase counter
                    label += 1
                else:
                    activations = np.fromstring(line, sep=' ')
        # close the file
        f.close()
        # cast to Activations
        obj = np.asarray(activations).view(cls)
        # set file name
        # TODO: do we need the file name? remove it?
        obj.__filename = filename
        # set attributes
        obj.__labels = labels
        obj.__fps = fps
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.__labels = getattr(obj, '__labels', None)
        self.__fps = getattr(obj, '__fps', None)
        # TODO: do we need the file name? remove it?
        self.__filename = getattr(obj, '__filename', None)

    @property
    def labels(self):
        """Labels for classes."""
        return self.__labels

    @property
    def filename(self):
        """Name of the activations file."""
        return self.__filename


# helper functions for .nc file creation
def max_len(strings):
    """Determine the maximum length of an array of the given strings."""
    return len(max(strings, key=len))


def expand_and_terminate(strings):
    """Expand and null-terminate the given strings."""
    # each string must have the same length and must be null-terminated
    terminated_strings = []
    max_length = max_len(strings) + 1
    for string in strings:
        # expand each string to the maximum length with \0's
        string += '\0' * (max_length - len(string))
        terminated_strings.append(string)
    return terminated_strings


# FIXME: if we inherit from scipy.io.netcdf.NetCDFFile and omit the self.nc
# stuff and try to extend the class with properties directly, the setters do not
# work! why?
class NetCDF(object):
    """
    NetCDF Class is a simple NetCDFFile wrapper with some extensions for use
    with RNNLIB.

    """
    def __init__(self, filename, mode):
        """
        Creates a new NetCDF object.

        :param filename: open a .nc-file with the given filename
        :param mode: r/w open an existing file for reading/writing

        Note: The scipy module has a bug which let's you alter variables and
              dimensions if the file is opened in 'r' mode, but does not throw
              an error if the flush() or close() methods are called and does
              not save the altered values!

        """
        from scipy.io.netcdf import NetCDFFile
        # open the file
        self.nc = NetCDFFile(filename, mode)

    def close(self):
        """Closes the file."""
        self.nc.close()

    def flush(self):
        """Flushes the file to disk."""
        self.nc.flush()

    @property
    def filename(self):
        """Name of the file."""
        return self.nc.filename

    @property
    def numSeqs(self):
        """Number of sequences."""
        # mandatory
        try:
            return self.nc.dimensions['numSeqs']
        except KeyError:
            return 0

    @numSeqs.setter
    def numSeqs(self, numSeqs):
        try:
            self.nc.createDimension('numSeqs', numSeqs)
        except IOError:
            self.nc.dimensions['numSeqs'] = numSeqs

    @property
    def numDims(self):
        """Number of dimensions of the sequences."""
        # mandatory
        try:
            return self.nc.dimensions['numDims']
        except KeyError:
            return 0

    @numDims.setter
    def numDims(self, numDims):
        try:
            self.nc.createDimension('numDims', numDims)
        except IOError:
            self.nc.dimensions['numDims'] = numDims

    @property
    def numTimesteps(self):
        """Number of timesteps (of all sequences)."""
        # mandatory
        try:
            return self.nc.dimensions['numTimesteps']
        except KeyError:
            return 0

    @numTimesteps.setter
    def numTimesteps(self, numTimesteps):
        try:
            self.nc.createDimension('numTimesteps', numTimesteps)
        except IOError:
            self.nc.dimensions['numTimesteps'] = numTimesteps

    @property
    def inputPattSize(self):
        """Size of the input pattern."""
        # mandatory
        try:
            return self.nc.dimensions['inputPattSize']
        except KeyError:
            return None

    @inputPattSize.setter
    def inputPattSize(self, inputPattSize):
        try:
            self.nc.createDimension('inputPattSize', inputPattSize)
        except IOError:
            self.nc.dimensions['inputPattSize'] = inputPattSize

    # dimensions needed for indicated tasks
    # TODO: use decorators to check the presence of these depending on the task

    @property
    def numLabels(self):
        """Number of different labels."""
        # classification, sequence_classification, transcription
        try:
            return self.nc.dimensions['numLabels']
        except KeyError:
            return None

    @numLabels.setter
    def numLabels(self, numLabels):
        try:
            self.nc.createDimension('numLabels', numLabels)
        except IOError:
            self.nc.dimensions['numLabels'] = numLabels

    @property
    def maxLabelLength(self):
        """Maximum length of the labels."""
        # classification, sequence_classification, transcription
        try:
            return self.nc.dimensions['maxLabelLength']
        except KeyError:
            return None

    @maxLabelLength.setter
    def maxLabelLength(self, maxLabelLength):
        try:
            self.nc.createDimension('maxLabelLength', maxLabelLength)
        except IOError:
            self.nc.dimensions['maxLabelLength'] = maxLabelLength

    @property
    def targetPattSize(self):
        """Size of the target pattern vector."""
        # regression
        try:
            return self.nc.dimensions['targetPattSize']
        except KeyError:
            return None

    @targetPattSize.setter
    def targetPattSize(self, targetPattSize):
        try:
            self.nc.createDimension('targetPattSize', targetPattSize)
        except IOError:
            self.nc.dimensions['targetPattSize'] = targetPattSize

    @property
    def maxTargStringLength(self):
        """Maximum length of the target strings."""
        # sequence_classification, transcription
        try:
            return self.nc.dimensions['maxTargStringLength']
        except KeyError:
            return None

    @maxTargStringLength.setter
    def maxTargStringLength(self, maxTargStringLength):
        try:
            self.nc.createDimension('maxTargStringLength', maxTargStringLength)
        except IOError:
            self.nc.dimensions['maxTargStringLength'] = maxTargStringLength

    @property
    def maxSeqTagLength(self):
        """Maximum length of the sequence tags."""
        # optional
        try:
            return self.nc.dimensions['maxSeqTagLength']
        except KeyError:
            return None

    @maxSeqTagLength.setter
    def maxSeqTagLength(self, maxSeqTagLength):
        try:
            self.nc.createDimension('maxSeqTagLength', maxSeqTagLength)
        except IOError:
            self.nc.dimensions['maxSeqTagLength'] = maxSeqTagLength

    # VARIABLES

    @property
    def inputs(self):
        """Array of input vectors [float]."""
        # mandatory
        try:
            var = self.nc.variables['inputs']
            #return var.getValue()
            return var.data
        except KeyError:
            return None

    @inputs.setter
    def inputs(self, inputs):
        inputs = np.atleast_2d(inputs)
        # set the seqDims of not already done (e.g. because of multiple sequences)
        if not self.seqDims:
            self.seqDims = [np.shape(inputs)[0]]
        if not self.numTimesteps:
            self.numTimesteps = np.shape(inputs)[0]
        if not self.inputPattSize:
            self.inputPattSize = np.shape(inputs)[1]
        var = self.nc.createVariable('inputs', 'f', ('numTimesteps', 'inputPattSize'))
        var[:] = inputs.astype(np.float32)

    @property
    def seqDims(self):
        """Array of sequence dimensions [int]."""
        # mandatory
        try:
            var = self.nc.variables['seqDims']
            #return var.getValue()
            return var.data
        except KeyError:
            return None

    @seqDims.setter
    def seqDims(self, seqDims):
        seqDims = np.atleast_2d(seqDims)
        if not self.numSeqs:
            self.numSeqs = np.shape(seqDims)[0]
        if not self.numDims:
            self.numDims = np.shape(seqDims)[1]
        var = self.nc.createVariable('seqDims', 'i', ('numSeqs', 'numDims'))
        var[:] = seqDims.astype(np.int32)

    # variables needed for indicated tasks
    # TODO: use decorators to check the presence of these depending on the task

    @property
    def targetClasses(self):
        """Array of target classes [int]."""
        # classification
        try:
            var = self.nc.variables['targetClasses']
            return var.data
        except KeyError:
            return None

    @targetClasses.setter
    def targetClasses(self, targetClasses):
        if not self.numTimesteps:
            self.numTimesteps = np.shape(targetClasses)[0]
        if not self.labels:
            self.labels = np.unique(targetClasses)
        var = self.nc.createVariable('targetClasses', 'i', ('numTimesteps',))
        var[:] = targetClasses.astype(np.int64)

    @property
    def labels(self):
        """Array of labels [char]."""
        # classification, sequence_classification, transcription
        try:
            var = self.nc.variables['labels']
            return var.data
        except KeyError:
            return None

    @labels.setter
    def labels(self, labels):
        # TODO: make a list if a single value is given?
        # convert the labels to a integer array
        labels = np.asarray(labels, np.int)
        # convert the labels to a strings array
        labels = np.asarray(labels, np.str)
        # set the number of labels
        if not self.numLabels:
            self.numLabels = np.shape(labels)[0]
        # set the maximum length of the labels
        if not self.maxLabelLength:
            # set the maximum length of the label names
            self.maxLabelLength = max_len(labels) + 1
        # all labels must be the same length and null-terminated
        labels = expand_and_terminate(labels)
        var = self.nc.createVariable('labels', 'c', ('numLabels', 'maxLabelLength'))
        var[:] = labels

    @property
    def targetPatterns(self):
        """Array of target patterns [float]."""
        # regression
        try:
            var = self.nc.variables['targetPatterns']
            return var.data
        except KeyError:
            return None

    @targetPatterns.setter
    def targetPatterns(self, targetPatterns):
        # TODO: make a list if a single value is given?
        if not self.numTimesteps:
            self.numTimesteps = np.shape(targetPatterns)[0]
        if not self.targetPattSize:
            self.targetPattSize = np.shape(targetPatterns)[1]
        var = self.nc.createVariable('targetPatterns', 'f', ('numTimesteps', 'targetPattSize'))
        var[:] = targetPatterns.view(np.float32)

    @property
    def targetStrings(self):
        """Array of targetStrings [char]."""
        # sequence_classification, transcription
        try:
            var = self.nc.variables['targetStrings']
            return var.data
        except KeyError:
            return None

    @targetStrings.setter
    def targetStrings(self, targetStrings):
        # TODO: make a list if a single value is given?
        if not self.numSeqs:
            self.numSeqs = len(targetStrings)
        if not self.maxTargStringLength:
            self.maxTargStringLength = max_len(targetStrings) + 1
        # all targetStrings must be the same length and null-terminated
        targetStrings = expand_and_terminate(targetStrings)
        var = self.nc.createVariable('targetStrings', 'c', ('numTimesteps', 'maxTargStringLength'))
        var[:] = targetStrings

    @property
    def seqTags(self):
        """Array of sequence tags [char]."""
        # optional
        try:
            var = self.nc.variables['seqTags']
            #return var.getValue()
            return var.data
        except KeyError:
            return None

    @seqTags.setter
    def seqTags(self, seqTags):
        # make a list if a single value is given
        if isinstance(seqTags, str):
            seqTags = [seqTags]
        if not self.numSeqs:
            self.numSeqs = len(seqTags)
        if not self.maxSeqTagLength:
            self.maxSeqTagLength = max_len(seqTags) + 1
        # all seqTags must be the same length and null-terminated
        seqTags = expand_and_terminate(seqTags)
        var = self.nc.createVariable('seqTags', 'c', ('numSeqs', 'maxSeqTagLength'))
        var[:] = seqTags


# .nc file creation
def create_nc_file(filename, data, targets, tags=None):
    """
    Create a .nc file with the given input data and targets.

    :param filename: name of the file to create
    :param data:     input data
    :param targets:  corresponding targets
    :param tags:     additional [default=None]

    """
    # create the .nc file
    nc = NetCDF(filename, 'w')
    # input data handling
    if isinstance(data, np.ndarray):
        # data in correct format
        nc.inputs = data
    elif isinstance(data, list):
        # we need to stack the data
        inputs = None
        for d in data:
            if inputs is None:
                # use first as is
                inputs = d
            else:
                # stack all others
                inputs = np.hstack((inputs, d))
        # store them in .nc file
        nc.inputs = inputs
    else:
        raise TypeError("Invalid input data type.")
    # groud truth
    # FIXME: expand this also to patterns (regression)
    nc.targetClasses = targets
    # tags
    if tags:
        nc.seqTags = str(tags)
    # save file
    nc.close()
    # return
    # TODO: return a tuple (fd + filename)?
    return filename


# .nc file testing
class TestThread(Thread):
    def __init__(self, work_queue, return_queue, verbose=False):
        """
        Test a file against multiple neural networks.

        :param work_queue:   queue with work items
        :param return_queue: queue for the results
        :param verbose:      show RNNLIB's output [default=False]

        """
        # init the thread
        super(TestThread, self).__init__()
        #Thread.__init__(self)
        # set attributes
        self.work_queue = work_queue
        self.return_queue = return_queue
        self.verbose = verbose

    def run(self):
        """Test file against all neural networks in the queue."""
        import subprocess
        while True:
            # grab the first work item from queue
            nc_file, nn_file = self.work_queue.get()
            # create a tmp directory for each thread
            tmp_work_path = tempfile.mkdtemp()
            # test the file against the network
            args = [RNNLIB,
                    '--verbose=true',
                    '--display=true',
                    '--autosave=false',
                    '--dumpPath=%s/' % tmp_work_path,
                    '--dataset=test',
                    '--dataFileNum=0',
                    '--sequence=0',
                    '--trainFile=""',
                    '--valFile=""',
                    '--testFile=%s' % nc_file, nn_file]
            try:
                if self.verbose:
                    subprocess.call(args)
                else:
                    devnull = open(os.devnull, 'w')
                    subprocess.call(args, stdout=devnull, stderr=devnull)
            except OSError:
                # TODO: which exception should be raised?
                raise SystemExit('rnnlib binary not found')
            # read the activations
            act = np.empty(0)
            try:
                # classification output
                act = Activations('%s/output_outputActivations' % tmp_work_path)
                # TODO: make regression task work as well
            except IOError:
                # could not read in the activations, try regression
                pass
            # put a tuple with nc file, nn file and activations in the return queue
            self.return_queue.put((nc_file, nn_file, act))
            # ok, clean up
            shutil.rmtree(tmp_work_path)
            # signal to queue that job is done
            self.work_queue.task_done()


def test_nc_files(nc_files, nn_files, threads=2, verbose=False):
    """
    Test a list of .nc files against multiple neural networks.

    :param nc_files: list with .nc files to be tested
    :param nn_files: list with network files
    :param threads:  number of parallel threads [default=2]
    :param verbose:  be verbose [default=False]

    """
    if not nc_files:
        raise ValueError('no .nc files given')
    if not nn_files:
        raise ValueError('no pre-trained neural network files given')
    # a queue for the work items
    work_queue = Queue()
    # a queue for the results
    return_queue = Queue()
    # start N threads parallel
    for _ in range(threads):
        t = TestThread(work_queue, return_queue, verbose)
        t.setDaemon(True)
        t.start()
    # put a combination of .nc files and neural networks in the queue
    for nc_file in nc_files:
        for nn_file in nn_files:
            work_queue.put((nc_file, nn_file))
    # wait until everything has been processed
    work_queue.join()
    # init return list
    activations = [None] * len(nc_files)
    num_activations = [1] * len(nc_files)
    # get all the activations and process the accordingly
    while not return_queue.empty():
        # get the tuple
        nc_file, nn_file, act = return_queue.get()
        # at which index should we put the activations in the return list
        nc_idx = nc_files.index(nc_file)
        # copy the activations to the returning list
        if activations[nc_idx] is None:
            # store the activations
            activations[nc_idx] = act
        else:
            # add the activations to the existing ones
            activations[nc_idx] += act
            # increase counter
            num_activations[nc_idx] += 1
    # average the activations
    for i in range(len(activations)):
        if num_activations[i] > 0:
            activations[i] /= num_activations[i]
    # return activations
    return activations


class RnnConfig(object):
    """Rnnlib config file class."""

    def __init__(self, filename=None):
        """
        Creates a new RNNLIB object instance.

        :param filename: name of the config file for rnnlib

        """
        # attributes
        self.train_files = None
        self.val_files = None
        self.test_files = None
        # read in file if a file name is given
        self.filename = filename
        if filename:
            self.load(filename)

    def load(self, filename):
        """
        Load the configuration from file.

        :param filename: name of the configuration file

        """
        # open the config file
        f = open(filename, 'r')
        # read in every line
        for line in f.readlines():
            # store the file sets
            if line.startswith('trainFile'):
                self.train_files = line[:].split()[1].split(',')
            if line.startswith('valFile'):
                self.val_files = line[:].split()[1].split(',')
            if line.startswith('testFile'):
                self.test_files = line[:].split()[1].split(',')
        # close the file
        f.close()

    def test(self, out_dir=None, file_set='test', threads=2, sep=''):
        """
        Test the given set of files.

        :param out_dir:  output directory for activations [default=None]
        :param file_set: which set should be tested (train, val, test) [default='test']
        :param threads:  number of working threads [default=2]
        :param sep:      separator between activation values [default='']
        :returns:        the output directory

        Note: If given, out_dir must exist. If none is given, a standard output
              directory is created.

              Empty (“”) separator means the file should be treated as binary;
              spaces (” ”) in the separator match zero or more whitespace;
              separator consisting only of spaces must match at least one whitespace.

        """
        # if no output directory was given, use the name of the file + set
        if out_dir is None:
            out_dir = "%s.%s" % (os.path.splitext(self.filename)[0], file_set)
        # create output directory
        try:
            os.mkdir(out_dir)
        except OSError:
            # directory exists already, touch it to have a current modification date
            os.utime(out_dir, None)
        # test all files of the given set
        nc_files = eval("self.%s_files" % file_set)
        # test all files
        activations = test_nc_files(nc_files, [self.filename], threads)
        # save all activations
        for f in nc_files:
            # name of the activations file
            act_file = "%s/%s.activations" % (out_dir, os.path.basename(os.path.splitext(f)[0]))
            # position in the list
            f_idx = nc_files.index(f)
            # save
            activations[f_idx].tofile(act_file, sep)
        # return the output directory
        return out_dir

    def split_files(self, files, splitting=[0.75, 0.875]):
        """
        Split the files into 2 or 3 sets, depending on the given split points.

        :param splitting: an array of either 1 or 2 values (in the range of 0...1)
                          if 1 value is given, the files are split into 2 sets (training & validation)
                          if 2 values are given, the files are split into 3 sets (training, validation & test)

        """
        raise NotImplementedError


def test_save_files(nn_files, out_dir=None, file_set='test', threads=2, sep=''):
    """
    Test the given set of files.

    :param nn_files: list with network files
    :param out_dir:  output directory for activations
    :param file_set: which set should be tested (train, val, test) [default='test']
    :param threads:  number of working threads [default=2]
    :param sep:      separator between activation values [default='']

    Note: empty (“”) separator means the file should be treated as binary;
          spaces (” ”) in the separator match zero or more whitespace;
          separator consisting only of spaces must match at least one whitespace.

          If out_dir is set and multiple network files contain the same
          files, the activations get averaged.

    """
    if out_dir is None:
        # test all NN files individually
        for nn_file in nn_files:
            nn = RnnConfig(nn_file)
            nn.test(file_set=file_set, threads=threads, sep=sep)
    else:
        # average all activations and output them in the given directory
        try:
            # create output directory
            os.mkdir(out_dir)
        except OSError:
            # directory exists already
            pass
        # get a list of all .nc files
        nc_files = []
        for nn_file in nn_files:
            nc_files.extend(eval("RnnConfig(nn_file).%s_files" % file_set))
        # remove duplicates
        nc_files = list(set(nc_files))
        # test all .nc files against the NN files which have this file in the given set
        for nc_file in nc_files:
            # check in which NN files the .nc file is included
            nc_nn_files = []
            for nn_file in nn_files:
                if nc_file in eval("RnnConfig(nn_file).%s_files" % file_set):
                    nc_nn_files.append(nn_file)
            # test the .nc file against these networks
            activations = test_nc_files([nc_file], nc_nn_files, threads)
            # name of the activations file
            act_file = "%s/%s.activations" % (out_dir, os.path.basename(os.path.splitext(nc_file)[0]))
            # save the activations ( we only passed one .nc file, so it's the
            # first activation in the returned list)
            activations[0].tofile(act_file, sep)


def combine_activations(out_dir, in_dirs):
    pass
