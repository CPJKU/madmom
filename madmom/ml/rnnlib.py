#!/usr/bin/env python
# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments

"""
This file contains all functionality needed for interaction with RNNLIB.

You need a working RNNLIB which can be obtained here:
http://sourceforge.net/apps/mediawiki/rnnl/index.php?title=Main_Page
Please build the software and put the resulting binary somewhere in your $PATH
or set the binary location in the RNNLIB variable below.

Note: RNNLIB is rather slow, madmom.ml.rnn serves as a (faster) purely Python
      based replacement for testing previously trained neural networks. The
      network configurations can be converted for testing with madmom.ml.rnn by
      RnnConfig('trained_network.save').save_model('converted_file').

"""

import os.path
import re
import shutil
import tempfile
from Queue import Queue
from threading import Thread
import multiprocessing as mp
import subprocess

import numpy as np

from madmom.features import Activations
from madmom.ml.rnn import (REVERSE, tanh, FeedForwardLayer, RecurrentLayer,
                           BidirectionalLayer, LSTMLayer,
                           RecurrentNeuralNetwork)
from madmom.utils import search_files, match_file

# rnnlib binary, please see comment above
RNNLIB = 'rnnlib'
THREADS = mp.cpu_count() / 2


class RnnlibActivations(np.ndarray):
    """
    Class for reading in activations as written by RNNLIB.

    """
    # pylint: disable=no-init
    # pylint: disable=super-on-old-class
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init

    def __new__(cls, filename, fps=None):
        # default is only one label
        labels = [1]
        label = 0
        # read in the file
        with open(filename, 'r') as f:
            activations = None
            for line in f:
                # read in the header
                if line.startswith('#'):
                    continue
                if line.startswith('LABEL'):
                    labels = line.split(": ", 1)[1].split()
                    continue
                if line.startswith('DIMENSION'):
                    dimensions = int(line.split(": ", 1)[1])
                    # init the matrix
                    activations = np.zeros((dimensions, len(labels)))
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
        # cast to RnnlibActivations
        obj = np.asarray(activations.astype(np.float32)).view(cls)
        # set attributes
        obj.labels = labels
        obj.fps = fps
        # return the object
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # set default values here
        self.labels = getattr(obj, 'labels', None)
        self.fps = getattr(obj, 'fps', None)


# helper functions for .nc file creation
def max_len(strings):
    """
    Determine the maximum length of an array of the given strings.
    :param strings: list with strings
    :return:        maximum length of these strings

    """
    return len(max(strings, key=len))


def expand_and_terminate(strings):
    """
    Expand and null-terminate the given strings to a common length.

    :param strings: a list of strings
    :return:        expanded and null-terminated list of strings

    """
    # each string must have the same length and must be null-terminated
    terminated_strings = []
    max_length = max_len(strings) + 1
    for string in strings:
        # expand each string to the maximum length with \0's
        string += '\0' * (max_length - len(string))
        terminated_strings.append(string)
    return terminated_strings


# FIXME: if we inherit from scipy.io.netcdf.NetCDFFile and omit the self.nc
# stuff and try to extend the class with properties directly, the setters do
# not work! why?
# noinspection PyPep8Naming
class NetCDF(object):
    """
    NetCDF Class is a simple NetCDFFile wrapper with some extensions for use
    with RNNLIB.

    """

    def __init__(self, filename, mode):
        """
        Creates a new NetCDF object.

        :param filename: open a .nc-file with the given filename
        :param mode:     {'r', 'w'} open an existing file for reading / writing

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
    def num_sequences(self):
        """Number of sequences."""
        # mandatory
        try:
            return self.nc.dimensions['numSeqs']
        except KeyError:
            return 0

    @num_sequences.setter
    def num_sequences(self, num_sequences):
        """
        Set the number of sequences.

        :param num_sequences: number of sequences [int]

        """
        try:
            self.nc.createDimension('numSeqs', num_sequences)
        except IOError:
            self.nc.dimensions['numSeqs'] = num_sequences

    @property
    def num_dimensions(self):
        """Number of dimensions of the sequences."""
        # mandatory
        try:
            return self.nc.dimensions['numDims']
        except KeyError:
            return 0

    @num_dimensions.setter
    def num_dimensions(self, num_dimensions):
        """
        Set the number of dimensions of the sequences.

        :param num_dimensions: number of dimensions of the sequences [int]

        """
        try:
            self.nc.createDimension('numDims', num_dimensions)
        except IOError:
            self.nc.dimensions['numDims'] = num_dimensions

    @property
    def num_time_steps(self):
        """Total number of time steps (of all sequences)."""
        # mandatory
        try:
            return self.nc.dimensions['numTimesteps']
        except KeyError:
            return 0

    @num_time_steps.setter
    def num_time_steps(self, num_time_steps):
        """
        Set the total number of time steps (of all sequences).

        :param num_time_steps: number of time steps [int]

        """
        try:
            self.nc.createDimension('numTimesteps', num_time_steps)
        except IOError:
            self.nc.dimensions['numTimesteps'] = num_time_steps

    @property
    def input_pattern_size(self):
        """Size of the input patterns."""
        # mandatory
        try:
            return self.nc.dimensions['inputPattSize']
        except KeyError:
            return None

    @input_pattern_size.setter
    def input_pattern_size(self, input_pattern_size):
        """
        Set the size of the input patterns.

        :param input_pattern_size: input pattern size [int]

        """
        try:
            self.nc.createDimension('inputPattSize', input_pattern_size)
        except IOError:
            self.nc.dimensions['inputPattSize'] = input_pattern_size

    # dimensions needed for indicated tasks
    # TODO: use decorators to check the presence of these depending on the task

    @property
    def num_labels(self):
        """Number of different labels."""
        # classification, sequence_classification, transcription
        try:
            return self.nc.dimensions['numLabels']
        except KeyError:
            return None

    @num_labels.setter
    def num_labels(self, num_labels):
        """
        Set the number of different labels (classification, sequence
        classification, transcription).

        :param num_labels: number of labels [int]

        """
        try:
            self.nc.createDimension('numLabels', num_labels)
        except IOError:
            self.nc.dimensions['numLabels'] = num_labels

    @property
    def max_label_length(self):
        """Maximum length of the labels."""
        # classification, sequence_classification, transcription
        try:
            return self.nc.dimensions['maxLabelLength']
        except KeyError:
            return None

    @max_label_length.setter
    def max_label_length(self, max_label_length):
        """
        Set the maximum label length (classification, sequence classification,
        transcription).

        :param max_label_length: maximum label length [int]

        """
        try:
            self.nc.createDimension('maxLabelLength', max_label_length)
        except IOError:
            self.nc.dimensions['maxLabelLength'] = max_label_length

    @property
    def target_pattern_size(self):
        """Size of the target pattern vector."""
        # regression
        try:
            return self.nc.dimensions['targetPattSize']
        except KeyError:
            return None

    @target_pattern_size.setter
    def target_pattern_size(self, target_pattern_size):
        """
        Set the size of the target pattern vector (regression).

        :param target_pattern_size: target pattern vector size

        """
        try:
            self.nc.createDimension('targetPattSize', target_pattern_size)
        except IOError:
            self.nc.dimensions['targetPattSize'] = target_pattern_size

    @property
    def max_target_string_length(self):
        """Maximum length of the target strings."""
        # sequence_classification, transcription
        try:
            return self.nc.dimensions['maxTargStringLength']
        except KeyError:
            return None

    @max_target_string_length.setter
    def max_target_string_length(self, max_target_string_length):
        """
        Set the maximum target strings length (sequence classification,
        transcription).

        :param max_target_string_length: maximum target strings length [int]

        """
        try:
            self.nc.createDimension('maxTargStringLength',
                                    max_target_string_length)
        except IOError:
            self.nc.dimensions['maxTargStringLength'] = \
                max_target_string_length

    @property
    def max_sequence_tag_length(self):
        """Maximum length of the sequence tags."""
        # optional
        try:
            return self.nc.dimensions['maxSeqTagLength']
        except KeyError:
            return None

    @max_sequence_tag_length.setter
    def max_sequence_tag_length(self, max_sequence_tag_length):
        """
        Set the maximum sequence tag length.

        :param max_sequence_tag_length: maximum sequence tag length [int]

        """
        try:
            self.nc.createDimension('maxSeqTagLength', max_sequence_tag_length)
        except IOError:
            self.nc.dimensions['maxSeqTagLength'] = max_sequence_tag_length

    # VARIABLES

    @property
    def inputs(self):
        """Input vectors."""
        # mandatory
        try:
            var = self.nc.variables['inputs']
            return var.data
        except KeyError:
            return None

    @inputs.setter
    def inputs(self, inputs):
        """
        Set the inputs.

        :param inputs: input vectors [float array]

        """
        inputs = np.atleast_2d(inputs)
        # set the seqDims if not already done
        if not self.sequence_dimensions:
            self.sequence_dimensions = [np.shape(inputs)[0]]
        if not self.num_time_steps:
            self.num_time_steps = np.shape(inputs)[0]
        if not self.input_pattern_size:
            self.input_pattern_size = np.shape(inputs)[1]
        var = self.nc.createVariable('inputs', 'f',
                                     ('numTimesteps', 'inputPattSize'))
        var[:] = inputs.astype(np.float32)

    @property
    def sequence_dimensions(self):
        """Sequence dimensions."""
        # mandatory
        try:
            var = self.nc.variables['seqDims']
            return var.data
        except KeyError:
            return None

    @sequence_dimensions.setter
    def sequence_dimensions(self, sequence_dimensions):
        """
        Set the sequence dimensions.

        :param sequence_dimensions: sequence dimensions [int array]

        """
        sequence_dimensions = np.atleast_2d(sequence_dimensions)
        if not self.num_sequences:
            self.num_sequences = np.shape(sequence_dimensions)[0]
        if not self.num_dimensions:
            self.num_dimensions = np.shape(sequence_dimensions)[1]
        var = self.nc.createVariable('seqDims', 'i', ('numSeqs', 'numDims'))
        var[:] = sequence_dimensions.astype(np.int32)

    # variables needed for indicated tasks
    # TODO: use decorators to check the presence of these depending on the task

    @property
    def target_classes(self):
        """Target classes."""
        # classification
        try:
            var = self.nc.variables['targetClasses']
            return var.data
        except KeyError:
            return None

    @target_classes.setter
    def target_classes(self, target_classes):
        """
        Set the target classes (classification).

        :param target_classes: target classes [array with class indices]

        """
        if not self.num_time_steps:
            self.num_time_steps = np.shape(target_classes)[0]
        if not self.labels:
            self.labels = np.unique(target_classes)
        var = self.nc.createVariable('targetClasses', 'i', ('numTimesteps',))
        var[:] = target_classes.astype(np.int64)

    @property
    def labels(self):
        """Labels."""
        # classification, sequence_classification, transcription
        try:
            var = self.nc.variables['labels']
            return var.data
        except KeyError:
            return None

    @labels.setter
    def labels(self, labels):
        """
        Set the labels (classification, sequence classification,
        transcription).

        :param labels: labels [list of strings]

        """
        # TODO: make a list if a single value is given?
        # convert the labels to a integer array
        labels = np.asarray(labels, np.int)
        # convert the labels to a strings array
        labels = np.asarray(labels, np.str)
        # set the number of labels
        if not self.num_labels:
            self.num_labels = np.shape(labels)[0]
        # set the maximum length of the labels
        if not self.max_label_length:
            # set the maximum length of the label names
            self.max_label_length = max_len(labels) + 1
        # all labels must be the same length and null-terminated
        labels = expand_and_terminate(labels)
        var = self.nc.createVariable('labels', 'c',
                                     ('numLabels', 'maxLabelLength'))
        var[:] = labels

    @property
    def target_patterns(self):
        """Target patterns."""
        # regression
        try:
            var = self.nc.variables['targetPatterns']
            return var.data
        except KeyError:
            return None

    @target_patterns.setter
    def target_patterns(self, target_patterns):
        """
        Set the target patterns (regression).

        :param target_patterns: target patterns [float array]

        """
        # TODO: make a list if a single value is given?
        if not self.num_time_steps:
            self.num_time_steps = np.shape(target_patterns)[0]
        if not self.target_pattern_size:
            self.target_pattern_size = np.shape(target_patterns)[1]
        var = self.nc.createVariable('targetPatterns', 'f',
                                     ('numTimesteps', 'targetPattSize'))
        var[:] = target_patterns.astype(np.float32)

    @property
    def target_strings(self):
        """Target strings."""
        # sequence_classification, transcription
        try:
            var = self.nc.variables['targetStrings']
            return var.data
        except KeyError:
            return None

    @target_strings.setter
    def target_strings(self, target_strings):
        """
        Set the target strings (sequence classification, transcription).

        :param target_strings: target strings [list of strings]

        """
        # TODO: make a list if a single value is given?
        if not self.num_sequences:
            self.num_sequences = len(target_strings)
        if not self.max_target_string_length:
            self.max_target_string_length = max_len(target_strings) + 1
        # all targetStrings must be the same length and null-terminated
        targetStrings = expand_and_terminate(target_strings)
        var = self.nc.createVariable('targetStrings', 'c',
                                     ('numTimesteps', 'maxTargStringLength'))
        var[:] = targetStrings

    @property
    def sequence_tags(self):
        """Sequence tags."""
        # optional
        try:
            var = self.nc.variables['seqTags']
            return var.data
        except KeyError:
            return None

    @sequence_tags.setter
    def sequence_tags(self, sequence_tags):
        """
        Set the sequence tags (optional data).

        :param sequence_tags: sequence tags [list of strings]

        """
        # make a list if a single value is given
        if isinstance(sequence_tags, str):
            sequence_tags = [sequence_tags]
        if not self.num_sequences:
            self.num_sequences = len(sequence_tags)
        if not self.max_sequence_tag_length:
            self.max_sequence_tag_length = max_len(sequence_tags) + 1
        # all seqTags must be the same length and null-terminated
        sequence_tags = expand_and_terminate(sequence_tags)
        var = self.nc.createVariable('seqTags', 'c',
                                     ('numSeqs', 'maxSeqTagLength'))
        var[:] = sequence_tags


# .nc file creation
def create_nc_file(filename, data, targets, tags=None):
    """
    Create a .nc file with the given input data and targets.

    :param filename: name of the file to create
    :param data:     input data [numpy array]
    :param targets:  corresponding targets [numpy array]
    :param tags:     additional tags (optional) [dict]

    """
    # create the .nc file
    nc = NetCDF(filename, 'w')
    # tags
    if tags:
        nc.sequence_tags = str(tags)
    # ground truth
    if targets.ndim == 1:
        nc.target_classes = targets
    else:
        nc.target_patterns = targets
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
    # save file
    nc.close()
    # return
    return filename


# .nc file testing
class TestThread(Thread):
    """
    Class for testing a .nc file against multiple networks and distributing the
    work to multiple threads.

    """

    def __init__(self, work_queue, return_queue, verbose=2):
        """
        Test a file against multiple neural networks.

        :param work_queue:   queue with work items
        :param return_queue: queue for the results
        :param verbose:      show RNNLIB's output

        """
        # init the thread
        super(TestThread, self).__init__()
        # set attributes
        self.work_queue = work_queue
        self.return_queue = return_queue
        self.verbose = verbose
        self.kill = False

    def run(self):
        """Test file against all neural networks in the queue."""
        while not self.kill:
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
                if self.verbose > 1:
                    subprocess.call(args)
                else:
                    with open(os.devnull, 'w') as devnull:
                        subprocess.call(args, stdout=devnull, stderr=devnull)
            except OSError:
                # TODO: which exception should be raised?
                raise SystemExit('rnnlib binary not found')
            # read the activations
            act = None
            try:
                # classification output
                act = RnnlibActivations('%s/output_outputActivations' %
                                        tmp_work_path)
            except IOError:
                # could not read in the activations, try regression
                # TODO: make regression task work as well
                #       until then just output the log
                with open("%s/log" % tmp_work_path, 'rb') as log:
                    print log.read()
                raise RuntimeError("Error while RNNLIB processing.")
            finally:
                # put a tuple with nc file, nn file and activations
                # in the return queue
                self.return_queue.put((nc_file, nn_file, act))
                # clean up
                shutil.rmtree(tmp_work_path)
                # signal to queue that job is done
                self.work_queue.task_done()


def create_pool(threads=THREADS, verbose=False):
    """
    Create a pool of working threads.

    :param threads:  number of parallel threads
    :param verbose:  be verbose
    :return:         a tuple with working and return queues

    Note: the work queue must contain tuples with (nc_file, nn_file),
          the return queue contains the same tuples extended by the activations
          (nc_file, nn_file, activations).

    """
    # a queue for the work items
    work_queue = Queue()
    # a queue for the results
    return_queue = Queue()
    # start N threads parallel
    workers = [TestThread(work_queue, return_queue, verbose)
               for _ in range(threads)]
    for w in workers:
        w.setDaemon(True)
        w.start()
    # return the queues
    return work_queue, return_queue


def test_nc_files(nc_files, nn_files, work_queue, return_queue):
    """
    Test a list of .nc files against multiple neural networks.

    :param nc_files:     list with .nc files to be tested
    :param nn_files:     list with network .save files
    :param work_queue:   a work queue
    :param return_queue: a return queue
    :return:             list with activations
                         (a numpy array for each .nc file)

    """
    if not nc_files:
        raise ValueError('no .nc files given')
    if not nn_files:
        raise ValueError('no pre-trained neural network files given')
    # put a combination of .nc files and neural networks in the queue
    for nc_file in nc_files:
        for nn_file in nn_files:
            work_queue.put((nc_file, nn_file))
    # wait until everything has been processed
    work_queue.join()
    # init return list
    activations = [None] * len(nc_files)
    num_activations = [1] * len(nc_files)
    # get all the activations and process them accordingly
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


class RnnlibConfig(object):
    """
    Rnnlib config file class.

    """

    def __init__(self, filename=None):
        """
        Creates a new RnnlibConfig instance.

        :param filename: name of the config file for rnnlib

        """
        # container for weights
        self.w = {}
        # attributes
        self.train_files = None
        self.val_files = None
        self.test_files = None
        self.layer_sizes = None
        self.layer_types = None
        self.layer_transfer_fn = None
        self.bidirectional = False
        self.task = None
        self.learn_rate = None
        self.momentum = None
        self.optimizer = None
        self.rand_seed = 0
        self.noise = 0
        self.weight_noise = 0
        self.l1 = 0
        self.l2 = 0
        self.patience = 20
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
        lines = f.readlines()
        # get some general information
        for line in lines:
            # save the file sets
            if line.startswith('trainFile'):
                self.train_files = line[:].split()[1].split(',')
            elif line.startswith('valFile'):
                self.val_files = line[:].split()[1].split(',')
            elif line.startswith('testFile'):
                self.test_files = line[:].split()[1].split(',')
            elif line.startswith('#testFile'):
                self.test_files = line[:].split()[1].split(',')
            # size and type of hidden layers
            elif line.startswith('hiddenSize'):
                self.layer_sizes = np.array(line[:].split()[1].split(','),
                                            dtype=np.int).tolist()
                # number of the output layer
                num_output = len(self.layer_sizes)
            elif line.startswith('hiddenType'):
                hidden = line[:].split()[1]
                if hidden == 'lstm':
                    self.layer_types = ['LSTM'] * len(self.layer_sizes)
                    self.layer_transfer_fn = ['tanh'] * len(self.layer_sizes)
                else:
                    self.layer_types = ['Recurrent'] * len(self.layer_sizes)
                    self.layer_transfer_fn = [hidden] * len(self.layer_sizes)
            # task
            elif line.startswith('task'):
                self.task = line[:].split()[1]
                # set the output layer type
                if self.task == 'classification':
                    self.layer_types.append('FeedForward')
                    self.layer_transfer_fn.append('sigmoid')
                elif self.task == 'regression':
                    self.layer_types.append('FeedForward')
                    self.layer_transfer_fn.append('linear')
                else:
                    raise ValueError('unknown task, cannot set type of output '
                                     'layer.')
            # get bidirectional information
            # unfortunately RNNLIB does not store bidirectional information in
            # its .save files, so we do it two runs
            elif line.startswith('weightContainer_'):
                parts = line[:].split()
                if parts[0].endswith('0_1_weights'):
                    self.bidirectional = True
            # append output layer size
            if line.startswith('weightContainer_bias_to_output_weights'):
                output_size = int(line[:].split()[1])
                self.layer_sizes.append(output_size)

        # process the weights
        for line in lines:
            # save the weights
            if line.startswith('weightContainer_'):
                # line format: weightContainer_bias_to_hidden_0_0_weights \
                # num_weights weight0 weight1 ...
                parts = line[:].split()
                # only use the weights
                if parts[0].endswith('_weights'):
                    # get the weights
                    w = np.array(parts[2:], dtype=np.float32)
                    # get rid of beginning and end of name
                    name = re.sub('weightContainer_', '', str(parts[0][:]))
                    name = re.sub('_weights', '', name)
                    # alter the name to a more useful schema
                    name = re.sub('_to_', '_', name)
                    name = re.sub('input_', 'i_', name)
                    name = re.sub('hidden_', 'layer_', name)
                    name = re.sub('bias_', 'b_', name)
                    name = re.sub('_output', '_o', name)
                    name = re.sub('gather_._', 'i_', name)
                    name = re.sub('_peepholes', '_peephole_weights', name)
                    # hidden layer recurrent weights handling
                    for i in range(len(self.layer_sizes)):
                        name = re.sub('layer_%s_0_layer_%s_0_delay.*' % (i, i),
                                      'layer_%s_0_recurrent_weights' % i, name)
                        name = re.sub('layer_%s_1_layer_%s_1_delay.*' % (i, i),
                                      'layer_%s_1_recurrent_weights' % i, name)
                    # renaming / renumbering beginning
                    if name.startswith('i_'):
                        # weights
                        name = "%s_weights" % name[2:]
                    if name.startswith('b_'):
                        # bias
                        name = "%s_bias" % name[2:]
                    if name.startswith('o_'):
                        # output layer
                        name = "layer_%s_0_%s" % (num_output, name[2:])
                    if name.endswith('_o'):
                        name = re.sub('layer_%s_0_o' % (num_output - 1),
                                      'layer_%s_0_weights' % num_output,
                                      name)
                        name = re.sub('layer_%s_1_o' % (num_output - 1),
                                      'layer_%s_1_weights' % num_output,
                                      name)
                    # rename bidirectional stuff (not for output layer)
                    for i in range(len(self.layer_sizes)):
                        if self.bidirectional:
                            # fix the weird counting in gather layers
                            name = re.sub('layer_%s_1' % i, 'layer_%s' % i,
                                          name)
                            name = re.sub('layer_%s_0' % i,
                                          'layer_%s_%s' % (i, REVERSE), name)
                        else:
                            name = re.sub('layer_%s_0' % i, 'layer_%s' % i,
                                          name)
                    # reshape the weights
                    # FIXME: evil hack
                    layer_num = int(name[6])
                    layer_size = self.layer_sizes[layer_num]
                    # RNNLIB stacks bwd and fwd layers in a weird way, thus
                    # swap the weights of the hidden layers if it is
                    # bidirectional and not the first hidden layer
                    swap = (layer_num >= 1 and layer_num != num_output and
                            self.bidirectional and name.endswith("weights") and
                            "peephole" not in name and "recurrent" not in name)
                    # if we use LSTM units, align weights differently
                    if self.layer_types[layer_num] == 'LSTM':
                        if 'peephole' in name:
                            # peephole connections
                            w = w.reshape(3 * layer_size, -1)
                        else:
                            # bias, weights and recurrent connections
                            if swap:
                                w = w.reshape(
                                    4 * layer_size, 2, -1)[:, ::-1, :].ravel()
                            w = w.reshape(4 * layer_size, -1)
                    # "normal" units
                    else:
                        if swap:
                            w = w.reshape(
                                layer_size, 2, -1)[:, ::-1, :].ravel()
                        w = w.reshape(layer_size, -1).T
                    # save the weights
                    self.w[name] = w
        # stack the output weights
        if self.bidirectional:
            fwd = self.w.pop('layer_%s_weights' % num_output)
            bwd = self.w.pop('layer_%s_%s_weights' % (num_output, REVERSE))
            self.w['layer_%s_weights' % num_output] = np.vstack((fwd, bwd))
            # rename bias
            bias = self.w.pop('layer_%s_%s_bias' % (num_output, REVERSE))
            self.w['layer_%s_bias' % num_output] = bias
        # close the file
        f.close()

    def save(self, filename):
        """
        Save the RNNLIB config file.

        :param filename: name of the config file

        """
        # write the config file(s)
        # TODO: use madmom.utils.open
        f = open(filename, 'wb')
        f.write('task %s\n' % self.task)
        f.write('autosave true\n')
        # use the 1st hidden layer
        f.write('hiddenType %s\n' % self.layer_types[0])
        f.write('hiddenSize %s\n' % ",".join(str(x) for x in self.layer_sizes))
        f.write('bidirectional %s\n' % str(self.bidirectional).lower())
        f.write('dataFraction 1\n')
        f.write('maxTestsNoBest %s\n' % str(self.patience))
        f.write('learnRate %s\n' % str(self.learn_rate))
        f.write('momentum %s\n' % str(self.momentum))
        f.write('optimiser %s\n' % str(self.optimizer))
        f.write('randSeed %s\n' % str(self.rand_seed))
        f.write('inputNoiseDev %s\n' % str(self.noise))
        f.write('weightDistortion %s\n' % str(self.weight_noise))
        f.write('l1 %s\n' % str(self.l1))
        f.write('l2 %s\n' % str(self.l2))
        if len(self.train_files) > 0:
            f.write('trainFile %s\n' % ",".join(self.train_files))
        if len(self.val_files) > 0:
            f.write('valFile %s\n' % ",".join(self.val_files))
        if len(self.test_files) > 0:
            f.write('testFile %s\n' % ",".join(self.test_files))
        f.close()

    def test(self, out_dir=None, file_set='test', threads=THREADS,
             fps=None, verbose=False):
        """
        Test the given set of files.

        :param out_dir:  output directory for activations
        :param file_set: which set should be tested {train, val, test}
        :param threads:  number of working threads
        :param fps:      frames per seconds (used for saving the activations)
        :param verbose:  verbose output
        :return:         the output directory

        Note: If no `out_dir` is given, an output directory with the name based
              on the config file name is created.

        """
        # if no output directory was given, use the name of the file + set
        if out_dir is None:
            out_dir = "%s.%s" % (os.path.splitext(self.filename)[0], file_set)
        # create output directory
        try:
            os.mkdir(out_dir)
        except OSError:
            # directory exists already, update modification date
            os.utime(out_dir, None)
        # test all files of the given set
        nc_files = getattr(self, "%s_files" % file_set)
        # create a pool of workers
        work_queue, return_queue = create_pool(threads, verbose)
        # test all files
        activations = test_nc_files(nc_files, [self.filename], work_queue,
                                    return_queue)
        # save all activations
        for nc_file in nc_files:
            # name of the activations file
            basename = os.path.basename(os.path.splitext(nc_file)[0])
            act_file = "%s/%s" % (out_dir, basename)
            # try to get the fps from the .nc file
            try:
                import ast
                # it's always the first sequence tag entry (i.e. first file)
                tags = "".join(NetCDF(nc_file, 'r').sequence_tags[0])
                tags = ast.literal_eval(tags)
                fps = tags['fps']
            except SyntaxError:
                pass
            # save the activations (at the given index position)
            act = activations[nc_files.index(nc_file)]
            if fps:
                # save as Activations with fps
                Activations(act, fps=fps).save(act_file)
            else:
                # save as plain numpy array
                np.save(act_file, act)

        # return the output directory
        return out_dir

    def save_model(self, filename=None, comment=None, npz=True):
        """
        Save the model to a .h5 file which can be universally used and
        converted to .npz to create a madmom.ml.rnn.RNN instance.

        :param filename: save the model to this file
        :param comment:  optional comment for the model
        :param npz:      also convert to .npz format

        Note: If no filename is given, the filename of the .save file is used
              and the extension is set to .h5 or .npz respectively.

        """
        import h5py
        # check if weights are present
        if not self.w:
            raise ValueError('please load a configuration file first')
        # set a default file name
        if filename is None:
            filename = "%s.h5" % os.path.splitext(self.filename)[0]
        # set the number of the output layer
        num_output = len(self.layer_sizes) - 1
        if num_output > 8:
            # FIXME: I know that works only with layer nums 0..9, have to come
            #        up with a proper solution.
            raise ValueError('too many layers, please fix me.')
        # save model
        with h5py.File(filename, 'w') as h5:
            # model attributes
            h5_m = h5.create_group('model')
            h5_m.attrs['type'] = 'RNN'
            if comment:
                h5_m.attrs['comment'] = comment
            # layers
            h5_l = h5.create_group('layer')
            # create a subgroup for each layer
            for layer in range(len(self.layer_sizes)):
                # create group with layer number
                grp = h5_l.create_group(str(layer))
                # iterate over all weights
                for key in sorted(self.w.keys()):
                    # skip if it's not the right layer
                    if not key.startswith('layer_%s_' % layer):
                        continue
                    # get the weights
                    w = self.w[key]
                    key = re.sub('layer_%s_' % layer, '', key)
                    # save the weights
                    grp.create_dataset(key, data=w.astype(np.float32))
                    # include the layer type as attribute
                    layer_type = self.layer_types[layer]
                    grp.attrs['type'] = str(layer_type)
                    grp.attrs['transfer_fn'] = \
                        str(self.layer_transfer_fn[layer])
                    # also for the reverse bidirectional layer if it exists
                    if self.bidirectional and layer != num_output:
                        grp.attrs['%s_type' % REVERSE] = str(layer_type)
                        grp.attrs['%s_transfer_fn' % REVERSE] = \
                            str(self.layer_transfer_fn[layer])
                # next layer
        # also convert to .npz
        if npz:
            from .io import convert_model
            convert_model(filename)

    def create_rnn(self):
        """Create a RNN."""
        # shortcut
        w = self.w
        # create layers
        layers = []
        i = 0
        for i, layer_type in enumerate(self.layer_types[:-1]):
            if layer_type == 'lstm':
                # LSTM units
                transfer_fn = tanh
                # create a fwd layer
                w_ = w['layer_%d_weights' % i]
                b_ = w['layer_%d_bias' % i]
                r_ = w['layer_%d_recurrent_weights' % i]
                p_ = w['layer_%d_peephole_weights' % i]
                layer = LSTMLayer(w_, b_, r_, p_, transfer_fn)
                # create a bwd layer and a bidirectional layer
                if self.bidirectional:
                    w_ = w['layer_%d_reverse_weights' % i]
                    b_ = w['layer_%d_reverse_bias' % i]
                    r_ = w['layer_%d_reverse_recurrent_weights' % i]
                    p_ = w['layer_%d_reverse_peephole_weights' % i]
                    bwd_layer = LSTMLayer(w_, b_, r_, p_, transfer_fn)
                    layer = BidirectionalLayer(layer, bwd_layer)
            else:
                # "normal" units
                transfer_fn = globals()[layer_type]
                # create a fwd layer
                w_ = w['layer_%d_weights' % i]
                b_ = w['layer_%d_bias' % i]
                r_ = w['layer_%d_recurrent_weights' % i]
                layer = RecurrentLayer(w_, b_, r_, transfer_fn)
                # create a bwd layer and a bidirectional layer
                if self.bidirectional:
                    w_ = w['layer_%d_reverse_weights' % i]
                    b_ = w['layer_%d_reverse_bias' % i]
                    r_ = w['layer_%d_reverse_recurrent_weights' % i]
                    bwd_layer = RecurrentLayer(w_, b_, r_, transfer_fn)
                    layer = BidirectionalLayer(layer, bwd_layer)
            # append the layer
            layers.append(layer)
        # create output layer
        i += 1
        w_ = w['layer_%d_weights' % i]
        b_ = w['layer_%d_bias' % i]
        out = FeedForwardLayer(w_, b_, globals()[self.layer_types[-1]])
        layers.append(out)
        # create and return a RNN
        return RecurrentNeuralNetwork(layers)


def test_save_files(files, out_dir=None, file_set='test', threads=THREADS,
                    verbose=False, fps=100):
    """
    Test the given set of files.

    :param files:    list with RNNLIB .save files
    :param out_dir:  output directory for activations
    :param file_set: which set should be tested {train, val, test}
    :param threads:  number of working threads
    :param verbose:  be verbose
    :param fps:      frame rate of the Activations to be saved

    Note: If `out_dir` is set and multiple network files contain the same
          files, the activations get averaged and saved to `out_dir`.

          The activations are saved as Activations instances, i.e. .npz files
          which include a frame rate in fps (frames per second).

    """
    # FIXME: function only works if called in the directory of the NN file
    if out_dir is None:
        # test all NN files individually
        for save_file in files:
            rnn_config = RnnlibConfig(save_file)
            rnn_config.test(file_set=file_set, threads=threads)
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
        for save_file in files:
            rnn_config = RnnlibConfig(save_file)
            nc_files.extend(getattr(rnn_config, "%s_files" % file_set))
        # remove duplicates
        nc_files = list(set(nc_files))
        # create a pool of workers
        work_queue, return_queue = create_pool(threads, verbose)
        # test each .nc files against the NN files if it is in the given set
        # Note: do not flip the order of the loops, otherwise files could be
        #       tested even if they were included in the train set!
        # TODO: unify this with RnnlibConfig.test()
        for nc_file in nc_files:
            # check in which .save files the .nc file is included
            nc_nn_files = []
            for save_file in files:
                rnn_config = RnnlibConfig(save_file)
                if nc_file in getattr(rnn_config, "%s_files" % file_set):
                    nc_nn_files.append(save_file)
            # test the .nc file against these networks
            activations = test_nc_files([nc_file], nc_nn_files, work_queue,
                                        return_queue)
            # name of the activations file
            basename = os.path.basename(os.path.splitext(nc_file)[0])
            act_file = "%s/%s" % (out_dir, basename)
            # cast the activations to an Activations instance (we only passed
            # one .nc file, so it's the first activation in the returned list)
            if verbose:
                print act_file
            Activations(activations[0], fps=fps).save(act_file)


def create_config(files, config, out_dir, num_folds=8, randomize=False,
                  bidirectional=True, task='classification', learn_rate=1e-5,
                  layer_sizes=[25, 25, 25], layer_type='lstm', momentum=0.9,
                  optimizer='steepest', splits=None, noise=0, weight_noise=0,
                  l1=0, l2=0, patience=20):
    """
    Creates RNNLIB config files for N-fold cross validation.

    :param files:         use these .nc files
    :param config:        common base name for the config files
    :param out_dir:       output directory for config files
    :param num_folds:     number of folds
    :param randomize:     shuffle files before creating splits
    :param bidirectional: use bidirectional neural networks
    :param task:          neural network task
    :param learn_rate:    learn rate to use
    :param layer_sizes:   sizes of the hidden layers
    :param layer_type:    hidden layer types
    :param momentum:      momentum for steepest descent
    :param optimizer:     which optimizer to use {'steepest, 'rprop'}
    :param splits:        use pre-defined folds from splits folder
    :param noise:         add noise to inputs
    :param weight_noise:  add noise to weights
    :param l1:            L1 regularisation
    :param l2:            L2 regularisation
    :param patience:      early stop training after N epoch without improvement
                          of validation error

    """
    # filter the files to include only .nc files
    files = search_files(files, '.nc')
    # common output filename
    if out_dir is not None:
        try:
            # create output directory
            os.mkdir(out_dir)
        except OSError:
            # directory exists already
            pass
        out_file = '%s/%s' % (out_dir, config)
    else:
        out_file = config
    # shuffle the files
    if randomize:
        import random
        random.shuffle(files)
    # splits into N parts
    folds = []
    for _ in range(num_folds):
        folds.append([])
    if isinstance(splits, list):
        # we got a list of splits folders
        for split in splits:
            fold_files = search_files(split, '.fold')
            if len(fold_files) != num_folds:
                raise ValueError('number of folds must match.')
            for fold, fold_file in enumerate(fold_files):
                with open(fold_file, 'r') as s:
                    for line in s:
                        line = line.strip()
                        nc_file = match_file(line, files, match_suffix='.nc')
                        try:
                            folds[fold].append(str(nc_file[0]))
                        except IndexError:
                            print "can't find .nc file for file: %s" % line
    else:
        # use a standard splits
        for fold in range(num_folds):
            folds[fold] = [f for i, f in enumerate(files)
                           if i % num_folds == fold]
            if not folds[fold]:
                raise ValueError('not enough files for %d folds.' % num_folds)
    # create the rnnlib_config files
    if num_folds < 3:
        raise ValueError('cannot create splits with less than 3 folds.')
    for i in range(num_folds):
        rnnlib_config = RnnlibConfig()
        all_folds = np.arange(i, i + num_folds)
        test_fold = np.nonzero(all_folds % num_folds == 0)[0]
        val_fold = np.nonzero(all_folds % num_folds == 1)[0]
        train_fold = np.nonzero(all_folds % num_folds >= 2)[0]
        # assign the sets
        rnnlib_config.test_files = folds[int(test_fold)]
        rnnlib_config.val_files = folds[int(val_fold)]
        rnnlib_config.train_files = []
        for j in train_fold.tolist():
            rnnlib_config.train_files.extend(folds[j])
        rnnlib_config.task = task
        rnnlib_config.bidirectional = bidirectional
        rnnlib_config.learn_rate = learn_rate
        rnnlib_config.layer_sizes = layer_sizes
        rnnlib_config.layer_types = [layer_type] * len(layer_sizes)
        rnnlib_config.momentum = momentum
        rnnlib_config.optimizer = optimizer
        rnnlib_config.noise = noise
        rnnlib_config.weight_noise = weight_noise
        rnnlib_config.l1 = l1
        rnnlib_config.l2 = l2
        rnnlib_config.patience = patience
        rnnlib_config.save('%s_%s' % (out_file, i + 1))


def create_nc_files(files, annotations, out_dir, norm=False, att=0,
                    frame_size=2048, online=False, fps=100, filterbank=None,
                    num_bands=6, fmin=30, fmax=17000, norm_filters=True,
                    log=True, mul=1, add=0, diff=True, diff_ratio=0.5,
                    diff_frames=None, diff_max_bins=1, positive_diffs=True,
                    shift=0, spread=0, split=None, verbose=False):
    """
    Create .nc files for the given .wav and annotation files.

    :param files:          use the files (must contain both the audio files and
                           the annotation files)
    :param annotations:    use these annotation suffices [list of strings]
    :param out_dir:        output directory for the created .nc files

    Signal parameters:

    :param norm:           normalize the signal
    :param att:            attenuate the signal

    Framing parameters:

    :param frame_size:     size of one frame(s), if a list is given, the
                           individual spectrograms are stacked [int]
    :param fps:            use given frames per second [float]
    :param online:         online mode, i.e. use only past information

    Filterbank parameters:

    :param filterbank:     filterbank type [Filterbank]
    :param num_bands:      number of filter bands (per octave, depending on the
                           type of the filterbank)
    :param fmin:           the minimum frequency [Hz]
    :param fmax:           the maximum frequency [Hz]
    :param norm_filters:   normalize the filter to area 1 [bool]

    Logarithmic magnitude parameters:

    :param log:            scale the magnitude spectrogram logarithmically
                           [bool]
    :param mul:            multiply the magnitude spectrogram with this factor
                           before taking the logarithm [float]
    :param add:            add this value before taking the logarithm of the
                           magnitudes [float]

    Difference parameters:

    :param diff:           calculate the difference of the spectrogram [bool]
                           if the spectrograms are stacked, the differences
                           will be stacked, too
    :param diff_ratio:     calculate the difference to the frame at
                           which the window used for the STFT yields
                           this ratio of the maximum height [float]
    :param diff_frames:    calculate the difference to the N-th previous frame
                           [int] (if set, this overrides the value calculated
                           from the `diff_ratio`)
    :param diff_max_bins:  apply a maximum filter with this width (in bins in
                           frequency dimension) before calculating the diff;
                           (e.g. for the difference spectrogram of the SuperFlux
                           algorithm 3 `max_bins` are used together with a 24
                           band logarithmic filterbank)
    :param positive_diffs: keep only the positive differences, i.e. set all
                           diff values < 0 to 0

    Target parameters:

    :param shift:          shift the targets N seconds
    :param spread:         spread the targets N seconds
    :param split:          split the files into parts with N frames length

    Other parameters:

    :param verbose:        be verbose

    """
    from madmom.processors import SequentialProcessor
    from madmom.audio.signal import SignalProcessor
    from madmom.audio.spectrogram import (
        LogarithmicFilteredSpectrogramProcessor, StackedSpectrogramProcessor,
        SpectrogramDifferenceProcessor)

    from madmom.utils import load_events, quantize_events

    # define processing chain
    sig = SignalProcessor(num_channels=1, norm=norm, att=att)

    # we need to define which specs should be stacked
    spec = LogarithmicFilteredSpectrogramProcessor(filterbank=filterbank,
                                                   num_bands=num_bands,
                                                   fmin=fmin, fmax=fmax,
                                                   norm_filters=norm_filters,
                                                   log=log, mul=mul, add=add)
    if diff:
        diff = SpectrogramDifferenceProcessor(diff_ratio=diff_ratio,
                                              diff_frames=diff_frames,
                                              diff_max_bins=diff_max_bins,
                                              positive_diffs=positive_diffs)
    # stack specs with the given frame sizes
    stack = StackedSpectrogramProcessor(frame_size=frame_size,
                                        spectrogram=spec, difference=diff,
                                        online=online, fps=fps)
    processor = SequentialProcessor([sig, stack])

    # get all annotation files
    ann_files = []
    for annotation_suffix in annotations:
        ann_files.extend(search_files(files, annotation_suffix))

    # create .nc files
    for f in ann_files:
        # split the extension of the input file
        filename, annotation = os.path.splitext(f)
        # get the matching .wav or .flac file to the input file
        wav_files = match_file(f, files, annotation, '.wav')
        # no .wav file found, try .flac
        if len(wav_files) < 1:
            wav_files = match_file(f, files, annotation, '.flac')
        # no wav file found
        if len(wav_files) < 1:
            print "can't find audio file for %s" % f
            exit()
        # print file
        if verbose:
            print f

        # create the data for the .nc file from the .wav file
        nc_data = processor.process(wav_files[0])

        # targets
        if f.endswith('.notes'):
            # load notes
            from madmom.features.notes import load_notes
            notes = load_notes(f)
            # shift the notes if needed
            if shift:
                notes[:, 0] += shift
            # convert to frame numbers
            notes[:, 0] *= float(fps)
            # set the range of MIDI notes to 0..88
            notes[:, 2] -= 21
            # set the targets
            targets = np.zeros((len(nc_data), 88))
            for note in notes:
                try:
                    targets[int(note[0]), int(note[2])] = 1
                except IndexError:
                    pass
        else:
            # load events (onset/beat)
            targets = load_events(f)
            # spread the targets by simply adding a shifted version of itself
            if spread:
                targets = np.concatenate((targets, targets + spread,
                                          targets - spread))
                targets.sort()
            targets = quantize_events(targets, fps, length=len(nc_data),
                                      shift=shift)
        # tags
        tags = {'file': f, 'fps': fps, 'frame_size': frame_size,
                'online': online, 'filterbank': filterbank, 'log': log,
                'diff': diff}
        if filterbank:
            tags['filterbank'] = filterbank.__name__
            tags['num_bands'] = num_bands
            tags['fmin'] = fmin
            tags['fmax'] = fmax
            tags['norm_filters'] = norm_filters
        if log:
            tags['mul'] = mul
            tags['add'] = add
        if diff:
            tags['diff'] = diff.__class__.__name__
            tags['diff_ratio'] = diff_ratio
            tags['diff_frames'] = diff_frames
            tags['diff_max_bins'] = diff_max_bins
        if shift:
            tags['shift'] = shift
        if spread:
            tags['spread'] = spread
        # .nc file name
        if out_dir:
            nc_file = "%s/%s" % (out_dir, os.path.basename(filename))
        else:
            nc_file = "%s" % os.path.abspath(filename)
        # split files
        if split is None:
            # create a .nc file
            create_nc_file(nc_file + '.nc', nc_data, targets, tags)
        else:
            # length of one part
            length = int(split * fps)
            # number of parts
            parts = int(np.ceil(len(nc_data) / float(length)))
            digits = int(np.ceil(np.log10(parts + 1)))
            if digits > 4:
                raise ValueError('please chose longer splits')
            for part in range(parts):
                nc_part_file = "%s.part%04d.nc" % (nc_file, part)
                start = part * length
                stop = start + length
                if stop > len(nc_data):
                    stop = len(nc_data)
                tags['part'] = part
                tags['start'] = start
                tags['stop'] = stop - 1
                create_nc_file(nc_part_file, nc_data[start:stop],
                               targets[start:stop], tags)


def main():
    """
    Example script for testing RNNLIB .save files or creating .nc files
    understood by RNNLIB.

    """
    import argparse

    from madmom.utils import OverrideDefaultListAction
    from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
    from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                          LogarithmicSpectrogramProcessor,
                                          SpectrogramDifferenceProcessor)

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    This module creates .nc files to be used by RNNLIB.
    Tests .save files produced by RNNLIB.

    """)
    # add subparsers
    s = p.add_subparsers()

    # .save file testing options
    sp = s.add_parser('test', help='.save file testing help',
                      description="""
    Options for testing RNNLIB .save files.
    """)
    sp.set_defaults(func=test_save_files)
    sp.add_argument('files', nargs='+', help='.save files to be processed')
    sp.add_argument('-o', dest='out_dir', default=None,
                    help='output directory')
    sp.add_argument('--threads', action='store', type=int,
                    default=THREADS,
                    help='number of threads [default=%(default)i]')
    sp.add_argument('-v', dest='verbose', action='store_true',
                    help='be verbose')
    sp.add_argument('--set', action='store', type=str, dest='file_set',
                    default='test',
                    help='use this set {train, val, test} [default='
                         '%(default)s]')

    # config file creation options
    sp = s.add_parser('config', help='config file creation help',
                      description="""
    Options for creating RNNLIB config files.
    """)
    sp.set_defaults(func=create_config)
    sp.add_argument('files', nargs='+', help='files to be processed')
    sp.add_argument('-o', dest='out_dir', default=None,
                    help='output directory')
    sp.add_argument('-c', dest='config', default='config',
                    help='config file base name')
    sp.add_argument('--folds', dest='num_folds', default=8, type=int,
                    help='N-fold cross validation [default=%(default)s]')
    sp.add_argument('--splits', action='append', default=None,
                    help='use the pre-defined folds from this split folder '
                         '(argument can be given multiple times)')
    sp.add_argument('--randomize', action='store_true', default=False,
                    help='randomize splitting [default=%(default)s]')
    sp.add_argument('--task', default='classification', type=str,
                    help='learning task [default=%(default)s]')
    sp.add_argument('--bidirectional', action='store_true', default=False,
                    help='bidirectional network [default=%(default)s]')
    sp.add_argument('--learn_rate', default=1e-5, type=float,
                    help='learn rate [default=%(default)s]')
    sp.add_argument('--layer_sizes', default=[25, 25, 25], type=int,
                    action=OverrideDefaultListAction, sep=',',
                    help='layer sizes [default=%(default)s]')
    sp.add_argument('--layer_type', default='tanh', type=str,
                    help='layer type [default=%(default)s]')
    sp.add_argument('--momentum', default=0.9, type=float,
                    help='momentum for learning [default=%(default)s]')
    sp.add_argument('--optimizer', default='steepest', type=str,
                    help='optimizer [default=%(default)s]')
    sp.add_argument('--patience', default=20, type=int,
                    help='early stopping after N epochs [default=%(default)s]')
    sp.add_argument('--noise', default=0, type=float,
                    help='add noise to input [default=%(default).2f]')
    sp.add_argument('--weight_noise', default=0, type=float,
                    help='add noise to weight [default=%(default).2f]')
    sp.add_argument('--l1', default=0, type=float,
                    help='L1 regularisation [default=%(default).2f]')
    sp.add_argument('--l2', default=0, type=float,
                    help='L2 regularisation [default=%(default).2f]')

    # .nc file creation options
    sp = s.add_parser('create', help='.nc file creation help', description="""
    Options for creating .nc files from the given audio and annotation files.
    """)
    sp.set_defaults(func=create_nc_files)
    sp.add_argument('files', nargs='+', help='files to be processed')
    sp.add_argument('-o', dest='out_dir', default=None,
                    help='output directory')
    sp.add_argument('-v', dest='verbose', action='store_true',
                    help='be verbose')
    sp.add_argument('-a', dest='annotations', action=OverrideDefaultListAction,
                    default=['.onsets', '.beats', '.notes'],
                    help='annotations to use [default=%(default)s]')
    SignalProcessor.add_arguments(sp, norm=False, att=0)
    FramedSignalProcessor.add_arguments(sp, online=False,
                                        frame_size=[1024, 2048, 4096])

    from madmom.audio.filters import LogarithmicFilterbank as Filterbank
    FilteredSpectrogramProcessor.add_arguments(sp, filterbank=Filterbank,
                                               norm_filters=True,
                                               unique_filters=None)
    LogarithmicSpectrogramProcessor.add_arguments(sp, log=True, mul=1, add=1)
    SpectrogramDifferenceProcessor.add_arguments(sp, diff_ratio=0.5,
                                                 diff_max_bins=1,
                                                 positive_diffs=True)
    sp.add_argument('--split', default=None, type=float,
                    help='split files every N seconds')
    sp.add_argument('--shift', default=None, type=float,
                    help='shift targets N seconds')
    sp.add_argument('--spread', default=None, type=float,
                    help='spread targets N seconds')

    # parse arguments
    args = p.parse_args()

    # and call the appropriate function
    kwargs = vars(args)
    function = kwargs.pop('func')
    function(**kwargs)


if __name__ == '__main__':
    main()
