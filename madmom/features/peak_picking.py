#!/usr/bin/env python
# encoding: utf-8
"""
This file contains peak-picking functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""
import glob

import numpy as np
from scipy.ndimage import uniform_filter, maximum_filter
from madmom import IOProcessor, MODELS_PATH
from madmom.audio.signal import smooth as smooth_signal


# universal peak-picking method
from madmom.ml.rnn import RNNProcessor
from madmom.utils import write_events


def peak_picking(activations, threshold, smooth=None, pre_avg=0, post_avg=0,
                 pre_max=1, post_max=1):
    """
    Perform thresholding and peak-picking on the given activation function.

    :param activations: the activation function
    :param threshold:   threshold for peak-picking
    :param smooth:      smooth the activation function with the kernel
    :param pre_avg:     use N frames past information for moving average
    :param post_avg:    use N frames future information for moving average
    :param pre_max:     use N frames past information for moving maximum
    :param post_max:    use N frames future information for moving maximum
    :return:            indices of the detected peaks

    Notes: If no moving average is needed (e.g. the activations are independent
           of the signal's level as for neural network activations), set
           `pre_avg` and `post_avg` to 0.

           For offline peak picking, set `pre_max` and `post_max` to 1.

           For online peak picking, set all `post_` parameters to 0.

    "Evaluating the Online Capabilities of Onset Detection Methods"
    Sebastian Böck, Florian Krebs and Markus Schedl
    Proceedings of the 13th International Society for Music Information
    Retrieval Conference (ISMIR), 2012.

    """
    # smooth activations
    if smooth is not None:
        activations = smooth_signal(activations, smooth)
    # compute a moving average
    avg_length = pre_avg + post_avg + 1
    if avg_length > 1:
        # TODO: make the averaging function exchangeable (mean/median/etc.)
        avg_origin = int(np.floor((pre_avg - post_avg) / 2))
        if activations.ndim == 1:
            filter_size = avg_length
        elif activations.ndim == 2:
            filter_size = [avg_length, 1]
        else:
            raise ValueError('activations must be either 1D or 2D')
        mov_avg = uniform_filter(activations, filter_size, mode='constant',
                                 origin=avg_origin)
    else:
        # do not use a moving average
        mov_avg = 0
    # detections are those activations above the moving average + the threshold
    detections = activations * (activations >= mov_avg + threshold)
    # peak-picking
    max_length = pre_max + post_max + 1
    if max_length > 1:
        # compute a moving maximum
        max_origin = int(np.floor((pre_max - post_max) / 2))
        if activations.ndim == 1:
            filter_size = max_length
        elif activations.ndim == 2:
            filter_size = [max_length, 1]
        else:
            raise ValueError('activations must be either 1D or 2D')
        mov_max = maximum_filter(detections, filter_size, mode='constant',
                                 origin=max_origin)
        # detections are peak positions
        detections *= (detections == mov_max)
    # return indices
    if activations.ndim == 1:
        return np.nonzero(detections)[0]
    elif activations.ndim == 2:
        return np.nonzero(detections)
    else:
        raise ValueError('activations must be either 1D or 2D')


class PeakPickingProcessor(IOProcessor):
    """
    This class implements the peak-picking functionality which can be used
    universally.

    """
    FPS = 100
    THRESHOLD = 0.5  # binary threshold
    SMOOTH = 0
    PRE_AVG = 0
    POST_AVG = 0
    PRE_MAX = 1. / FPS  # corresponds to one frame
    POST_MAX = 1. / FPS
    COMBINE = 0.03
    DELAY = 0

    def __init__(self, threshold=THRESHOLD, smooth=SMOOTH, pre_avg=PRE_AVG,
                 post_avg=POST_AVG, pre_max=PRE_MAX, post_max=POST_MAX,
                 combine=COMBINE, delay=DELAY, online=False, fps=FPS,
                 *args, **kwargs):
        """
        Creates a new PeakPickingProcessor instance.

        :param threshold: threshold for peak-picking
        :param smooth:    smooth the activation function over N seconds
        :param pre_avg:   use N seconds past information for moving average
        :param post_avg:  use N seconds future information for moving average
        :param pre_max:   use N seconds past information for moving maximum
        :param post_max:  use N seconds future information for moving maximum
        :param combine:   only report one onset within N seconds
        :param delay:     report onsets N seconds delayed
        :param online:    use online peak-picking (i.e. no future information)
        :param fps:       frames per second used for conversion of timings

        Notes: If no moving average is needed (e.g. the activations are
               independent of the signal's level as for neural network
               activations), `pre_avg` and `post_avg` should be set to 0.

               For offline peak picking set `pre_max` >= 1. / `fps` and
               `post_max` >= 1. / `fps`

               For online peak picking, all `post_` parameters are set to 0.

        "Evaluating the Online Capabilities of Onset Detection Methods"
        Sebastian Böck, Florian Krebs and Markus Schedl
        Proceedings of the 13th International Society for Music Information
        Retrieval Conference (ISMIR), 2012.

        """
        # make this an IOProcessor by defining input and output processings
        super(PeakPickingProcessor, self).__init__(self.detect, write_events)
        # adjust some params for online mode
        if online:
            smooth = 0
            post_avg = 0
            post_max = 0
        self.threshold = threshold
        self.smooth = smooth
        self.pre_avg = pre_avg
        self.post_avg = post_avg
        self.pre_max = pre_max
        self.post_max = post_max
        self.combine = combine
        self.delay = delay
        self.fps = fps

    def detect(self, activations):
        """
        Detect the onsets in the given activation function.

        :param activations: onset activation function
        :return:            detected onsets

        """
        # convert timing information to frames and set default values
        # TODO: use at least 1 frame if any of these values are > 0?
        smooth = int(round(self.fps * self.smooth))
        pre_avg = int(round(self.fps * self.pre_avg))
        post_avg = int(round(self.fps * self.post_avg))
        pre_max = int(round(self.fps * self.pre_max))
        post_max = int(round(self.fps * self.post_max))
        # detect the peaks (function returns int indices)
        detections = peak_picking(activations, self.threshold, smooth,
                                  pre_avg, post_avg, pre_max, post_max)
        # TODO: make this multi-dim!

        # convert detections to a list of timestamps
        detections = detections.astype(np.float) / self.fps
        # shift if necessary
        if self.delay != 0:
            detections += self.delay
        # always use the first detection and all others if none was reported
        # within the last `combine` seconds
        if detections.size > 1:
            # filter all detections which occur within `combine` seconds
            combined_detections = detections[1:][np.diff(detections) >
                                                 self.combine]
            # add them after the first detection
            detections = np.append(detections[0], combined_detections)
        else:
            detections = detections
        # return the detections
        return detections

    @classmethod
    def add_arguments(cls, parser, threshold=THRESHOLD, smooth=None,
                      pre_avg=None, post_avg=None, pre_max=None, post_max=None,
                      combine=COMBINE, delay=DELAY):
        """
        Add onset peak-picking related arguments to an existing parser.

        :param parser:    existing argparse parser
        :param threshold: threshold for peak-picking
        :param smooth:    smooth the activation function over N seconds
        :param pre_avg:   use N seconds past information for moving average
        :param post_avg:  use N seconds future information for moving average
        :param pre_max:   use N seconds past information for moving maximum
        :param post_max:  use N seconds future information for moving maximum
        :param combine:   only report one event within N seconds
        :param delay:     report events N seconds delayed
        :return:          onset peak-picking argument parser group

        Parameters are included in the group only if they are not 'None'.

        """
        # add onset peak-picking related options to the existing parser
        g = parser.add_argument_group('onset peak-picking arguments')
        g.add_argument('-t', dest='threshold', action='store', type=float,
                       default=threshold,
                       help='detection threshold [default=%(default).2f]')
        if smooth is not None:
            g.add_argument('--smooth', action='store', type=float,
                           default=smooth,
                           help='smooth the activation function over N '
                                'seconds [default=%(default).2f]')
        if pre_avg is not None:
            g.add_argument('--pre_avg', action='store', type=float,
                           default=pre_avg,
                           help='build average over N previous seconds '
                                '[default=%(default).2f]')
        if post_avg is not None:
            g.add_argument('--post_avg', action='store', type=float,
                           default=post_avg, help='build average over N '
                           'following seconds [default=%(default).2f]')
        if pre_max is not None:
            g.add_argument('--pre_max', action='store', type=float,
                           default=pre_max,
                           help='search maximum over N previous seconds '
                                '[default=%(default).2f]')
        if post_max is not None:
            g.add_argument('--post_max', action='store', type=float,
                           default=post_max,
                           help='search maximum over N following seconds '
                                '[default=%(default).2f]')
        if combine is not None:
            g.add_argument('--combine', action='store', type=float,
                           default=combine,
                           help='combine events within N seconds '
                                '[default=%(default).2f]')
        if delay is not None:
            g.add_argument('--delay', action='store', type=float,
                           default=delay,
                           help='report the events N seconds delayed '
                                '[default=%(default)i]')
        # return the argument group so it can be modified if needed
        return g


class NNPeakPickingProcessor(IOProcessor):
    """
    Class for peak-picking with neural networks.

    """
    NN_FILES = glob.glob("%s/onsets_brnn_peak_picking_[1-8].npz" % MODELS_PATH)
    FPS = 100
    THRESHOLD = 0.4
    SMOOTH = 0.07
    COMBINE = 0.04
    DELAY = 0

    def __init__(self, nn_files=NN_FILES, threshold=THRESHOLD, smooth=SMOOTH,
                 combine=COMBINE, delay=DELAY, fps=FPS, *args, **kwargs):
        """
        Creates a new NNSpectralOnsetDetection instance.

        :param nn_files:  neural network files with models for peak-picking
        :param threshold: threshold for peak-picking
        :param smooth:    smooth the activation function over N seconds
        :param combine:   only report one onset within N seconds
        :param delay:     report onsets N seconds delayed

        "Enhanced peak picking for onset detection with recurrent neural
         networks"
        Sebastian Böck, Jan Schlüter and Gerhard Widmer
        Proceedings of the 6th International Workshop on Machine Learning and
        Music (MML), 2013.

        """
        # first perform RNN processing, then onset peak-picking
        rnn = RNNProcessor(nn_files=nn_files, num_threads=1)
        pp = PeakPickingProcessor(threshold=threshold, smooth=smooth,
                                     pre_max=1. / fps, post_max=1. / fps,
                                     combine=combine, delay=delay, fps=fps)
        # make this an IOProcessor by defining input and output processings
        super(NNPeakPickingProcessor, self).__init__(rnn, pp)

    @classmethod
    def add_arguments(cls, parser, nn_files=NN_FILES, threshold=THRESHOLD,
                      smooth=SMOOTH, combine=COMBINE, delay=DELAY):
        """
        Add peak-picking related arguments to an existing parser object.

        :param parser:    existing argparse parser object
        :param nn_files:  list with files of RNN models
        :param threshold: threshold for peak-picking
        :param smooth:    smooth the activation function over N seconds
        :param combine:   only report one event within N seconds
        :param delay:     report events N seconds delayed
        :return:          peak-picking argument parser group object

        """
        # add RNN parser arguments (but without number of threads)
        RNNProcessor.add_arguments(parser, nn_files=nn_files, num_threads=0)
        PeakPickingProcessor.add_arguments(parser, threshold=threshold,
                                              smooth=smooth, combine=combine,
                                              delay=delay)