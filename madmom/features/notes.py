#!/usr/bin/env python
# encoding: utf-8
"""
This file contains note transcription related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import os
import glob
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage.filters import median_filter, uniform_filter, maximum_filter

from ..utils import open
# from ..audio.filters import midi2hz, LogarithmicFilterbank

from . import Activations, RNNEventDetection


def load_notes(filename):
    """
    Load the target notes from a file.

    :param filename: input file name or file handle
    :returns:        numpy array with notes

    """
    with open(filename, 'rb') as f:
        return np.loadtxt(f)


# class SpectralNoteTranscription(object):
#     """
#     The SpectralNoteTranscription class implements a very basic method of note
#     transcription based on a spectrogram.
#
#     """
#     def __init__(self, spectrogram, num_harmonics=5, harmonic_frames=11,
#                  lgd=True, *args, **kwargs):
#         """
#         Creates a new SpectralNoteTranscription instance.
#
#         :param spectrogram:     the spectrogram object on which the note
#                                 transcription operates
#         :param num_harmonics:   model N harmonics
#         :param harmonic_frames: perform harmonic median filtering over N frames
#         :param lgd:             use local group delay to weight the spectrogram
#
#         """
#         # import
#         from ..audio.spectrogram import Spectrogram
#         # check spectrogram type
#         if isinstance(spectrogram, Spectrogram):
#             # already the right format
#             self._spectrogram = spectrogram
#         else:
#             # assume a file name, try to instantiate a Spectrogram object
#             self._spectrogram = Spectrogram(spectrogram, *args, **kwargs)
#         # attributes
#         self._num_harmonics = num_harmonics
#         self._harmonic_frames = harmonic_frames
#         self._lgd = lgd
#         # hidden attributes
#         self._notes = None
#         self._onsets = None
#
#     @property
#     def spectrogram(self):
#         """Spectrogram."""
#         return self._spectrogram
#
#     @property
#     def num_harmonics(self):
#         """Number of harmonics to model."""
#         return self._num_harmonics
#
#     @property
#     def harmonic_frames(self):
#         """Number of harmonics frames."""
#         return self._harmonic_frames
#
#     @property
#     def lgd(self):
#         """Use local group delay weighting."""
#         return self._lgd
#
#     @property
#     def notes(self):
#         """Naïve implementation of notes."""
#         if self._notes is None:
#             # alias
#             spec = self.spectrogram.spec
#
#             # use only harmonic parts
#             if self.harmonic_frames > 1:
#                 spec = median_filter(spec, (self.harmonic_frames, 1))
#
#             # add local group delay weighting
#             if self.lgd:
#                 lgd = 1 - np.abs(self.spectrogram.lgd) / np.pi
#                 # harmonic LGD?
#                 if self.harmonic_frames > 1:
#                     lgd = median_filter(lgd, (self.harmonic_frames, 1))
#                 # weight the spec
#                 spec *= lgd
#
#             # bin of the last fundamental
#             last_fundamental_bin = np.argmax(self.spectrogram.fft_freqs >=
#                                              midi2hz(109))
#             # spec sums
#             sums = np.zeros_like(spec)
#             for f in range(1, last_fundamental_bin):
#                 # sum the given number of harmonics
#                 f_sum = np.sum(spec[:, f::f][:, :self.num_harmonics], axis=1)
#                 # weight with the fundamental
#                 f_sum *= spec[:, f]
#                 # save for the given fundamental
#                 sums[:, f] = f_sum
#             # convert to MIDI scale
#             fmin = midi2hz(-1)
#             fb = LogarithmicFilterbank(
#                 self.spectrogram.num_fft_bins,
#                 self.spectrogram.frames.signal.sample_rate, 12, norm=True,
#                 duplicates=True, fmin=fmin)
#             # save the notes
#             self._notes = np.dot(sums, fb)
#         # return notes
#         return self._notes
#
#     @property
#     def onsets(self):
#         """Onsets."""
#         if self._onsets is None:
#             self._onsets = np.zeros_like(self._notes)
#             self._onsets[1:] = np.diff(self._notes, axis=0)
#             self._onsets *= self._onsets > 0
#         # return onsets
#         return self._onsets


# universal peak-picking method
def peak_picking(activations, threshold, smooth=None, pre_avg=0, post_avg=0,
                 pre_max=1, post_max=1):
    """
    Perform thresholding and peak-picking on the given activation function.

    :param activations: note activations (2D numpy array)
    :param threshold:   threshold for peak-picking (1D numpy array)
    :param smooth:      smooth the activation function with the kernel
                        [default=None]
    :param pre_avg:     use N frames past information for moving average
                        [default=0]
    :param post_avg:    use N frames future information for moving average
                        [default=0]
    :param pre_max:     use N frames past information for moving maximum
                        [default=1]
    :param post_max:    use N frames future information for moving maximum
                        [default=1]

    Notes: If no moving average is needed (e.g. the activations are independent
           of the signal's level as for neural network activations), set
           `pre_avg` and `post_avg` to 0.

           For offline peak picking set `pre_max` and `post_max` to 1.

           For online peak picking set all `post_` parameters to 0.

    """
    # TODO: this code is very similar to features.onsets.peak_picking();
    #       unify these 2 functions!
    # smooth activations
    kernel = None
    if isinstance(smooth, int):
        # size for the smoothing kernel is given
        if smooth > 1:
            kernel = np.hamming(smooth)
    elif isinstance(smooth, np.ndarray):
        # otherwise use the given smooth kernel directly
        if smooth.size > 1:
            kernel = smooth
    if kernel is not None:
        # convolve with the kernel
        activations = convolve2d(activations, kernel[:, np.newaxis], 'same')
    # threshold activations
    avg_length = pre_avg + post_avg + 1
    if avg_length > 1:
        # compute a moving average
        avg_origin = int(np.floor((pre_avg - post_avg) / 2))
        # TODO: make the averaging function exchangeable (mean/median/etc.)
        mov_avg = uniform_filter(activations, [avg_length, 1], mode='constant',
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
        mov_max = maximum_filter(detections, [max_length, 1], mode='constant',
                                 origin=max_origin)
        # detections are peak positions
        detections *= (detections == mov_max)
    # return indices
    return np.nonzero(detections)



class NoteTranscription(RNNEventDetection):
    """
    NoteTranscription class.

    """
    # set the path to saved neural networks and generate lists of NN files
    NN_PATH = '%s/../ml/data' % (os.path.dirname(__file__))
    NN_FILES = glob.glob("%s/notes_brnn*npz" % NN_PATH)

    # TODO: this information should be included/extracted in/from the NN files
    FPS = 100
    BANDS_PER_OCTAVE = 12
    MUL = 5
    ADD = 1

    # TODO: These do not seem to be used anywhere
    FMIN = 27.5
    FMAX = 18000
    RATIO = 0.5
    NORM_FILTERS = True

    # default values for note peak-picking
    THRESHOLD = 0.35
    SMOOTH = 0.09
    PRE_AVG = 0
    POST_AVG = 0
    PRE_MAX = 1. / FPS
    POST_MAX = 1. / FPS
    # default values for note reporting
    COMBINE = 0.05
    DELAY = 0

    # def __init__(self, data, nn_files=NN_FILES, threshold=THRESHOLD,
    #              combine=COMBINE, delay=DELAY, smooth=SMOOTH, pre_avg=PRE_AVG,
    #              post_avg=POST_MAX, pre_max=PRE_MAX, post_max=POST_AVG,
    #              **kwargs):
    #     """
    #     Creates a new NoteTranscription instance with the given
    #     activations (can be read from a file).
    #
    #     :param data:       Signal, activations or filename.
    #                        See EventDetection class for more details.
    #     :param nn_files:   list of files that define the RNN
    #     :param threshold: array with thresholds for peak-picking
    #     :param combine:   only report one note within N seconds
    #     :param delay:     report onsets N seconds delayed
    #     :param smooth:    smooth the activation function over N seconds
    #     :param pre_avg:   use N seconds past information for moving average
    #     :param post_avg:  use N seconds future information for moving average
    #     :param pre_max:   use N seconds past information for moving maximum
    #     :param post_max:  use N seconds future information for moving maximum
    #
    #     Notes: If no moving average is needed (e.g. the activations are
    #            independent of the signal's level as for neural network
    #            activations), `pre_avg` and `post_avg` should be set to 0.
    #     """
    #     spr = super(NoteTranscription, self)
    #     spr.__init__(data, nn_files, fps=NoteTranscription.FPS,
    #                  bands_per_octave=NoteTranscription.BANDS_PER_OCTAVE,
    #                  window_sizes=NoteTranscription.WINDOW_SIZES,
    #                  mul=NoteTranscription.MUL, add=NoteTranscription.ADD,
    #                  norm_filters=NoteTranscription.NORM_FILTERS, **kwargs)
    #
    #     self.threshold = threshold
    #     self.combine = combine
    #     self.delay = delay
    #
    #     # convert timing information to frames and set default values
    #     # TODO: use at least 1 frame if any of these values are > 0?
    #     self.smooth = int(round(self.fps * smooth))
    #     self.pre_avg = int(round(self.fps * pre_avg))
    #     self.post_avg = int(round(self.fps * post_avg))
    #     self.pre_max = int(round(self.fps * pre_max))
    #     self.post_max = int(round(self.fps * post_max))

    def pre_process(self, frame_sizes=[1024, 2048, 4096], origin='offline'):
        """
        Pre-process the signal to obtain a data representation suitable for RNN
        processing.

        :param frame_sizes: frame sizes for the spectrograms
        :param origin:      origin of the frames
        """
        from ..audio.spectrogram import LogFiltSpec
        data = []
        for frame_size in frame_sizes:
            # TODO: the signal processing parameters should be included in and
            #       extracted from the NN model files
            s = LogFiltSpec(self.signal, frame_size=frame_size, fps=100,
                            origin=origin, bands_per_octave=12, mul=5, add=1,
                            norm_filters=True, fmin=30, fmax=17000,
                            ratio=0.5)
            # append the spec and the positive first order diff to the data
            data.append(s.spec)
            data.append(s.pos_diff)
        # stack the data and return it
        self._data = np.hstack(data)
        return self._data

    def process(self):
        """
        Test the data with the defined RNNs.

        :return: activations

        """
        # process the data
        super(NoteTranscription, self).process()
        # reshape the activations
        self._activations = self._activations.reshape(-1, 88)
        # and return them
        return self._activations

    def detect(self, threshold=THRESHOLD, smooth=SMOOTH, combine=COMBINE,
               delay=DELAY):
        """
        Detect the notes with the given peak-picking parameters.

        """
        # detect onsets
        detections = peak_picking(self.activations, threshold, smooth)
        # convert to seconds / MIDI note numbers
        onsets = detections[0].astype(np.float) / self.fps
        midi_notes = detections[1] + 21
        # shift if necessary
        if delay != 0:
            onsets += delay
        # combine multiple notes
        if combine > 0:
            detections = []
            # iterate over each detected note separately
            for note in np.unique(midi_notes):
                # get all note detections
                note_onsets = onsets[midi_notes == note]
                # always use the first note
                detections.append((note_onsets[0], note))
                # filter all notes which occur within `combine` seconds
                combined_note_onsets = note_onsets[1:][np.diff(note_onsets) >
                                                       combine]
                # zip them with the MIDI note number and add them to the list
                detections.extend(zip(combined_note_onsets,
                                      [note] * len(combined_note_onsets)))
        else:
            # just zip all detected notes
            detections = zip(onsets, midi_notes)
        # sort and save the detections
        self._detections = sorted(detections)
        # also return them
        return self._detections

    def write(self, filename, sep='\t'):
        """
        Write the detected notes to a file.

        :param filename: output file name or file handle
        :param sep:    separator for the fields [default='\t']

        """
        from ..utils import open
        # write the detected notes to the output
        with open(filename, 'wb') as f:
            for note in self.detections:
                f.write(sep.join([str(x) for x in note]) + '\n')

    @classmethod
    def add_arguments(cls, parser, nn_files=NN_FILES, threshold=THRESHOLD,
                      smooth=SMOOTH, combine=COMBINE, delay=DELAY,
                      pre_avg=PRE_AVG, post_avg=POST_AVG,
                      pre_max=PRE_MAX, post_max=POST_MAX, **kwargs):
        """
        Add note transcription related arguments to an existing parser object.

        :param parser:    existing argparse parser object
        :param nn_files:  list of files that define the RNN
        :param threshold: threshold for peak-picking
        :param smooth:    smooth the note activations over N seconds
        :param combine:   only report one note within N seconds and pitch
        :param delay:     report notes N seconds delayed
        :param pre_avg:   use N seconds past information for moving average
        :param post_avg:  use N seconds future information for moving average
        :param pre_max:   use N seconds past information for moving maximum
        :param post_max:  use N seconds future information for moving maximum
        :return:          note argument parser group object

        """
        # add Activations parser
        Activations.add_arguments(parser)
        # add RNNEventDetection arguments
        RNNEventDetection.add_arguments(parser, nn_files=nn_files)
        # add note transcription related options to the existing parser
        g = parser.add_argument_group('note transcription arguments')
        g.add_argument('-t', dest='threshold', action='store', type=float,
                       default=threshold, help='detection threshold '
                       '[default=%(default)s]')
        g.add_argument('--smooth', action='store', type=float, default=smooth,
                       help='smooth the note activations over N seconds '
                       '[default=%(default).2f]')
        g.add_argument('--combine', action='store', type=float,
                       default=combine, help='combine notes within N seconds '
                       '(per pitch) [default=%(default).2f]')
        g.add_argument('--pre_avg', action='store', type=float,
                       default=pre_avg, help='build average over N previous '
                       'seconds [default=%(default).2f]')
        g.add_argument('--post_avg', action='store', type=float,
                       default=post_avg, help='build average over N following '
                       'seconds [default=%(default).2f]')
        g.add_argument('--pre_max', action='store', type=float,
                       default=pre_max, help='search maximum over N previous '
                       'seconds [default=%(default).2f]')
        g.add_argument('--post_max', action='store', type=float,
                       default=post_max, help='search maximum over N '
                       'following seconds [default=%(default).2f]')
        g.add_argument('--delay', action='store', type=float, default=delay,
                       help='report the notes N seconds delayed '
                       '[default=%(default)i]')
        # return the argument group so it can be modified if needed
        return g

#
# def parser():
#     """
#     Command line argument parser for note transcription.
#
#     """
#     import argparse
#     from ..utils.params import (audio, spec, filtering, log, note, save_load)
#
#     # define parser
#     p = argparse.ArgumentParser(
#         formatter_class=argparse.RawDescriptionHelpFormatter, description="""
#     If invoked without any parameters, the software detects all notes in
#     the given files.
#
#     """)
#     # general options
#     p.add_argument('files', metavar='files', nargs='+',
#                    help='files to be processed')
#     p.add_argument('-v', dest='verbose', action='store_true',
#                    help='be verbose')
#     p.add_argument('--length', action='store', type=float, default=None,
#                    help='length of the signal to process')
#     # add other argument groups
#     audio(p, online=False)
#     spec(p)
#     filtering(p, default=False)
#     log(p, default=True)
#     note(p)
#     save_load(p)
#     # parse arguments
#     args = p.parse_args()
#     # print arguments
#     if args.verbose:
#         print args
#     if args.length is not None:
#         args.length *= args.fps
#     else:
#         args.length = 'extend'
#     # return args
#     return args
#
#
# def main():
#     """
#     Example note transcription program.
#
#     """
#     import os.path
#
#     from ..utils import files
#     from ..audio.wav import Wav
#     from ..audio.spectrogram import Spectrogram
#     from ..audio.filters import LogarithmicFilterbank
#
#     # parse arguments
#     args = parser()
#
#     # init filterbank
#     fb = None
#
#     # which files to process
#     if args.load:
#         # load the activations
#         ext = '.activations'
#     else:
#         # only process .wav files
#         ext = '.wav'
#     # process the files
#     for f in files(args.files, ext):
#         if args.verbose:
#             print f
#
#         # use the name of the file without the extension
#         filename = os.path.splitext(f)[0]
#
#         # do the processing stuff unless the activations are loaded from file
#         if args.load:
#             # load the activations from file
#             # FIXME: fps must be encoded in the file
#             n = NoteTranscription(f, args.fps)
#         else:
#             # create a Wav object
#             w = Wav(f, mono=True, norm=args.norm, att=args.att)
#             if args.filter:
#                 # (re-)create filterbank if the sample rate is not the same
#                 if fb is None or fb.sample_rate != w.sample_rate:
#                     # create filterbank if needed
#                     fb = LogarithmicFilterbank(args.window / 2, w.sample_rate,
#                                                args.bands, args.fmin,
#                                                args.fmax, args.equal)
#             # create a Spectrogram object
#             s = Spectrogram(w, frame_size=args.window, filterbank=fb,
#                             log=args.log, mul=args.mul, add=args.add,
#                             ratio=args.ratio, diff_frames=args.diff_frames,
#                             num_frames=args.length)
#             # create a SpectralNoteTranscription object
#             snt = SpectralNoteTranscription(s)
#             snt.notes()
#         # save note activations or detect notes
#         if args.save:
#             # save the raw ODF activations
#             n.save_activations("%s.%s" % (filename, args.odf))
#         else:
#             # detect the notes
#             n.detect(args.thresholds, combine=args.combine, delay=args.delay,
#                      smooth=args.smooth, pre_avg=args.pre_avg,
#                      post_avg=args.post_avg, pre_max=args.pre_max,
#                      post_max=args.post_max)
#             # write the notes to a file
#             n.write("%s.%s" % (filename, args.ext))
#         # continue with next file
#
# if __name__ == '__main__':
#     main()
