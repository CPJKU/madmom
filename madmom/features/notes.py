#!/usr/bin/env python
# encoding: utf-8
"""
This file contains note transcription related functionality.

@author: Sebastian Böck <sebastian.boeck@jku.at>

"""

import glob
import numpy as np

from madmom import MODELS_PATH
from madmom.utils import open

from . import Activations, RNNEventDetection
from .onsets import peak_picking


def load_notes(filename):
    """
    Load the target notes from a file.

    :param filename: input file name or file handle
    :return:         numpy array with notes

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
#         from madmom.audio.spectrogram import Spectrogram
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


class RNNNoteTranscription(RNNEventDetection):
    """
    Note transcription with RNNs.

    """
    # define NN files
    NN_FILES = glob.glob("%s/notes_brnn*npz" % MODELS_PATH)

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

    def pre_process(self):
        """
        Pre-process the signal to obtain a data representation suitable for RNN
        processing.

        :return: pre-processed data

        """
        spr = super(RNNNoteTranscription, self)
        spr.pre_process(frame_sizes=[1024, 2048, 4096], bands_per_octave=12,
                        mul=5, ratio=0.5)
        # return data
        return self._data

    def process(self):
        """
        Test the data with the defined RNNs.

        :return: note activations

        """
        # process the data
        super(RNNNoteTranscription, self).process()
        # reshape the activations
        self._activations = self._activations.reshape(-1, 88)
        # and return them
        return self._activations

    def detect(self, threshold=THRESHOLD, smooth=SMOOTH, combine=COMBINE,
               delay=DELAY):
        """
        Detect the notes with the given peak-picking parameters.

        :param threshold: threshold for note detection
        :param smooth:    smooth activations over N seconds
        :param combine:   combine note onsets within N seconds
        :param delay:     report note onsets N seconds delayed
        :return:          detected notes

        """
        # convert timing information to frames
        smooth = int(round(self.fps * smooth))
        # detect notes
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
        from .utils import open
        # write the detected notes to the output
        with open(filename, 'wb') as f:
            for note in self.detections:
                f.write(sep.join([str(x) for x in note]) + '\n')

    @classmethod
    def add_arguments(cls, parser, nn_files=NN_FILES, threshold=THRESHOLD,
                      smooth=SMOOTH, combine=COMBINE, delay=DELAY,
                      pre_avg=PRE_AVG, post_avg=POST_AVG,
                      pre_max=PRE_MAX, post_max=POST_MAX):
        """
        Add note transcription related arguments to an existing parser object.

        :param parser:    existing argparse parser object
        :param nn_files:  list with files of NN models
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


# def parser():
#     """
#     Command line argument parser for note transcription.
#
#     """
#     import argparse
#     from madmom.utils.params import (audio, spec, filtering, log, note, save_load)
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
#     from madmom.utils import files
#     from madmom.audio.wav import Wav
#     from madmom.audio.spectrogram import Spectrogram
#     from madmom.audio.filters import LogarithmicFilterbank
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
