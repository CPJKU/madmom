# encoding: utf-8
"""
This module contains chord recognition related functionality.

"""
from __future__ import absolute_import, division, print_function

import string

from functools import partial
from madmom.processors import SequentialProcessor


def majmin_targets_to_chord_labels(targets, fps):
    """
    Converts a series of major/minor chord targets to human readable chord
    labels. Targets are assumed to be spaced equidistant in time as defined
    by the `fps` parameter (each target represents one 'frame').

    Ids 0-11 encode major chords starting with root 'A', 12-23 minor chords.
    Id 24 represents 'N', the no-chord class.

    Parameters
    ----------
    targets : iterable
        Iterable containing chord class ids.
    fps : float
        Frames per second. Consecutive class

    Returns
    -------
    chord labels : list
        List of tuples of the form (start time, end time, chord label)

    """
    natural = zip([0, 2, 3, 5, 7, 8, 10], string.uppercase[:7])
    sharp = map(lambda v: ((v[0] + 1) % 12, v[1] + '#'), natural)

    semitone_to_label = dict(sharp + natural)

    def pred_to_cl(pred):
        if pred == 24:
            return 'N'
        return '{}:{}'.format(semitone_to_label[pred % 12],
                              'maj' if pred < 12 else 'min')

    spf = 1. / fps
    labels = [(i * spf, pred_to_cl(p)) for i, p in enumerate(targets)]

    # join same consecutive predictions
    prev_label = (None, None)
    uniq_labels = []

    for label in labels:
        if label[1] != prev_label[1]:
            uniq_labels.append(label)
            prev_label = label

    # end time of last label is one frame duration after
    # the last prediction time
    start_times, chord_labels = zip(*uniq_labels)
    end_times = start_times[1:] + (labels[-1][0] + spf,)

    return zip(start_times, end_times, chord_labels)


class DeepChromaChordRecognitionProcessor(SequentialProcessor):
    """
    Recognise major and minor chords from deep chroma vectors [1]_ using a
    Conditional Random Field.

    Parameters
    ----------
    model : str
        File containing the CRF model. If None, use the model supplied with
        madmom.
    fps : float
        Frames per second. Must correspond to the fps of the incoming
        activations and the model.

    References
    ----------
    .. [1] Filip Korzeniowski and Gerhard Widmer,
           "Feature Learning for Chord Recognition: The Deep Chroma Extractor",
           Proceedings of the 17th International Society for Music Information
           Retrieval Conference (ISMIR), 2016.
    """

    def __init__(self, model=None, fps=10, **kwargs):
        from ..ml.crf import ConditionalRandomField
        from ..models import CHORDS_DCCRF
        crf = ConditionalRandomField.load(model or CHORDS_DCCRF[0])
        lbl = partial(majmin_targets_to_chord_labels, fps=fps)
        super(DeepChromaChordRecognitionProcessor, self).__init__((crf, lbl))
