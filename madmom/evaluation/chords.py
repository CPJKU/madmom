# encoding: utf-8
"""
This module contains chord evaluation functionality.

It provides the evaluation measures used for the MIREX ACE task, and
tries to follow [1]_ and [2]_ as closely as possible.

Notes
-----
This implementation tries to follow the references and their implementation
(e.g., https://github.com/jpauwels/MusOOEvaluator for [2]_). However, there
are some known (and possibly some unknown) differences. If you find one not
listed in the following, please file an issue:

 - Detected chord segments are adjusted to fit the length of the annotations.
   In particular, this means that, if necessary, filler segments of 'no chord'
   are added at beginnings and ends. This can result in different segmentation
   scores compared to the original implementation.

References
----------
.. [1] Christopher Harte, "Towards Automatic Extraction of Harmony Information
       from Music Signals." Dissertation,
       Department for Electronic Engineering, Queen Mary University of London,
       2010.
.. [2] Johan Pauwels and Geoffroy Peeters.
       "Evaluating Automatically Estimated Chord Sequences."
       In Proceedings of ICASSP 2013, Vancouver, Canada, 2013.

"""

import numpy as np

from . import evaluation_io, EvaluationMixin
from ..io import load_chords


CHORD_DTYPE = [('root', np.int),
               ('bass', np.int),
               ('intervals', np.int, (12,))]

CHORD_ANN_DTYPE = [('start', np.float),
                   ('end', np.float),
                   ('chord', CHORD_DTYPE)]

NO_CHORD = (-1, -1, np.zeros(12, dtype=np.int))
UNKNOWN_CHORD = (-1, -1, np.ones(12, dtype=np.int) * -1)


def encode(chord_labels):
    """
    Encodes chord labels to numeric interval representations.

    Parameters
    ----------
    chord_labels : numpy structured array
        Chord segments in `madmom.io.SEGMENT_DTYPE` format

    Returns
    -------
    encoded_chords : numpy structured array
        Chords in `CHORD_ANN_DTYPE` format

    """
    encoded_chords = np.zeros(len(chord_labels), dtype=CHORD_ANN_DTYPE)
    encoded_chords['start'] = chord_labels['start']
    encoded_chords['end'] = chord_labels['end']
    encoded_chords['chord'] = chords(chord_labels['label'])
    return encoded_chords


def chords(labels):
    """
    Transform a list of chord labels into an array of internal numeric
    representations.

    Parameters
    ----------
    labels : list
        List of chord labels (str).

    Returns
    -------
    chords : numpy.array
        Structured array with columns 'root', 'bass', and 'intervals',
        containing a numeric representation of chords (`CHORD_DTYPE`).

    """
    crds = np.zeros(len(labels), dtype=CHORD_DTYPE)
    cache = {}
    for i, lbl in enumerate(labels):
        cv = cache.get(lbl, None)
        if cv is None:
            cv = chord(lbl)
            cache[lbl] = cv
        crds[i] = cv
    return crds


def chord(label):
    """
    Transform a chord label into the internal numeric represenation of
    (root, bass, intervals array) as defined by `CHORD_DTYPE`.

    Parameters
    ----------
    label : str
        Chord label.

    Returns
    -------
    chord : tuple
        Numeric representation of the chord: (root, bass, intervals array).

    """
    if label == 'N':
        return NO_CHORD
    if label == 'X':
        return UNKNOWN_CHORD

    c_idx = label.find(':')
    s_idx = label.find('/')

    if c_idx == -1:
        quality_str = 'maj'
        if s_idx == -1:
            root_str = label
            bass_str = ''
        else:
            root_str = label[:s_idx]
            bass_str = label[s_idx + 1:]
    else:
        root_str = label[:c_idx]
        if s_idx == -1:
            quality_str = label[c_idx + 1:]
            bass_str = ''
        else:
            quality_str = label[c_idx + 1:s_idx]
            bass_str = label[s_idx + 1:]

    root = pitch(root_str)
    bass = interval(bass_str) if bass_str else 0
    ivs = chord_intervals(quality_str)
    ivs[bass] = 1

    return root, bass, ivs


_l = [0, 1, 1, 0, 1, 1, 1]
_chroma_id = (np.arange(len(_l) * 2) + 1) + np.array(_l + _l).cumsum() - 1


def modify(base_pitch, modifier):
    """
    Modify a pitch class in integer representation by a given modifier string.

    A modifier string can be any sequence of 'b' (one semitone down)
    and '#' (one semitone up).

    Parameters
    ----------
    base_pitch : int
        Pitch class as integer.
    modifier : str
        String of modifiers ('b' or '#').

    Returns
    -------
    modified_pitch : int
        Modified root note.

    """
    for m in modifier:
        if m == 'b':
            base_pitch -= 1
        elif m == '#':
            base_pitch += 1
        else:
            raise ValueError('Unknown modifier: {}'.format(m))
    return base_pitch


def pitch(pitch_str):
    """
    Convert a string representation of a pitch class (consisting of root
    note and modifiers) to an integer representation.

    Parameters
    ----------
    pitch_str : str
        String representation of a pitch class.

    Returns
    -------
    pitch : int
        Integer representation of a pitch class.

    """
    return modify(_chroma_id[(ord(pitch_str[0]) - ord('C')) % 7],
                  pitch_str[1:]) % 12


def interval(interval_str):
    """
    Convert a string representation of a musical interval into a pitch class
    (e.g. a minor seventh 'b7' into 10, because it is 10 semitones above its
    base note).

    Parameters
    ----------
    interval_str : str
        Musical interval.

    Returns
    -------
    pitch_class : int
        Number of semitones to base note of interval.

    """
    for i, c in enumerate(interval_str):
        if c.isdigit():
            return modify(_chroma_id[int(interval_str[i:]) - 1],
                          interval_str[:i]) % 12


def interval_list(intervals_str, given_pitch_classes=None):
    """
    Convert a list of intervals given as string to a binary pitch class
    representation. For example, 'b3, 5' would become
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0].

    Parameters
    ----------
    intervals_str : str
        List of intervals as comma-separated string (e.g. 'b3, 5').
    given_pitch_classes : None or numpy array
        If None, start with empty pitch class array, if numpy array of length
        12, this array will be modified.

    Returns
    -------
    pitch_classes : numpy array
        Binary pitch class representation of intervals.

    """
    if given_pitch_classes is None:
        given_pitch_classes = np.zeros(12, dtype=np.int)
    for int_def in intervals_str[1:-1].split(','):
        int_def = int_def.strip()
        if int_def[0] == '*':
            given_pitch_classes[interval(int_def[1:])] = 0
        else:
            given_pitch_classes[interval(int_def)] = 1
    return given_pitch_classes


# mapping of shorthand interval notations to the actual interval representation
_shorthands = {
    'maj': interval_list('(1,3,5)'),
    'min': interval_list('(1,b3,5)'),
    'dim': interval_list('(1,b3,b5)'),
    'aug': interval_list('(1,3,#5)'),
    'maj7': interval_list('(1,3,5,7)'),
    'min7': interval_list('(1,b3,5,b7)'),
    '7': interval_list('(1,3,5,b7)'),
    '5': interval_list('(1,5)'),
    '1': interval_list('(1)'),
    'dim7': interval_list('(1,b3,b5,bb7)'),
    'hdim7': interval_list('(1,b3,b5,b7)'),
    'minmaj7': interval_list('(1,b3,5,7)'),
    'maj6': interval_list('(1,3,5,6)'),
    'min6': interval_list('(1,b3,5,6)'),
    '9': interval_list('(1,3,5,b7,9)'),
    'maj9': interval_list('(1,3,5,7,9)'),
    'min9': interval_list('(1,b3,5,b7,9)'),
    'sus2': interval_list('(1,2,5)'),
    'sus4': interval_list('(1,4,5)'),
    '11': interval_list('(1,3,5,b7,9,11)'),
    'min11': interval_list('(1,b3,5,b7,9,11)'),
    '13': interval_list('(1,3,5,b7,13)'),
    'maj13': interval_list('(1,3,5,7,13)'),
    'min13': interval_list('(1,b3,5,b7,13)')
}


def chord_intervals(quality_str):
    """
    Convert a chord quality string to a pitch class representation. For
    example, 'maj' becomes [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0].

    Parameters
    ----------
    quality_str : str
        String defining the chord quality.

    Returns
    -------
    pitch_classes : numpy array
        Binary pitch class representation of chord quality.

    """
    list_idx = quality_str.find('(')
    if list_idx == -1:
        return _shorthands[quality_str].copy()
    if list_idx != 0:
        ivs = _shorthands[quality_str[:list_idx]].copy()
    else:
        ivs = np.zeros(12, dtype=np.int)

    return interval_list(quality_str[list_idx:], ivs)


def merge_chords(chords):
    """
    Merge consecutive chord annotations if they represent the same chord.

    Parameters
    ----------
    chords : numpy structured arrray
        Chord annotations to be merged, in `CHORD_ANN_DTYPE` format.

    Returns
    -------
    merged_chords : numpy structured array
        Merged chord annotations, in `CHORD_ANN_DTYPE` format.

    """
    merged_starts = []
    merged_ends = []
    merged_chords = []
    prev_chord = None
    for start, end, chord in chords:
        if chord != prev_chord:
            prev_chord = chord
            merged_starts.append(start)
            merged_ends.append(end)
            merged_chords.append(chord)
        else:
            # prolong the previous chord
            merged_ends[-1] = end

    crds = np.zeros(len(merged_chords), dtype=CHORD_ANN_DTYPE)
    crds['start'] = merged_starts
    crds['end'] = merged_ends
    crds['chord'] = merged_chords
    return crds


def evaluation_pairs(det_chords, ann_chords):
    """
    Match detected with annotated chords and create paired label segments
    for evaluation.

    Parameters
    ----------
    det_chords : numpy structured array
        Chord detections with 'start' and 'end' fields.
    ann_chords : numpy structured array
        Chord annotations with 'start' and 'end' fields.

    Returns
    -------
    annotations : numpy structured array
        Annotated chords of evaluation segments.
    detections : numpy structured array
        Detected chords of evaluation segments.
    durations : numpy array
        Durations of evaluation segments.

    """
    times = np.unique(np.hstack([ann_chords['start'], ann_chords['end'],
                                 det_chords['start'], det_chords['end']]))

    durations = times[1:] - times[:-1]
    annotations = ann_chords['chord'][
        np.searchsorted(ann_chords['start'], times[:-1], side='right') - 1]
    detections = det_chords['chord'][
        np.searchsorted(det_chords['start'], times[:-1], side='right') - 1]

    return annotations, detections, durations


def score_root(det_chords, ann_chords):
    """
    Score similarity of chords based on only the root, i.e. returns a score of
    1 if roots match, 0 otherwise.

    Parameters
    ----------
    det_chords : numpy structured array
        Detected chords.
    ann_chords : numpy structured array
        Annotated chords.

    Returns
    -------
    scores : numpy array
        Similarity score for each chord.

    """
    return (ann_chords['root'] == det_chords['root']).astype(np.float)


def score_exact(det_chords, ann_chords):
    """
    Score similarity of chords. Returns 1 if all chord information (root,
    bass, and intervals) match exactly.

    Parameters
    ----------
    det_chords : numpy structured array
        Detected chords.
    ann_chords : numpy structured array
        Annotated chords.

    Returns
    -------
    scores : numpy array
        Similarity score for each chord.

    """
    return ((ann_chords['root'] == det_chords['root']) &
            (ann_chords['bass'] == det_chords['bass']) &
            ((ann_chords['intervals'] == det_chords['intervals']).all(axis=1))
            ).astype(np.float)


def reduce_to_triads(chords, keep_bass=False):
    """
    Reduce chords to triads.

    The function follows the reduction rules implemented in [1]_. If a chord
    chord does not contain a third, major second or fourth, it is reduced to
    a power chord. If it does not contain neither a third nor a fifth, it is
    reduced to a single note "chord".

    Parameters
    ----------
    chords : numpy structured array
        Chords to be reduced.
    keep_bass : bool
        Indicates whether to keep the bass note or set it to 0.

    Returns
    -------
    reduced_chords : numpy structured array
        Chords reduced to triads.

    References
    ----------
    .. [1] Johan Pauwels and Geoffroy Peeters.
           "Evaluating Automatically Estimated Chord Sequences."
           In Proceedings of ICASSP 2013, Vancouver, Canada, 2013.

    """
    unison = chords['intervals'][:, 0].astype(bool)
    maj_sec = chords['intervals'][:, 2].astype(bool)
    min_third = chords['intervals'][:, 3].astype(bool)
    maj_third = chords['intervals'][:, 4].astype(bool)
    perf_fourth = chords['intervals'][:, 5].astype(bool)
    dim_fifth = chords['intervals'][:, 6].astype(bool)
    perf_fifth = chords['intervals'][:, 7].astype(bool)
    aug_fifth = chords['intervals'][:, 8].astype(bool)
    no_chord = (chords['intervals'] == NO_CHORD[-1]).all(axis=1)

    reduced_chords = chords.copy()
    ivs = reduced_chords['intervals']

    ivs[~no_chord] = interval_list('(1)')
    ivs[unison & perf_fifth] = interval_list('(1,5)')
    ivs[~perf_fourth & maj_sec] = _shorthands['sus2']
    ivs[perf_fourth & ~maj_sec] = _shorthands['sus4']

    ivs[min_third] = _shorthands['min']
    ivs[min_third & aug_fifth & ~perf_fifth] = interval_list('(1,b3,#5)')
    ivs[min_third & dim_fifth & ~perf_fifth] = _shorthands['dim']

    ivs[maj_third] = _shorthands['maj']
    ivs[maj_third & dim_fifth & ~perf_fifth] = interval_list('(1,3,b5)')
    ivs[maj_third & aug_fifth & ~perf_fifth] = _shorthands['aug']

    if not keep_bass:
        reduced_chords['bass'] = 0
    else:
        # remove bass notes if they are not part of the intervals anymore
        reduced_chords['bass'] *= ivs[range(len(reduced_chords)),
                                      reduced_chords['bass']]
    # keep -1 in bass for no chords
    reduced_chords['bass'][no_chord] = -1

    return reduced_chords


def reduce_to_tetrads(chords, keep_bass=False):
    """
    Reduce chords to tetrads.

    The function follows the reduction rules implemented in [1]_. If a chord
    does not contain a third, major second or fourth, it is reduced to a power
    chord. If it does not contain neither a third nor a fifth, it is reduced
    to a single note "chord".

    Parameters
    ----------
    chords : numpy structured array
        Chords to be reduced.
    keep_bass : bool
        Indicates whether to keep the bass note or set it to 0.

    Returns
    -------
    reduced_chords : numpy structured array
        Chords reduced to tetrads.

    References
    ----------
    .. [1] Johan Pauwels and Geoffroy Peeters.
           "Evaluating Automatically Estimated Chord Sequences."
           In Proceedings of ICASSP 2013, Vancouver, Canada, 2013.

    """
    unison = chords['intervals'][:, 0].astype(bool)
    maj_sec = chords['intervals'][:, 2].astype(bool)
    min_third = chords['intervals'][:, 3].astype(bool)
    maj_third = chords['intervals'][:, 4].astype(bool)
    perf_fourth = chords['intervals'][:, 5].astype(bool)
    dim_fifth = chords['intervals'][:, 6].astype(bool)
    perf_fifth = chords['intervals'][:, 7].astype(bool)
    aug_fifth = chords['intervals'][:, 8].astype(bool)
    maj_sixth = chords['intervals'][:, 9].astype(bool)
    dim_seventh = maj_sixth
    min_seventh = chords['intervals'][:, 10].astype(bool)
    maj_seventh = chords['intervals'][:, 11].astype(bool)
    no_chord = (chords['intervals'] == NO_CHORD[-1]).all(axis=1)

    reduced_chords = chords.copy()
    ivs = reduced_chords['intervals']

    ivs[~no_chord] = interval_list('(1)')
    ivs[unison & perf_fifth] = interval_list('(1,5)')

    sus2 = ~perf_fourth & maj_sec
    sus2_ivs = _shorthands['sus2']
    ivs[sus2] = sus2_ivs
    ivs[sus2 & maj_sixth] = interval_list('(6)', sus2_ivs.copy())
    ivs[sus2 & maj_seventh] = interval_list('(7)', sus2_ivs.copy())
    ivs[sus2 & min_seventh] = interval_list('(b7)', sus2_ivs.copy())

    sus4 = perf_fourth & ~maj_sec
    sus4_ivs = _shorthands['sus4']
    ivs[sus4] = sus4_ivs
    ivs[sus4 & maj_sixth] = interval_list('(6)', sus4_ivs.copy())
    ivs[sus4 & maj_seventh] = interval_list('(7)', sus4_ivs.copy())
    ivs[sus4 & min_seventh] = interval_list('(b7)', sus4_ivs.copy())

    ivs[min_third] = _shorthands['min']
    ivs[min_third & maj_sixth] = _shorthands['min6']
    ivs[min_third & maj_seventh] = _shorthands['minmaj7']
    ivs[min_third & min_seventh] = _shorthands['min7']
    minaugfifth = min_third & ~perf_fifth & aug_fifth
    ivs[minaugfifth] = interval_list('(1,b3,#5)')
    ivs[minaugfifth & maj_seventh] = interval_list('(1,b3,#5,7)')
    ivs[minaugfifth & min_seventh] = interval_list('(1,b3,#5,b7)')
    mindimfifth = min_third & ~perf_fifth & dim_fifth
    ivs[mindimfifth] = _shorthands['dim']
    ivs[mindimfifth & dim_seventh] = _shorthands['dim7']
    ivs[mindimfifth & min_seventh] = _shorthands['hdim7']

    ivs[maj_third] = _shorthands['maj']
    ivs[maj_third & maj_sixth] = _shorthands['maj6']
    ivs[maj_third & maj_seventh] = _shorthands['maj7']
    ivs[maj_third & min_seventh] = _shorthands['7']
    majdimfifth = maj_third & ~perf_fifth & dim_fifth
    ivs[majdimfifth] = interval_list('(1,3,b5)')
    ivs[majdimfifth & maj_seventh] = interval_list('(1,3,b5,7)')
    ivs[majdimfifth & min_seventh] = interval_list('(1,3,b5,b7)')
    majaugfifth = maj_third & ~perf_fifth & aug_fifth
    aug_ivs = _shorthands['aug']
    ivs[majaugfifth] = _shorthands['aug']
    ivs[majaugfifth & maj_seventh] = interval_list('(7)', aug_ivs.copy())
    ivs[majaugfifth & min_seventh] = interval_list('(b7)', aug_ivs.copy())

    if not keep_bass:
        reduced_chords['bass'] = 0
    else:
        # remove bass notes if they are not part of the intervals anymore
        reduced_chords['bass'] *= ivs[range(len(reduced_chords)),
                                      reduced_chords['bass']]
    # keep -1 in bass for no chords
    reduced_chords['bass'][no_chord] = -1

    return reduced_chords


def select_majmin(chords):
    """
    Compute a mask that selects all major, minor, and
    "no chords" with a 1, and all other chords with a 0.

    Parameters
    ----------
    chords : numpy structured array
        Chords to compute the mask for.

    Returns
    -------
    mask : numpy array (boolean)
        Selection mask for major, minor, and "no chords".

    """
    return ((chords['intervals'] == _shorthands['maj']).all(axis=1) |
            (chords['intervals'] == _shorthands['min']).all(axis=1) |
            (chords['intervals'] == NO_CHORD[-1]).all(axis=1))


def select_sevenths(chords):
    """
    Compute a mask that selects all major, minor, seventh, and
    "no chords" with a 1, and all other chords with a 0.

    Parameters
    ----------
    chords : numpy structured array
        Chords to compute the mask for.

    Returns
    -------
    mask : numpy array (boolean)
        Selection mask for major, minor, seventh, and "no chords".

    """
    return (select_majmin(chords) |
            (chords['intervals'] == _shorthands['7']).all(axis=1) |
            (chords['intervals'] == _shorthands['min7']).all(axis=1) |
            (chords['intervals'] == _shorthands['maj7']).all(axis=1))


def adjust(det_chords, ann_chords):
    """
    Adjust the length of detected chord segments to the annotation
    length.

    Discard detected chords that start after the annotation ended,
    and shorten the last detection to fit the last annotation;
    discared detected chords that end before the annotation begins,
    and shorten the first detection to match the first annotation.

    Parameters
    ----------
    det_chords : numpy structured array
        Detected chord segments.
    ann_chords : numpy structured array
        Annotated chord segments.

    Returns
    -------
    det_chords : numpy structured array
        Adjusted detected chord segments.

    """
    det_start = det_chords[0]['start']
    ann_start = ann_chords[0]['start']
    if det_start > ann_start:
        filler = np.array((ann_start, det_start, chord('N')),
                          dtype=CHORD_ANN_DTYPE)
        det_chords = np.hstack([filler, det_chords])
    elif det_start < ann_start:
        det_chords = det_chords[det_chords['end'] > ann_start]
        det_chords[0]['start'] = ann_start

    det_end = det_chords[-1]['end']
    ann_end = ann_chords[-1]['end']
    if det_end < ann_end:
        filler = np.array((det_end, ann_end, chord('N')),
                          dtype=CHORD_ANN_DTYPE)
        det_chords = np.hstack([det_chords, filler])
    elif det_end > ann_end:
        det_chords = det_chords[det_chords['start'] < ann_end]
        det_chords[-1]['end'] = ann_chords[-1]['end']

    return det_chords


def segmentation(ann_starts, ann_ends, det_starts, det_ends):
    """
    Compute the normalized Hamming divergence between chord
    segmentations as defined in [1]_ (Eqs. 8.37 and 8.38).

    Parameters
    ----------
    ann_starts : list or numpy array
        Start times of annotated chord segments.
    ann_ends : list or numpy array
        End times of annotated chord segments.
    det_starts : list or numpy array
        Start times of detected chord segments.
    det_ends : list or numpy array
        End times of detected chord segments.

    Returns
    -------
    distance : float
        Normalised Hamming divergence between annotated and
        detected chord segments.

    References
    ----------
    .. [1] Christopher Harte, "Towards Automatic Extraction of Harmony
           Information from Music Signals." Dissertation,
           Department for Electronic Engineering, Queen Mary University of
           London, 2010.

    """
    est_ts = np.unique(np.hstack([det_starts, det_ends]))
    seg = 0.
    for start, end in zip(ann_starts, ann_ends):
        dur = end - start
        seg_ts = np.hstack([
            start, est_ts[(est_ts > start) & (est_ts < end)], end])
        seg += dur - np.diff(seg_ts).max()

    return seg / (ann_ends[-1] - ann_starts[0])


class ChordEvaluation(EvaluationMixin):
    """
    Provide various chord evaluation scores.

    Parameters
    ----------
    detections : str
        File containing chords detections.
    annotations : str
        File containing chord annotations.
    name : str, optional
        Name of the evaluation object (e.g., the name of the song).

    """

    METRIC_NAMES = [
        ('root', 'Root'),
        ('majmin', 'MajMin'),
        ('majminbass', 'MajMinBass'),
        ('sevenths', 'Sevenths'),
        ('seventhsbass', 'SeventhsBass'),
        ('segmentation', 'Segmentation'),
        ('oversegmentation', 'OverSegmentation'),
        ('undersegmentation', 'UnderSegmentation'),
    ]

    def __init__(self, detections, annotations, name=None, **kwargs):
        self.name = name or ''
        self.ann_chords = merge_chords(encode(annotations))
        self.det_chords = merge_chords(adjust(encode(detections),
                                              self.ann_chords))
        self.annotations, self.detections, self.durations = evaluation_pairs(
            self.det_chords, self.ann_chords)
        self._underseg = None
        self._overseg = None

    @property
    def length(self):
        """Length of annotations."""
        return self.ann_chords['end'][-1] - self.ann_chords['start'][0]

    @property
    def root(self):
        """Fraction of correctly detected chord roots."""
        return np.average(score_root(self.detections, self.annotations),
                          weights=self.durations)

    @property
    def majmin(self):
        """
        Fraction of correctly detected chords that can be reduced to major
        or minor triads (plus no-chord). Ignores the bass pitch class.
        """
        det_triads = reduce_to_triads(self.detections)
        ann_triads = reduce_to_triads(self.annotations)
        majmin_sel = select_majmin(ann_triads)
        return np.average(score_exact(det_triads, ann_triads),
                          weights=self.durations * majmin_sel)

    @property
    def majminbass(self):
        """
        Fraction of correctly detected chords that can be reduced to major
        or minor triads (plus no-chord). Considers the bass pitch class.
        """
        det_triads = reduce_to_triads(self.detections, keep_bass=True)
        ann_triads = reduce_to_triads(self.annotations, keep_bass=True)
        majmin_sel = select_majmin(ann_triads)
        return np.average(score_exact(det_triads, ann_triads),
                          weights=self.durations * majmin_sel)

    @property
    def sevenths(self):
        """
        Fraction of correctly detected chords that can be reduced to a seventh
        tetrad (plus no-chord). Ignores the bass pitch class.
        """
        det_tetrads = reduce_to_tetrads(self.detections)
        ann_tetrads = reduce_to_tetrads(self.annotations)
        sevenths_sel = select_sevenths(ann_tetrads)
        return np.average(score_exact(det_tetrads, ann_tetrads),
                          weights=self.durations * sevenths_sel)

    @property
    def seventhsbass(self):
        """
        Fraction of correctly detected chords that can be reduced to a seventh
        tetrad (plus no-chord). Considers the bass pitch class.
        """
        det_tetrads = reduce_to_tetrads(self.detections, keep_bass=True)
        ann_tetrads = reduce_to_tetrads(self.annotations, keep_bass=True)
        sevenths_sel = select_sevenths(ann_tetrads)
        return np.average(score_exact(det_tetrads, ann_tetrads),
                          weights=self.durations * sevenths_sel)

    @property
    def undersegmentation(self):
        """
        Normalized Hamming divergence (directional) between annotations and
        detections. Captures missed chord segments.
        """
        if self._underseg is None:
            self._underseg = 1 - segmentation(
                self.det_chords['start'], self.det_chords['end'],
                self.ann_chords['start'], self.ann_chords['end'],
            )
        return self._underseg

    @property
    def oversegmentation(self):
        """
        Normalized Hamming divergence (directional) between detections and
        annotations. Captures how fragmented the detected chord segments are.
        """
        if self._overseg is None:
            self._overseg = 1 - segmentation(
                self.ann_chords['start'], self.ann_chords['end'],
                self.det_chords['start'], self.det_chords['end'],
            )
        return self._overseg

    @property
    def segmentation(self):
        """Minimum of `oversegmentation` and `undersegmentation`."""
        return min(self.undersegmentation, self.oversegmentation)

    def tostring(self, **kwargs):
        """
        Format the evaluation metrics as a human readable string.

        Returns
        -------
        eval_string : str
            Evaluation metrics formatted as a human readable string.

        """
        ret = (
            '{}\n'
            '  Root: {:5.2f} MajMin: {:5.2f} MajMinBass: {:5.2f} '
            'Sevenths: {:5.2f} SeventhsBass: {:5.2f}\n'
            '  Seg: {:5.2f} UnderSeg: {:5.2f} OverSeg: {:5.2f}'.format(
                self.name,
                self.root * 100, self.majmin * 100, self.majminbass * 100,
                self.sevenths * 100, self.seventhsbass * 100,
                self.segmentation * 100, self.undersegmentation * 100,
                self.oversegmentation * 100)
        )
        return ret


class ChordSumEvaluation(ChordEvaluation):
    """
    Class for averaging Chord evaluation scores, considering the lengths
    of the pieces. For a detailed description of the available metrics,
    refer to ChordEvaluation.

    Parameters
    ----------
    eval_objects : list
        Evaluation objects.
    name : str, optional
        Name to be displayed.

    """
    # pylint: disable=super-init-not-called

    def __init__(self, eval_objects, name=None):
        self.name = name or 'weighted mean for %d files' % len(eval_objects)

        self.annotations = np.hstack([e.annotations for e in eval_objects])
        self.detections = np.hstack([e.detections for e in eval_objects])
        self.durations = np.hstack([e.durations for e in eval_objects])

        un_segs = [e.undersegmentation for e in eval_objects]
        over_segs = [e.oversegmentation for e in eval_objects]
        segs = [e.segmentation for e in eval_objects]
        lens = [e.length for e in eval_objects]

        self._underseg = np.average(un_segs, weights=lens)
        self._overseg = np.average(over_segs, weights=lens)
        self._seg = np.average(segs, weights=lens)
        self._length = sum(lens)

    def length(self):
        """Length of all evaluation objects."""
        return self._length

    @property
    def segmentation(self):
        return self._seg


class ChordMeanEvaluation(ChordEvaluation):
    """
    Class for averaging chord evaluation scores, averaging piecewise (i.e.
    ignoring the lengths of the pieces). For a detailed description of the
    available metrics, refer to ChordEvaluation.

    Parameters
    ----------
    eval_objects : list
        Evaluation objects.
    name : str, optional
        Name to be displayed.

    """
    # pylint: disable=super-init-not-called

    def __init__(self, eval_objects, name=None):
        self.name = name or 'piecewise mean for %d files' % len(eval_objects)
        self.eval_objects = eval_objects

    def length(self):
        """Number of evaluation objects."""
        return len(self.eval_objects)

    @property
    def root(self):
        return np.mean([e.root for e in self.eval_objects])

    @property
    def majmin(self):
        return np.mean([e.majmin for e in self.eval_objects])

    @property
    def majminbass(self):
        return np.mean([e.majminbass for e in self.eval_objects])

    @property
    def sevenths(self):
        return np.mean([e.sevenths for e in self.eval_objects])

    @property
    def seventhsbass(self):
        return np.mean([e.seventhsbass for e in self.eval_objects])

    @property
    def undersegmentation(self):
        return np.mean([e.undersegmentation for e in self.eval_objects])

    @property
    def oversegmentation(self):
        return np.mean([e.oversegmentation for e in self.eval_objects])

    @property
    def segmentation(self):
        return np.mean([e.segmentation for e in self.eval_objects])


def add_parser(parser):
    """
    Add a chord evaluation sub-parser to an existing parser.

    Parameters
    ----------
    parser : argparse parser instance
        Existing argparse parser object.

    Returns
    -------
    sub_parser : argparse sub-parser instance
        Chord evaluation sub-parser.

    """
    import argparse
    # add chord evaluation sub-parser to the existing parser
    p = parser.add_parser(
        'chords', help='chord evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
    This program evaluates pairs of files containing the chord annotations and
    predictions. Suffixes can be given to filter them from the list of files.

    Each line represents a chord and must have the following format with values
    being separated by whitespace (chord_label follows the syntax as defined
    by Harte 2010):
    `start_time end_time chord_label`
    ''')
    # set defaults
    p.set_defaults(eval=ChordEvaluation, sum_eval=ChordSumEvaluation,
                   mean_eval=ChordMeanEvaluation, load_fn=load_chords)
    # file I/O
    evaluation_io(p, ann_suffix='.chords', det_suffix='.chords.txt')
    # return the sub-parser and evaluation argument group
    return p
