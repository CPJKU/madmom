import numpy as np
from madmom.evaluation import evaluation_io

CHORD_DTYPE = [('root', np.int), ('bass', np.int), ('intervals', np.int, (12,))]

NO_CHORD = (-1, -1, np.zeros(12, dtype=np.int))
UNKNOWN_CHORD = (-1, -1, np.ones(12, dtype=np.int) * -1)


def chords(labels):
    """
    Transform a list of chord labels into an array of internal numeric
    representations.

    Parameters
    ----------
    labels : list
        List of chord labels (str)

    Returns
    -------
    numpy.array
        Structured array with columns 'root', 'bass', and 'intervals',
        containing a numeric representation of chords.

    """
    crds = np.zeros(len(labels), dtype=CHORD_DTYPE)
    cache = {}
    for i, lbl in enumerate(labels):
        cv = cache.get(lbl, None)
        if cv is None:
            cv = chord(lbl)
            cache[lbl] = cv
            crds[i] = cv
        else:
            crds[i] = cv

    return crds


def chord(label):
    """
    Transform a chord label into the internal numeric represenation of
    (root, bass, intervals array).

    Parameters
    ----------
    label : str
        Chord label

    Returns
    -------
    tuple
        Numeric representation of the chord: (root, bass, intervals array)

    """
    if label == 'N':
        return NO_CHORD
    if label == 'X':
        return UNKNOWN_CHORD

    c_idx = label.find(':')
    s_idx = label.find('/')

    if c_idx == -1:
        int_str = 'maj'
        if s_idx == -1:
            rt_str = label
            bs_str = ''
        else:
            rt_str = label[:s_idx]
            bs_str = label[s_idx + 1:]
    else:
        rt_str = label[:c_idx]
        if s_idx == -1:
            int_str = label[c_idx + 1:]
            bs_str = ''
        else:
            int_str = label[c_idx + 1:s_idx]
            bs_str = label[s_idx + 1:]

    root = pitch(rt_str)
    bass = interval(bs_str) if bs_str else 0
    ints = intervals(int_str)
    ints[bass] = 1

    return root, bass, ints


_l = [0, 1, 1, 0, 1, 1, 1]
_chroma_id = (np.arange(len(_l) * 2) + 1) + np.array(_l + _l).cumsum() - 1


def modify(base, modstr):
    """
    Modify a pitch class in integer representation by a given modifier string.
    A modifier string can be any sequence of 'b' (one semitone down)
    and '#' (one semitone up).

    Parameters
    ----------
    base : int
        Pitch class as integer
    modstr : str
        String of modifiers ('b' or '#')

    Returns
    -------
    int
        Modified root note

    """
    for m in modstr:
        if m == 'b':
            base -= 1
        elif m == '#':
            base += 1
        else:
            raise ValueError('Unknown modifier: {}'.format(m))
    return base


def pitch(s):
    """
    Converts a string representation of a pitch class (consisting of root
    note and modifiers) to an integer representation.

    Parameters
    ----------
    s : str
        String representation of a pitch class

    Returns
    -------
    int
        Integer representation of a pitch class

    """
    return modify(_chroma_id[(ord(s[0]) - ord('C')) % 7], s[1:]) % 12


def interval(s):
    """
    Converts a string representation of a musical interval into a semitone
    integer (e.g. a minor seventh 'b7' into 10, because it is 10 semitones
    above its base note).

    Parameters
    ----------
    s : str
        Musical interval

    Returns
    -------
    int
        Number of semitones to base note of interval

    """
    for i, c in enumerate(s):
        if c.isdigit():
            return modify(_chroma_id[int(s[i:]) - 1], s[:i]) % 12


def interval_list(s, intervals=None):
    """
    Convert a list of intervals given as string to a binary semitone array
    representation. For example, 'b3, 5' would become
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0].

    Parameters
    ----------
    s : str
        List of intervals as comma-separated string (e.g. 'b3, 5')

    intervals : None or numpy array
        If None, start with empty interval array, if numpy array of length
        12, this array will be modified.

    Returns
    -------
    numpy array
        Binary semitone representation of intervals

    """
    intervals = intervals if intervals is not None else np.zeros(12, dtype=np.int)
    for int_def in s[1:-1].split(','):
        int_def = int_def.strip()
        if int_def[0] == '*':
            intervals[interval(int_def[1:])] = 0
        else:
            intervals[interval(int_def)] = 1
    return intervals

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
    'min6': interval_list('(1,b3,5,b6)'),
    '9': interval_list('(1,3,5,b7,9)'),
    'maj9': interval_list('(1,3,5,7,9)'),
    'min9': interval_list('(1,b3,5,b7,9)'),
    'sus2': interval_list('(1,2,5)'),
    'sus4': interval_list('(1,4,5)'),
}


def intervals(s):
    list_idx = s.find('(')
    if list_idx == -1:
        return _shorthands[s].copy()
    if list_idx != 0:
        ivs = _shorthands[s[:list_idx]].copy()
    else:
        ivs = np.zeros(12, dtype=np.int)

    return interval_list(s[list_idx:], ivs)


def load_chords(filename):
    start, end, chord_labels = np.loadtxt(
        filename,
        dtype=[('start', np.float),
               ('end', np.float),
               ('chord', 'U32')],
        unpack=True,
        comments=''
    )
    crds = np.zeros(len(start), dtype=[('start', np.float),
                                       ('end', np.float),
                                       ('chord', CHORD_DTYPE)])
    crds['start'] = start
    crds['end'] = end
    crds['chord'] = chords(chord_labels)
    return crds


def evaluation_pairs(est_chords, ref_chords):
    times = np.unique(np.hstack([ref_chords['start'], ref_chords['end'],
                                 est_chords['start'], est_chords['end']]))

    pairs = np.zeros(len(times) - 1, dtype=[('duration', np.float),
                                            ('ref_chord', CHORD_DTYPE),
                                            ('est_chord', CHORD_DTYPE)])

    pairs['duration'] = times[1:] - times[:-1]
    pairs['ref_chord'] = ref_chords['chord'][
        np.searchsorted(ref_chords['start'], times[:-1], side='right') - 1]
    pairs['est_chord'] = est_chords['chord'][
        np.searchsorted(est_chords['start'], times[:-1], side='right') - 1]

    return pairs


def score_root(pairs):
    return (pairs['ref_chord']['root'] == pairs['est_chord']['root']).astype(np.float)


def score_exact(pairs):
    return ((pairs['ref_chord']['root'] == pairs['est_chord']['root']) &
            (pairs['ref_chord']['bass'] == pairs['est_chord']['bass']) &
            ((pairs['ref_chord']['intervals'] ==
              pairs['est_chord']['intervals']).all(axis=1))).astype(np.float)


def map_triads(chords, keep_bass=False):
    unison = chords['intervals'][:, 0].astype(bool)
    maj_sec = chords['intervals'][:, 2].astype(bool)
    min_third = chords['intervals'][:, 3].astype(bool)
    maj_third = chords['intervals'][:, 4].astype(bool)
    perf_fourth = chords['intervals'][:, 5].astype(bool)
    dim_fifth = chords['intervals'][:, 6].astype(bool)
    perf_fifth = chords['intervals'][:, 7].astype(bool)
    aug_fifth = chords['intervals'][:, 8].astype(bool)

    no_chord = (chords['intervals'] == NO_CHORD[-1]).all(axis=1)

    mapped_chords = chords.copy()
    if not keep_bass:
        mapped_chords['bass'] = 0
    ivs = mapped_chords['intervals']

    ivs[~no_chord] = interval_list('(1)')
    ivs[unison & perf_fifth] = interval_list('(1,5)')
    ivs[~perf_fourth & maj_sec] = _shorthands['sus4']
    ivs[perf_fourth & ~maj_sec] = _shorthands['sus4']

    ivs[min_third & perf_fifth] = _shorthands['min']
    ivs[min_third & aug_fifth & ~perf_fifth] = interval_list('(1,b3,#5)')
    ivs[min_third & dim_fifth & ~perf_fifth] = _shorthands['dim']

    ivs[maj_third & perf_fifth] = _shorthands['maj']
    ivs[maj_third & dim_fifth & ~perf_fifth] = interval_list('(1,3,b5)')
    ivs[maj_third & aug_fifth & ~perf_fifth] = _shorthands['aug']

    return mapped_chords


def select_majmin(chords):
    return ((chords['intervals'] == _shorthands['maj']).all(axis=1) |
            (chords['intervals'] == _shorthands['min']).all(axis=1) |
            (chords['intervals'] == NO_CHORD[-1]).all(axis=1))


def adjust(det_chords, ann_chords):
    # from numpy.lib.recfunctions import stack_arrays
    # TODO: fill at beginning!
    if det_chords[-1]['end'] < ann_chords[-1]['end']:
        filler = np.array(
            (det_chords[-1]['end'], ann_chords[-1]['end'], chord('N')),
            dtype=[('start', np.float), ('end', np.float),
                   ('chord', CHORD_DTYPE)])
        # filler['start'] = det_chords[-1]['end']
        # filler['end'] = ann_chords[-1]['end']
        # filler['chord'] = chord('N')
        det_chords = np.hstack([det_chords, filler])
    if det_chords[-1]['end'] > ann_chords[-1]['end']:
        # TODO: remove all detected chords that are outside the annotations
        # shorten last detected chord
        det_chords[-1]['end'] = ann_chords[-1]['end']

    # start = min(det_chords[0].start, ann_chords[0].start)
    # end = max(det_chords[-1].end, ann_chords[-1].end)
    #
    # filler.chord = chord('N')
    #
    # if det_chords[0].start > start:
    #     det_chords = np.vstack([
    #         [start, det_chords[0].start, chord('N')], det_chords
    #     ])
    # if ann_chords[0].start > start:
    #     ann_chords = np.vstack([
    #         [start, ann_chords[0].start, chord('N')], ann_chords
    #     ])
    # if det_chords[-1].end < end:
    #     filler.start = det_chords[-1].end
    #     filler.end = end
    #     det_chords = stack_arrays([det_chords, filler], usemask=False, asrecarray=True)
    # if ann_chords[-1].end < end:
    #     filler.start = ann_chords[-1].end
    #     filler.end = end
    #     ann_chords = stack_arrays([ann_chords, filler], usemask=False, asrecarray=True)
    #
    return det_chords, ann_chords


class ChordEvaluation(object):

    METRIC_NAMES = [
        ('root', 'Root'),
        ('majmin', 'MajMin'),
        ('majminbass', 'MajMinBass')
    ]

    def __init__(self, detections, annotations, name, **kwargs):
        det_chords = load_chords(detections)
        ann_chords = load_chords(annotations)
        det_chords, ann_chords = adjust(det_chords, ann_chords)
        self.eval_pairs = evaluation_pairs(det_chords, ann_chords)
        self.name = name

    @property
    def root(self):
        return np.average(score_root(self.eval_pairs),
                          weights=self.eval_pairs['duration'])

    @property
    def majmin(self):
        mapped_pairs = self.eval_pairs.copy()
        mapped_pairs['ref_chord'] = map_triads(self.eval_pairs['ref_chord'])
        mapped_pairs['est_chord'] = map_triads(self.eval_pairs['est_chord'])
        majmin_sel = select_majmin(mapped_pairs['ref_chord'])
        print (mapped_pairs['duration'] * majmin_sel).sum()
        return np.average(score_exact(mapped_pairs),
                          weights=mapped_pairs['duration'] * majmin_sel)

    @property
    def majminbass(self):
        mapped_pairs = self.eval_pairs.copy()
        mapped_pairs['ref_chord'] = map_triads(self.eval_pairs['ref_chord'],
                                               keep_bass=True)
        mapped_pairs['est_chord'] = map_triads(self.eval_pairs['est_chord'],
                                               keep_bass=True)
        majmin_sel = select_majmin(mapped_pairs['ref_chord'])
        return np.average(score_exact(mapped_pairs),
                          weights=mapped_pairs['duration'] * majmin_sel)

    def tostring(self, **kwargs):
        """
        Format the evaluation metrics as a human readable string.

        Returns
        -------
        str
            Evaluation metrics formatted as a human readable string

        """
        ret = ''
        if self.name is not None:
            ret += '%s\n  ' % self.name
        ret += 'Root: %.6f MajMin: %.6f MajMinBass: %.6f' % (
            self.root, self.majmin, self.majminbass)
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
    name : str
        Name to be displayed.
    """
    # pylint: disable=super-init-not-called

    def __init__(self, eval_objects, name=None):
        self.name = name or 'mean for %d files' % len(eval_objects)
        self.eval_pairs = np.hstack([
            e.eval_pairs for e in eval_objects
        ])


class ChordMeanEvaluation(ChordEvaluation):
    """
    Class for averaging chord evaluation scores, averaging piecewise (i.e.
    ignoring the lengths of the pieces). For a detailed description of the
    available metrics, refer to ChordEvaluation.

    Parameters
    ----------
    eval_objects : list
        Evaluation objects.
    name : str
        Name to be displayed.
    """
    # pylint: disable=super-init-not-called

    def __init__(self, eval_objects, name=None):
        self.name = name or 'piecewise mean for %d files' % len(eval_objects)
        self.eval_objects = eval_objects

    @property
    def root(self):
        return np.mean([e.root for e in self.eval_objects])

    @property
    def majmin(self):
        return np.mean([e.majmin for e in self.eval_objects])

    @property
    def majminbass(self):
        return np.mean([e.majminbass for e in self.eval_objects])


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
    # add beat evaluation sub-parser to the existing parser
    p = parser.add_parser(
        'chords', help='chord evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
    This program evaluates pairs of files containing the chord annotations and
    predictions. Suffixes can be given to filter them from the list of files.

    Each line represents a beat and must have the following format with values
    being separated by whitespace:
    `start_time end_time chord_label`

    Lines starting with # are treated as comments and are ignored.
    ''')
    # set defaults
    p.set_defaults(eval=ChordEvaluation, sum_eval=ChordSumEvaluation,
                   mean_eval=ChordMeanEvaluation)
    # file I/O
    evaluation_io(p, ann_suffix='.chords', det_suffix='.chords.txt')
    # return the sub-parser and evaluation argument group
    return p
