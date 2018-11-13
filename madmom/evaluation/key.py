# encoding: utf-8
"""
This module contains key evaluation functionality.

"""

from collections import Counter

from . import EvaluationMixin, evaluation_io
from ..io import load_key


_KEY_TO_SEMITONE = {'c': 0, 'c#': 1, 'db': 1, 'd': 2, 'd#': 3, 'eb': 3, 'e': 4,
                    'f': 5, 'f#': 6, 'gb': 6, 'g': 7, 'g#': 8, 'ab': 8, 'a': 9,
                    'a#': 10, 'bb': 10, 'b': 11, 'cb': 11}


def key_label_to_class(key_label):
    """
    Convert key label to key class number.

    The key label must follow the MIREX syntax defined at
    http://music-ir.org/mirex/wiki/2017:Audio_Key_Detection:
    `tonic mode`, where tonic is in {C, C#, Db, ... Cb} and mode in {'major',
    'maj', 'minor', 'min'}. The label will be converted into a class id based
    on the root pitch id (c .. 0, c# .. 1, ..., cb ... 11) plus 12 if in minor
    mode.

    Parameters
    ----------
    key_label : str
        Key label.

    Returns
    -------
    key_class : int
        Key class.

    Examples
    --------
    >>> from madmom.evaluation.key import key_label_to_class
    >>> key_label_to_class('D major')
    2

    >>> key_label_to_class('D minor')
    14

    """
    tonic, mode = key_label.split()
    if tonic.lower() not in _KEY_TO_SEMITONE.keys():
        raise ValueError('Unknown tonic: {}'.format(tonic))
    key_class = _KEY_TO_SEMITONE[tonic.lower()]
    if mode in ['minor', 'min']:
        key_class += 12
    elif mode in ['major', 'maj']:
        key_class += 0
    else:
        raise ValueError('Unknown mode: {}'.format(mode))
    return key_class


def error_type(det_key, ann_key, strict_fifth=False):
    """
    Compute the evaluation score and error category for a predicted key
    compared to the annotated key.

    Categories and evaluation scores follow the evaluation strategy used
    for MIREX (see http://music-ir.org/mirex/wiki/2017:Audio_Key_Detection).
    There are two evaluation modes for the 'fifth' category: by default,
    a detection falls into the 'fifth' category if it is the fifth of the
    annotation, or the annotation is the fifth of the detection.
    If `strict_fifth` is `True`, only the former case is considered. This is
    the mode used for MIREX.

    Parameters
    ----------
    det_key : int
        Detected key class.
    ann_key : int
        Annotated key class.
    strict_fifth: bool
        Use strict interpretation of the 'fifth' category, as in MIREX.

    Returns
    -------
    score, category : float, str
        Evaluation score and error category.

    """
    ann_root = ann_key % 12
    ann_mode = ann_key // 12
    det_root = det_key % 12
    det_mode = det_key // 12
    major, minor = 0, 1

    if det_root == ann_root and det_mode == ann_mode:
        return 1.0, 'correct'
    if det_mode == ann_mode and ((det_root - ann_root) % 12 == 7):
        return 0.5, 'fifth'
    if not strict_fifth and (det_mode == ann_mode and
                             ((det_root - ann_root) % 12 == 5)):
        return 0.5, 'fifth'
    if (ann_mode == major and det_mode != ann_mode and (
            (det_root - ann_root) % 12 == 9)):
        return 0.3, 'relative'
    if (ann_mode == minor and det_mode != ann_mode and (
            (det_root - ann_root) % 12 == 3)):
        return 0.3, 'relative'
    if det_mode != ann_mode and det_root == ann_root:
        return 0.2, 'parallel'
    else:
        return 0.0, 'other'


class KeyEvaluation(EvaluationMixin):
    """
    Provide the key evaluation score.

    Parameters
    ----------
    detection : str
        File containing detected key
    annotation : str
        File containing annotated key
    strict_fifth : bool, optional
        Use strict interpretation of the 'fifth' category, as in MIREX.
    name : str, optional
        Name of the evaluation object (e.g., the name of the song).

    """

    METRIC_NAMES = [
        ('score', 'Score'),
        ('error_category', 'Error Category')
    ]

    def __init__(self, detection, annotation, strict_fifth=False, name=None,
                 **kwargs):
        self.name = name or ''
        self.detection = key_label_to_class(detection)
        self.annotation = key_label_to_class(annotation)
        self.score, self.error_category = error_type(
            self.detection, self.annotation, strict_fifth
        )

    def tostring(self, **kwargs):
        """
        Format the evaluation as a human readable string.

        Returns
        -------
        str
            Evaluation score and category as a human readable string.

        """
        ret = '{}: '.format(self.name) if self.name else ''
        ret += '{:3.1f}, {}'.format(self.score, self.error_category)
        return ret


class KeyMeanEvaluation(EvaluationMixin):
    """
    Class for averaging key evaluations.

    Parameters
    ----------
    eval_objects : list
        Key evaluation objects.
    name : str, optional
        Name to be displayed.

    """

    METRIC_NAMES = [
        ('correct', 'Correct'),
        ('fifth', 'Fifth'),
        ('relative', 'Relative'),
        ('parallel', 'Parallel'),
        ('other', 'Other'),
        ('weighted', 'Weighted'),
    ]

    def __init__(self, eval_objects, name=None):
        self.name = name or 'mean for {:d} files'.format(len(eval_objects))

        n = len(eval_objects)
        c = Counter(e.error_category for e in eval_objects)

        self.correct = float(c['correct']) / n
        self.fifth = float(c['fifth']) / n
        self.relative = float(c['relative']) / n
        self.parallel = float(c['parallel']) / n
        self.other = float(c['other']) / n
        self.weighted = sum(e.score for e in eval_objects) / n

    def tostring(self, **kwargs):
        return ('{}\n  Weighted: {:.3f}  Correct: {:.3f}  Fifth: {:.3f}  '
                'Relative: {:.3f}  Parallel: {:.3f}  Other: {:.3f}'.format(
                    self.name, self.weighted, self.correct, self.fifth,
                    self.relative, self.parallel, self.other))


def add_parser(parser):
    """
    Add a key evaluation sub-parser to an existing parser.

    Parameters
    ----------
    parser : argparse parser instance
        Existing argparse parser object.

    Returns
    -------
    sub_parser : argparse sub-parser instance
        Key evaluation sub-parser.

    """
    import argparse
    # add key evaluation sub-parser to the existing parser
    p = parser.add_parser(
        'key', help='key evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''
    This program evaluates pairs of files containing global key annotations
    and predictions. Suffixes can be given to filter them from the list of
    files.

    Each file must contain only the global key and follow the syntax outlined
    in http://music-ir.org/mirex/wiki/2017:Audio_Key_Detection:
    `tonic mode`, where tonic is in {C, C#, Db, ... Cb} and mode in {'major',
    'maj', 'minor', 'min'}.

    To maintain compatibility with MIREX evaluation scores, use the
    --strict_fifth flag.
    ''')
    # set defaults
    p.set_defaults(eval=KeyEvaluation, mean_eval=KeyMeanEvaluation,
                   sum_eval=None, load_fn=load_key)
    # file I/O
    evaluation_io(p, ann_suffix='.key', det_suffix='.key.txt')
    p.add_argument('--strict_fifth', dest='strict_fifth', action='store_true',
                   help='Strict interpretation of the \"fifth\" category.')
    # return the sub-parser and evaluation argument group
    return p
