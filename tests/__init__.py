# encoding: utf-8
# pylint: skip-file
"""
This module contains tests.

"""

from __future__ import absolute_import, division, print_function

import os

from madmom import MODELS_PATH

DATA_PATH = os.path.dirname(os.path.realpath(__file__)) + '/data/'
AUDIO_PATH = DATA_PATH + 'audio/'
ACTIVATIONS_PATH = DATA_PATH + 'activations/'
ANNOTATIONS_PATH = DATA_PATH + 'annotations/'
DETECTIONS_PATH = DATA_PATH + 'detections/'
MODELS_PATH = DATA_PATH + 'models/'
