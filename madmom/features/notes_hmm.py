# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
"""
This module contains HMM state spaces, transition and observation models used
for note transcription.

Notes
-----
Please note that (almost) everything within this module is discretised to
integer values because of performance reasons.

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from madmom.ml.hmm import TransitionModel, ObservationModel


class ADSRStateSpace(object):
    """
    Map state numbers to actual states.

    State 0 refers to silence, the ADSR states (attack, decay, sustain,
    release) are numbered from 1 onwards.

    Parameters
    ----------
    attack_length : int, optional
        Length of the attack phase.
    decay_length : int, optional
        Length of the decay phase.
    release_length : int, optional
        Length of the release phase.

    Sustain phase has no specific minimum length, since self-transitions from
    this state are used to model the note length.

    """

    def __init__(self, attack_length=1, decay_length=1, release_length=1):

        # define note with states which must be transitioned
        self.silence = 0
        self.attack = 1
        self.decay = self.attack + attack_length
        self.sustain = self.decay + decay_length
        self.release = self.sustain + release_length

    @property
    def num_states(self):
        return self.release + 1


class ADSRTransitionModel(TransitionModel):
    """
    Transition model for note transcription with a HMM.

    Parameters
    ----------
    state_space : :class:`ADSRStateSpace` instance
        ADSRStateSpace which maps state numbers to states.
    onset_prob : float, optional
        Probability to enter/stay in the attack and decay phase. When entering
        this phase from a previously sounding note, this probability will be
        divided by the sum of `onset_prob`, `note_prob`, and `offset_prob`.
    note_prob : float, optional
        Probability to enter the sustain phase. Notes can stay in the sustain
        phase given by this probability divided by the sum of `onset_prob`,
        `note_prob`, and `offset_prob`.
    offset_prob : float, optional
        Probability to enter/stay in the release phase.
    end_prob : float, optional
        Probability to go back from release to silence.

    """

    def __init__(self, state_space, onset_prob=0.8, note_prob=0.8,
                 offset_prob=0.2, end_prob=1.):
        # save attributes
        self.state_space = state_space
        # states
        silence = state_space.silence
        attack = state_space.attack
        decay = state_space.decay
        sustain = state_space.sustain
        release = state_space.release
        # transitions = [(from_state, to_state, prob), ...]
        # onset phase & min_onset_length
        t = [(silence, silence, 1. - onset_prob),
             (silence, attack, onset_prob)]
        for s in range(attack, decay):
            t.append((s, silence, 1. - onset_prob))
            t.append((s, s + 1, onset_prob))
        # transition to note & min_note_duration
        for s in range(decay, sustain):
            t.append((s, silence, 1. - note_prob))
            t.append((s, s + 1, note_prob))
        # 3 possibilities to continue note
        prob_sum = onset_prob + note_prob + offset_prob
        # 1) sustain note (keep sounding)
        t.append((sustain, sustain, note_prob / prob_sum))
        # 2) new note
        t.append((sustain, attack, onset_prob / prob_sum))
        # 3) release note (end note)
        t.append((sustain, sustain + 1, offset_prob / prob_sum))
        # release phase
        for s in range(sustain + 1, release):
            t.append((s, sustain, offset_prob))
            t.append((s, s + 1, 1. - offset_prob))
        # after releasing a note, go back to silence or start new note
        t.append((release, silence, end_prob))
        t.append((release, release, 1. - end_prob))
        t = np.array(t)
        # make the transitions sparse
        t = self.make_sparse(t[:, 1].astype(np.int), t[:, 0].astype(np.int),
                             t[:, 2])
        # instantiate a TransitionModel
        super(ADSRTransitionModel, self).__init__(*t)


class ADSRObservationModel(ObservationModel):
    """
    Observation model for note transcription tracking with a HMM.

    Parameters
    ----------
    state_space : :class:`ADSRStateSpace` instance
        ADSRStateSpace instance.

    The observed probabilities for note onsets, sounding notes, and offsets are
    mapped to the states defined in the state space. The observation for
    'silence' is defined as 1 - p(onset), 'onset' as p(onset), 'decay' and
    'sustain' as p(note) and 'offset' as p(offset).

    """

    def __init__(self, state_space):
        # define observation pointers
        pointers = np.zeros(state_space.num_states, dtype=np.uint32)
        # map from densities to states
        pointers[state_space.silence:] = 0
        pointers[state_space.attack:] = 1
        pointers[state_space.decay:] = 2
        # Note: sustain uses the same observations as decay
        pointers[state_space.release:] = 3
        # instantiate a ObservationModel with the pointers
        super(ADSRObservationModel, self).__init__(pointers)

    def log_densities(self, observations):
        """
        Computes the log densities of the observations.

        Parameters
        ----------
        observations : tuple with two numpy arrays
            Observations (i.e. 3d activations of the CNN).

        Returns
        -------
        numpy array
            Log densities of the observations.

        """
        # observations: notes, onsets, offsets
        densities = np.ones((len(observations), 4), dtype=np.float)
        # silence (not onset)
        densities[:, 0] = 1. - observations[:, 1]
        # attack: onset
        densities[:, 1] = observations[:, 1]
        # decay + sustain: note
        densities[:, 2] = observations[:, 0]
        # release: offset
        densities[:, 3] = observations[:, 2]
        # return the log densities
        return np.log(densities)
