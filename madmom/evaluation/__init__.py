# encoding: utf-8
"""
Evaluation package.

All evaluation methods of this package can be used as scripts directly, if the
package is in $PYTHONPATH.

Example:

python -m madmom.evaluation.onsets /dir/to/be/evaluated

"""
import onsets
import beats
import helpers

from simple import Evaluation, SumEvaluation, MeanEvaluation
