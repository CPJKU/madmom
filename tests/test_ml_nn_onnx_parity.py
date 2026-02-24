from __future__ import absolute_import, division, print_function

import os
import unittest

import numpy as np


INTENTIONAL_PARITY_BREAK_ENV = "MADMOM_ONNX_PARITY_BREAK_HARNESS"

PARITY_TOLERANCE_POLICY = {
    "primitives": {
        "default": {"atol": 1e-6, "rtol": 1e-6},
    },
    "conv": {
        "default": {"atol": 1e-6, "rtol": 1e-6},
    },
    "structure": {
        "default": {"atol": 1e-6, "rtol": 1e-6},
    },
    "recurrent": {
        "default": {"atol": 1e-6, "rtol": 1e-6},
        "online": {"atol": 1e-6, "rtol": 1e-6},
        "state": {"atol": 1e-6, "rtol": 1e-6},
    },
    "tcn": {
        "default": {"atol": 1e-6, "rtol": 1e-6},
    },
    "wrapped": {
        "default": {"atol": 1e-6, "rtol": 1e-6},
    },
    "runtime": {
        "default": {"atol": 1e-6, "rtol": 1e-6},
        "online": {"atol": 1e-6, "rtol": 1e-6},
    },
}


def parity_tolerance(family, layer_group="default"):
    if family not in PARITY_TOLERANCE_POLICY:
        raise KeyError("unknown ONNX parity family: %s" % family)
    family_policy = PARITY_TOLERANCE_POLICY[family]
    if layer_group in family_policy:
        return family_policy[layer_group]
    return family_policy["default"]


def assert_parity_close(
    observed,
    expected,
    *,
    family,
    layer_group="default",
    context="",
):
    tolerance = parity_tolerance(family, layer_group)
    atol = float(tolerance["atol"])
    rtol = float(tolerance["rtol"])

    observed_array = np.asarray(observed, dtype=np.float32)
    expected_array = np.asarray(expected, dtype=np.float32)

    if observed_array.shape != expected_array.shape:
        raise AssertionError(
            "ONNX parity shape mismatch family=%s layer_group=%s context=%s "
            "expected_shape=%s observed_shape=%s"
            % (
                family,
                layer_group,
                context,
                expected_array.shape,
                observed_array.shape,
            )
        )

    if np.allclose(observed_array, expected_array, atol=atol, rtol=rtol):
        return

    abs_diff = np.abs(observed_array - expected_array)
    if abs_diff.size == 0:
        max_index = ()
        max_abs_diff = 0.0
        max_rel_diff = 0.0
        observed_value = float(observed_array.reshape(-1)[0])
        expected_value = float(expected_array.reshape(-1)[0])
    else:
        flat_index = int(np.argmax(abs_diff))
        max_index = np.unravel_index(flat_index, abs_diff.shape)
        max_abs_diff = float(abs_diff[max_index])
        denominator = max(abs(float(expected_array[max_index])), 1e-12)
        max_rel_diff = max_abs_diff / denominator
        observed_value = float(observed_array[max_index])
        expected_value = float(expected_array[max_index])

    raise AssertionError(
        "ONNX parity mismatch family=%s layer_group=%s context=%s "
        "atol=%g rtol=%g max_abs_diff=%.10g max_rel_diff=%.10g "
        "max_index=%s observed=%r expected=%r"
        % (
            family,
            layer_group,
            context,
            atol,
            rtol,
            max_abs_diff,
            max_rel_diff,
            max_index,
            observed_value,
            expected_value,
        )
    )


class TestMlNnOnnxParity(unittest.TestCase):
    def test_parity_tolerance_policy_covers_supported_families(self):
        expected_families = {
            "primitives",
            "conv",
            "structure",
            "recurrent",
            "tcn",
            "wrapped",
            "runtime",
        }
        self.assertEqual(set(PARITY_TOLERANCE_POLICY), expected_families)

    def test_parity_failure_diagnostics_include_family_layer_and_context(self):
        with self.assertRaises(AssertionError) as ctx:
            assert_parity_close(
                observed=np.asarray([0.1, 0.2], dtype=np.float32),
                expected=np.asarray([0.1, 1.2], dtype=np.float32),
                family="recurrent",
                layer_group="online",
                context="layer=lstm chunk=tail",
            )

        message = str(ctx.exception)
        self.assertIn("family=recurrent", message)
        self.assertIn("layer_group=online", message)
        self.assertIn("layer=lstm", message)
        self.assertIn("chunk=tail", message)

    @unittest.skipUnless(
        os.getenv(INTENTIONAL_PARITY_BREAK_ENV) == "1",
        "intentional ONNX parity break harness is disabled",
    )
    def test_intentional_parity_break_harness(self):
        assert_parity_close(
            observed=np.asarray([0.0, 0.0], dtype=np.float32),
            expected=np.asarray([0.0, 0.5], dtype=np.float32),
            family="primitives",
            layer_group="default",
            context="intentional-break-harness",
        )


if __name__ == "__main__":
    unittest.main()
