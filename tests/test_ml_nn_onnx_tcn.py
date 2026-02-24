from __future__ import absolute_import, annotations, division, print_function

import importlib
import unittest
from types import ModuleType
from typing import Any, cast, final
from unittest import mock

import numpy as np

from tests.test_ml_nn_onnx_parity import assert_parity_close
from tests.test_ml_nn_onnx_primitives import _fake_onnx_module, _get_converter_module


def linear(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    if out is None or out is x:
        return x
    out[:] = x
    return out


def relu(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    res = np.maximum(x, 0)
    if out is not None:
        out[:] = res
        return out
    return res


@final
class ConvolutionalLayer(object):
    def __init__(self, weights, bias, stride=None, pad="valid", activation_fn=None):
        self.weights = np.asarray(weights, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.stride = stride
        self.pad = pad
        self.activation_fn = activation_fn


@final
class FeedForwardLayer(object):
    def __init__(self, weights, bias, activation_fn=None):
        self.weights = np.asarray(weights, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.activation_fn = activation_fn


@final
class TCNBlock(object):
    def __init__(
        self,
        dilated_conv,
        dilation_rate,
        activation_fn=None,
        skip_conv=None,
        residual_conv=None,
    ):
        self.dilated_conv = dilated_conv
        self.dilation_rate = dilation_rate
        self.activation_fn = activation_fn
        self.skip_conv = skip_conv
        self.residual_conv = residual_conv


@final
class TCNLayer(object):
    def __init__(self, tcn_blocks, activation_fn=None, skip_connections=False):
        self.tcn_blocks = tcn_blocks
        self.skip_connections = skip_connections
        self.activation_fn = activation_fn


@final
class LayerContainer(object):
    def __init__(self, layers):
        self.layers = list(layers)


def _to_int_tuple(value: np.ndarray) -> tuple[int, ...]:
    flattened = np.asarray(value, dtype=np.int64).reshape(-1)
    return tuple(int(item) for item in flattened)


def _evaluate_conv_with_padding_and_dilation(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    strides: tuple[int, int],
    pads: tuple[int, int, int, int],
    dilations: tuple[int, int],
) -> np.ndarray:
    n, c_in, height, width = x.shape
    c_out, c_weights, kernel_height, kernel_width = weights.shape
    if c_weights != c_in:
        raise AssertionError("conv channel mismatch")

    stride_h, stride_w = strides
    dilation_h, dilation_w = dilations
    pad_top, pad_left, pad_bottom, pad_right = pads
    effective_kernel_h = (kernel_height - 1) * dilation_h + 1
    effective_kernel_w = (kernel_width - 1) * dilation_w + 1

    padded = np.pad(
        x,
        ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
    )
    out_height = (padded.shape[2] - effective_kernel_h) // stride_h + 1
    out_width = (padded.shape[3] - effective_kernel_w) // stride_w + 1
    out = np.zeros((n, c_out, out_height, out_width), dtype=np.float32)

    for batch_idx in range(n):
        for out_channel in range(c_out):
            for out_row in range(out_height):
                start_row = out_row * stride_h
                for out_col in range(out_width):
                    start_col = out_col * stride_w
                    total = float(bias[out_channel])
                    for in_channel in range(c_in):
                        for kernel_row in range(kernel_height):
                            source_row = start_row + kernel_row * dilation_h
                            for kernel_col in range(kernel_width):
                                source_col = start_col + kernel_col * dilation_w
                                total += (
                                    padded[
                                        batch_idx, in_channel, source_row, source_col
                                    ]
                                    * weights[
                                        out_channel, in_channel, kernel_row, kernel_col
                                    ]
                                )
                    out[batch_idx, out_channel, out_row, out_col] = total
    return out


def _evaluate_fake_tcn_outputs(
    payload: dict[str, object], input_array: np.ndarray
) -> dict[str, np.ndarray]:
    graph = cast(dict[str, object], payload["graph"])
    initializers = cast(list[dict[str, object]], graph["initializer"])
    values: dict[str, np.ndarray] = {"input": np.asarray(input_array, dtype=np.float32)}

    for item in initializers:
        dtype = np.dtype(cast(str, item["dtype"]))
        values[cast(str, item["name"])] = np.asarray(item["array"], dtype=dtype)

    nodes = cast(list[dict[str, object]], graph["nodes"])
    for node in nodes:
        op_type = cast(str, node["op_type"])
        inputs = [values[name] for name in cast(list[str], node["inputs"])]
        attrs = cast(dict[str, object], node.get("attrs", {}))
        output_name = cast(list[str], node["outputs"])[0]

        if op_type == "Unsqueeze":
            axes = list(_to_int_tuple(inputs[1]))
            output = inputs[0]
            for axis in sorted(axes):
                normalized_axis = axis
                if normalized_axis < 0:
                    normalized_axis += output.ndim + 1
                output = np.expand_dims(output, axis=normalized_axis)
            values[output_name] = output
        elif op_type == "Squeeze":
            if len(inputs) == 1:
                values[output_name] = np.squeeze(inputs[0])
            else:
                axes = _to_int_tuple(inputs[1])
                values[output_name] = np.squeeze(inputs[0], axis=axes)
        elif op_type == "Transpose":
            perm = attrs.get("perm")
            if perm is None:
                values[output_name] = np.transpose(inputs[0])
            else:
                values[output_name] = np.transpose(
                    inputs[0], axes=tuple(cast(list[int], perm))
                )
        elif op_type == "Conv":
            stride_attr = attrs.get("strides", [1, 1])
            pad_attr = attrs.get("pads", [0, 0, 0, 0])
            dilation_attr = attrs.get("dilations", [1, 1])
            values[output_name] = _evaluate_conv_with_padding_and_dilation(
                np.asarray(inputs[0], dtype=np.float32),
                np.asarray(inputs[1], dtype=np.float32),
                np.asarray(inputs[2], dtype=np.float32).reshape(-1),
                strides=cast(tuple[int, int], tuple(cast(list[int], stride_attr))),
                pads=cast(tuple[int, int, int, int], tuple(cast(list[int], pad_attr))),
                dilations=cast(tuple[int, int], tuple(cast(list[int], dilation_attr))),
            )
        elif op_type == "MatMul":
            values[output_name] = np.matmul(inputs[0], inputs[1])
        elif op_type == "Concat":
            axis = int(cast(int, attrs["axis"]))
            values[output_name] = np.concatenate(inputs, axis=axis)
        elif op_type == "Relu":
            values[output_name] = np.maximum(inputs[0], 0)
        elif op_type == "Add":
            values[output_name] = inputs[0] + inputs[1]
        else:
            raise AssertionError("unsupported fake op in tcn test: %s" % op_type)

    outputs: dict[str, np.ndarray] = {}
    output_infos = cast(list[dict[str, object]], graph["outputs"])
    for output_info in output_infos:
        name = cast(str, output_info["name"])
        outputs[name] = values[name]
    return outputs


def _evaluate_tcn_block_reference(
    input_data: np.ndarray, block: TCNBlock
) -> tuple[np.ndarray, np.ndarray]:
    conv = cast(ConvolutionalLayer, block.dilated_conv)
    weights = np.asarray(conv.weights, dtype=np.float32)
    bias = np.asarray(conv.bias, dtype=np.float32).reshape(-1)
    rate = int(block.dilation_rate)

    in_data = np.asarray(input_data, dtype=np.float32)
    time_steps, freq_bins, in_channels = in_data.shape
    if freq_bins != 1:
        raise AssertionError("reference supports only freq_bins == 1")
    out_channels = weights.shape[1]
    out = np.zeros((time_steps, 1, out_channels), dtype=np.float32)

    for time_index in range(time_steps):
        for out_channel in range(out_channels):
            total = float(bias[out_channel])
            for in_channel in range(in_channels):
                for kernel_index in range(weights.shape[3]):
                    source_index = time_index - kernel_index * rate
                    if source_index < 0:
                        continue
                    total += (
                        in_data[source_index, 0, in_channel]
                        * weights[in_channel, out_channel, 0, kernel_index]
                    )
            out[time_index, 0, out_channel] = total

    if block.activation_fn is not None:
        out = np.maximum(out, 0)
    skip = out
    residual = in_data + skip
    return residual, skip


class TestTCNConversion(unittest.TestCase):
    def _patched_import(self):
        fake_onnx = _fake_onnx_module()
        original_import = importlib.import_module

        def side_effect(name: str, package: str | None = None) -> ModuleType | object:
            del package
            if name == "onnx":
                return fake_onnx
            return original_import(name)

        return mock.patch("importlib.import_module", side_effect=side_effect)

    def test_tcn_block_single_conv(self):
        conv = ConvolutionalLayer(
            weights=np.ones((1, 4, 1, 3)), bias=np.zeros(4), pad="valid"
        )
        block = TCNBlock(dilated_conv=conv, dilation_rate=2, activation_fn=relu)

        converter: Any = _get_converter_module()
        with self._patched_import():
            onnx_model: Any = converter._build_onnx_model_from_primitive_layers(
                [block], opset=18
            )
        self.assertIsNotNone(onnx_model)
        # Check that we have a Conv node with dilations
        graph = cast(dict[str, Any], onnx_model.payload["graph"])
        nodes = cast(list[dict[str, Any]], graph["nodes"])
        conv_nodes = [n for n in nodes if n["op_type"] == "Conv"]
        self.assertEqual(len(conv_nodes), 1)
        self.assertEqual(conv_nodes[0]["attrs"]["dilations"], [2, 1])  # dilations

    def test_tcn_layer(self):
        conv = ConvolutionalLayer(
            weights=np.ones((1, 4, 1, 3)), bias=np.zeros(4), pad="valid"
        )
        block = TCNBlock(dilated_conv=conv, dilation_rate=1, activation_fn=relu)
        layer = TCNLayer(
            tcn_blocks=[block, block], activation_fn=relu, skip_connections=True
        )

        converter: Any = _get_converter_module()
        with self._patched_import():
            onnx_model: Any = converter._build_onnx_model_from_primitive_layers(
                [layer], opset=18
            )
        self.assertIsNotNone(onnx_model)
        # 2 blocks * 1 conv = 2 Conv nodes
        graph = cast(dict[str, Any], onnx_model.payload["graph"])
        nodes = cast(list[dict[str, Any]], graph["nodes"])
        conv_nodes = [n for n in nodes if n["op_type"] == "Conv"]
        self.assertEqual(len(conv_nodes), 2)

    def test_tcn_block_multi_conv(self):
        conv1 = ConvolutionalLayer(
            weights=np.ones((1, 2, 1, 3)), bias=np.zeros(2), pad="valid"
        )
        conv2 = ConvolutionalLayer(
            weights=np.ones((1, 2, 1, 3)), bias=np.zeros(2), pad="valid"
        )
        block = TCNBlock(
            dilated_conv=[conv1, conv2], dilation_rate=[1, 2], activation_fn=relu
        )

        converter: Any = _get_converter_module()
        with self._patched_import():
            onnx_model: Any = converter._build_onnx_model_from_primitive_layers(
                [block], opset=18
            )
        self.assertIsNotNone(onnx_model)
        # 2 convs = 2 Conv nodes + 1 Concat node
        graph = cast(dict[str, Any], onnx_model.payload["graph"])
        nodes = cast(list[dict[str, Any]], graph["nodes"])
        conv_nodes = [n for n in nodes if n["op_type"] == "Conv"]
        self.assertEqual(len(conv_nodes), 2)
        concat_nodes = [n for n in nodes if n["op_type"] == "Concat"]
        self.assertTrue(len(concat_nodes) >= 1)

    def test_tcn_block_residual_path(self):
        conv = ConvolutionalLayer(
            weights=np.ones((1, 4, 1, 3)), bias=np.zeros(4), pad="valid"
        )
        res_conv = ConvolutionalLayer(
            weights=np.ones((1, 4, 1, 1)), bias=np.zeros(4), pad="valid"
        )
        block = TCNBlock(
            dilated_conv=conv,
            dilation_rate=1,
            activation_fn=relu,
            residual_conv=res_conv,
        )

        converter: Any = _get_converter_module()
        with self._patched_import():
            onnx_model: Any = converter._build_onnx_model_from_primitive_layers(
                [block], opset=18
            )
        self.assertIsNotNone(onnx_model)
        graph = cast(dict[str, Any], onnx_model.payload["graph"])
        nodes = cast(list[dict[str, Any]], graph["nodes"])
        # Should have 2 Conv nodes (one for dilated_conv, one for residual_conv)
        # plus one for skip_conv (none here)
        # and one Add node for residual summation
        conv_nodes = [n for n in nodes if n["op_type"] == "Conv"]
        self.assertEqual(len(conv_nodes), 2)
        add_nodes = [n for n in nodes if n["op_type"] == "Add"]
        self.assertTrue(len(add_nodes) >= 1)

    def test_tcn_block_skip_conv(self):
        conv = ConvolutionalLayer(
            weights=np.ones((1, 4, 1, 3)), bias=np.zeros(4), pad="valid"
        )
        skip_conv = ConvolutionalLayer(
            weights=np.ones((4, 8, 1, 1)), bias=np.zeros(8), pad="valid"
        )
        block = TCNBlock(
            dilated_conv=conv, dilation_rate=1, activation_fn=relu, skip_conv=skip_conv
        )

        converter: Any = _get_converter_module()
        with self._patched_import():
            onnx_model: Any = converter._build_onnx_model_from_primitive_layers(
                [block], opset=18
            )
        self.assertIsNotNone(onnx_model)
        graph = cast(dict[str, Any], onnx_model.payload["graph"])
        nodes = cast(list[dict[str, Any]], graph["nodes"])
        # 1 dilated_conv + 1 skip_conv = 2 Conv nodes
        conv_nodes = [n for n in nodes if n["op_type"] == "Conv"]
        self.assertEqual(len(conv_nodes), 2)

    def test_tcn_block_parity_residual_and_skip_outputs(self):
        conv = ConvolutionalLayer(
            weights=np.array(
                [
                    [[[0.2, -0.1, 0.4], [0.0, 0.0, 0.0]]],
                    [[[0.1, 0.3, -0.2], [0.0, 0.0, 0.0]]],
                ],
                dtype=np.float32,
            ).transpose(0, 2, 1, 3),
            bias=np.array([0.05, -0.02], dtype=np.float32),
            pad="valid",
        )
        block = TCNBlock(dilated_conv=conv, dilation_rate=2, activation_fn=relu)
        model = LayerContainer([block])
        input_data = np.array(
            [
                [[0.2, -0.1]],
                [[0.0, 0.3]],
                [[-0.4, 0.5]],
                [[0.7, -0.2]],
                [[0.1, 0.4]],
                [[-0.3, 0.6]],
            ],
            dtype=np.float32,
        )
        expected_residual, expected_skip = _evaluate_tcn_block_reference(
            input_data, block
        )

        converter: Any = _get_converter_module()
        with self._patched_import():
            onnx_model: Any = converter._build_onnx_model_from_primitive_layers(
                model.layers, opset=18
            )

        payload = cast(dict[str, object], onnx_model.payload)
        graph = cast(dict[str, object], payload["graph"])
        output_infos = cast(list[dict[str, object]], graph["outputs"])
        observed_outputs = _evaluate_fake_tcn_outputs(payload, input_data)
        observed_ordered = [
            observed_outputs[cast(str, output_info["name"])]
            for output_info in output_infos
        ]
        self.assertEqual(len(observed_ordered), 2)

        assert_parity_close(
            observed_ordered[0],
            expected_residual.squeeze(),
            family="tcn",
            layer_group="default",
            context="output=residual",
        )
        assert_parity_close(
            observed_ordered[1],
            expected_skip.squeeze(),
            family="tcn",
            layer_group="default",
            context="output=skip",
        )

    def test_tcn_block_invalid_weights_rank(self):
        # Weight rank 3 instead of 4
        conv = ConvolutionalLayer(
            weights=np.ones((1, 4, 3)), bias=np.zeros(4), pad="valid"
        )
        block = TCNBlock(dilated_conv=conv, dilation_rate=1, activation_fn=relu)

        converter: Any = _get_converter_module()
        with self._patched_import():
            with self.assertRaisesRegex(Exception, "weights must be rank-4"):
                converter._build_onnx_model_from_primitive_layers([block], opset=18)

    def test_tcn_block_feedforward_skip_and_residual_paths(self):
        conv = ConvolutionalLayer(
            weights=np.array(
                [
                    [
                        [[0.2, -0.1, 0.05]],
                        [[-0.05, 0.12, 0.08]],
                    ],
                    [
                        [[-0.3, 0.25, 0.1]],
                        [[0.2, -0.15, 0.07]],
                    ],
                ],
                dtype=np.float32,
            ),
            bias=np.array([0.02, -0.03], dtype=np.float32),
            pad="valid",
        )
        skip_ff = FeedForwardLayer(
            weights=np.array([[0.4, -0.2], [0.1, 0.3]], dtype=np.float32),
            bias=np.array([0.05, -0.04], dtype=np.float32),
            activation_fn=linear,
        )
        residual_ff = FeedForwardLayer(
            weights=np.array([[0.3, 0.2], [-0.1, 0.5]], dtype=np.float32),
            bias=np.array([0.01, -0.02], dtype=np.float32),
            activation_fn=linear,
        )
        block = TCNBlock(
            dilated_conv=conv,
            dilation_rate=1,
            activation_fn=relu,
            skip_conv=skip_ff,
            residual_conv=residual_ff,
        )
        model = LayerContainer([block])
        input_data = np.array(
            [
                [[0.2, -0.1]],
                [[0.0, 0.3]],
                [[-0.4, 0.5]],
                [[0.7, -0.2]],
            ],
            dtype=np.float32,
        )

        _, skip_source = _evaluate_tcn_block_reference(input_data, block)
        expected_skip = np.matmul(skip_source, skip_ff.weights) + skip_ff.bias
        expected_residual_path = (
            np.matmul(input_data, residual_ff.weights) + residual_ff.bias
        )
        expected_residual = expected_residual_path + expected_skip

        converter: Any = _get_converter_module()
        with self._patched_import():
            onnx_model: Any = converter._build_onnx_model_from_primitive_layers(
                model.layers, opset=18
            )

        payload = cast(dict[str, object], onnx_model.payload)
        graph = cast(dict[str, object], payload["graph"])
        nodes = cast(list[dict[str, object]], graph["nodes"])
        conv_nodes = [n for n in nodes if n["op_type"] == "Conv"]
        matmul_nodes = [n for n in nodes if n["op_type"] == "MatMul"]
        self.assertEqual(len(conv_nodes), 1)
        self.assertEqual(len(matmul_nodes), 2)

        output_infos = cast(list[dict[str, object]], graph["outputs"])
        observed_outputs = _evaluate_fake_tcn_outputs(payload, input_data)
        observed_ordered = [
            observed_outputs[cast(str, output_info["name"])]
            for output_info in output_infos
        ]
        self.assertEqual(len(observed_ordered), 2)

        assert_parity_close(
            observed_ordered[0],
            expected_residual.squeeze(),
            family="tcn",
            layer_group="default",
            context="output=residual feedforward-branches",
        )
        assert_parity_close(
            observed_ordered[1],
            expected_skip.squeeze(),
            family="tcn",
            layer_group="default",
            context="output=skip feedforward-branches",
        )
