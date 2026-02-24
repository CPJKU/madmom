from __future__ import absolute_import, annotations, division, print_function

import hashlib
import importlib
import importlib.util
import json
import os
import pickle
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from typing import (
    Callable,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
    cast,
    final,
)

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias
from unittest import mock

import numpy as np

from tests.test_ml_nn_onnx_parity import assert_parity_close


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
SCRIPT_PATH = os.path.join(REPO_ROOT, "tools", "convert_models_to_onnx.py")

ActivationFn: TypeAlias = Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]
AxisType: TypeAlias = Union[int, Tuple[int, ...], None]
AverageDType: TypeAlias = Union[Type[np.float32], np.dtype[np.float32], None]


def linear(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    if out is None or x is out:
        return x
    out[:] = x
    return out


def tanh(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    return np.tanh(x, out)


def sigmoid(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    if out is None:
        out = np.asarray(0.5 * x)
    else:
        if out is not x:
            out[:] = x
        out *= 0.5
    _ = np.tanh(out, out=out)
    out += 1.0
    out *= 0.5
    return out


def relu(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    return np.maximum(x, 0, out=out)


def elu(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    if out is None:
        out = x.copy()
    elif out is not x:
        out[:] = x[:]
    mask = x < 0
    out[mask] = np.exp(x[mask]) - 1
    return out


def softmax(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    shifted = cast(np.ndarray, x - np.amax(x, axis=1, keepdims=True))
    exponentiated = np.exp(shifted)
    denominator = cast(np.ndarray, np.sum(exponentiated, axis=1, keepdims=True))
    probabilities = cast(np.ndarray, exponentiated / denominator)
    if out is None:
        return probabilities
    out[:] = probabilities
    return out


def unsupported_activation(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    if out is None:
        return x
    out[:] = x
    return out


for _activation in (linear, tanh, sigmoid, relu, elu, softmax, unsupported_activation):
    _activation.__module__ = __name__


@final
class FeedForwardLayer(object):
    weights: np.ndarray
    bias: np.ndarray
    activation_fn: ActivationFn | None

    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray,
        activation_fn: ActivationFn | None = None,
    ):
        self.weights = np.asarray(weights, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32).flatten()
        self.activation_fn = activation_fn

    def __call__(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        out = cast(np.ndarray, np.matmul(data, self.weights) + self.bias)
        if self.activation_fn is not None:
            _ = self.activation_fn(out, out)
        return out


@final
class BatchNormLayer(object):
    beta: np.ndarray
    gamma: np.ndarray
    mean: np.ndarray
    inv_std: np.ndarray
    activation_fn: ActivationFn | None

    def __init__(
        self,
        beta: np.ndarray,
        gamma: np.ndarray,
        mean: np.ndarray,
        inv_std: np.ndarray,
        activation_fn: ActivationFn | None = None,
    ):
        self.beta = np.asarray(beta, dtype=np.float32)
        self.gamma = np.asarray(gamma, dtype=np.float32)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.inv_std = np.asarray(inv_std, dtype=np.float32)
        self.activation_fn = activation_fn

    def __call__(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        out = (data - self.mean) * (self.gamma * self.inv_std) + self.beta
        if self.activation_fn is not None:
            _ = self.activation_fn(out, out)
        return out


@final
class ReshapeLayer(object):
    newshape: tuple[int, ...]
    order: Literal["C", "F", "A"]

    def __init__(self, newshape: tuple[int, ...], order: Literal["C", "F", "A"] = "C"):
        self.newshape = newshape
        self.order = order

    def __call__(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        return np.reshape(data, self.newshape, order=self.order)


@final
class TransposeLayer(object):
    axes: tuple[int, ...] | None

    def __init__(self, axes: tuple[int, ...] | None = None):
        self.axes = axes

    def __call__(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        return np.transpose(data, self.axes)


@final
class PadLayer(object):
    width: int
    axes: tuple[int, ...]
    value: float

    def __init__(self, width: int, axes: tuple[int, ...], value: float = 0.0):
        self.width = width
        self.axes = axes
        self.value = value

    def __call__(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        shape = list(data.shape)
        data_idxs = [slice(None) for _ in range(len(shape))]
        for axis in self.axes:
            shape[axis] += self.width * 2
            data_idxs[axis] = slice(self.width, -self.width)
        padded = np.full(tuple(shape), self.value, dtype=np.float32)
        padded[tuple(data_idxs)] = data
        return padded


@final
class AverageLayer(object):
    axis: AxisType
    dtype: AverageDType
    keepdims: bool

    def __init__(
        self,
        axis: AxisType = None,
        dtype: AverageDType = None,
        keepdims: bool = False,
    ):
        self.axis = axis
        self.dtype = dtype
        self.keepdims = keepdims

    def __call__(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        return cast(
            np.ndarray,
            np.mean(data, axis=self.axis, dtype=self.dtype, keepdims=self.keepdims),
        )


@final
class PrimitiveNetwork(object):
    layers: list[object]

    def __init__(self, layers: list[object]):
        self.layers = layers

    def process(self, data: np.ndarray) -> np.ndarray:
        if data.ndim < 2:
            data = np.array(data, subok=True, copy=False, ndmin=2)
        output = data
        for layer in self.layers:
            output = cast(Callable[[np.ndarray], np.ndarray], layer)(output)
        return np.asarray(output).squeeze()


@final
class UnsupportedLayer(object):
    def __call__(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        return data


class ConverterModule(Protocol):
    DEFAULT_OPSET: int

    def convert_manifest_entries(
        self,
        manifest: dict[str, object],
        models_dir: str,
        converter: object = None,
        run_onnx_checker: bool = True,
    ) -> dict[str, object]: ...


converter_module: ConverterModule | None = None


def _load_converter_module() -> ConverterModule:
    spec = importlib.util.spec_from_file_location("convert_models_to_onnx", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load converter module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module_object = cast(object, module)
    return cast(ConverterModule, module_object)


def _get_converter_module() -> ConverterModule:
    global converter_module
    if converter_module is None:
        converter_module = _load_converter_module()
    return converter_module


def _sha256_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as infile:
        while True:
            chunk = infile.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:%s" % digest.hexdigest()


def _single_convertible_manifest(
    module: ConverterModule,
    source_rel: str,
    target_rel: str,
    source_path: str,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "source_pkl": source_rel,
        "target_onnx": target_rel,
        "hash": _sha256_digest(source_path),
        "opset": module.DEFAULT_OPSET,
        "status": "nn_convertible",
        "convertible": True,
        "conversion_mode": "direct_nn",
    }
    return {"schema_version": 1, "artifacts": [artifact]}


@final
class _FakeTensorProto(object):
    FLOAT: int = 1
    INT64: int = 7


@final
class _FakeNumpyHelper(object):
    @staticmethod
    def from_array(array: np.ndarray, name: str | None = None) -> dict[str, object]:
        value = np.asarray(array)
        return {
            "name": name,
            "array": value.tolist(),
            "dtype": str(value.dtype),
        }


@final
class _FakeHelper(object):
    @staticmethod
    def make_tensor_value_info(
        name: str, elem_type: int, shape: object
    ) -> dict[str, object]:
        return {"name": name, "elem_type": elem_type, "shape": shape}

    @staticmethod
    def make_node(
        op_type: str,
        inputs: list[str],
        outputs: list[str],
        name: str | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        return {
            "op_type": op_type,
            "inputs": list(inputs),
            "outputs": list(outputs),
            "name": name,
            "attrs": kwargs,
        }

    @staticmethod
    def make_graph(
        nodes: list[dict[str, object]],
        name: str,
        inputs: list[dict[str, object]],
        outputs: list[dict[str, object]],
        initializer: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return {
            "name": name,
            "nodes": list(nodes),
            "inputs": list(inputs),
            "outputs": list(outputs),
            "initializer": list(initializer or []),
        }

    @staticmethod
    def make_model(
        graph: dict[str, object],
        producer_name: str,
        opset_imports: list[dict[str, object]],
    ) -> object:
        return _FakeModel(
            {
                "graph": graph,
                "producer_name": producer_name,
                "opset_imports": opset_imports,
            }
        )

    @staticmethod
    def make_opsetid(domain: str, version: int) -> dict[str, object]:
        return {"domain": domain, "version": version}


@final
class _FakeModel(object):
    payload: dict[str, object]

    def __init__(self, payload: dict[str, object]):
        self.payload = payload

    def SerializeToString(self, deterministic: bool = False) -> bytes:
        return json.dumps(
            self.payload,
            sort_keys=deterministic,
            separators=(",", ":"),
        ).encode("utf-8")


@final
class _FakeChecker(object):
    @staticmethod
    def check_model(model_path: str) -> None:
        del model_path


def _fake_onnx_module() -> object:
    return SimpleNamespace(
        TensorProto=_FakeTensorProto,
        helper=_FakeHelper(),
        numpy_helper=_FakeNumpyHelper(),
        checker=_FakeChecker(),
    )


def _model_payload(model_bytes: bytes) -> dict[str, object]:
    return cast(dict[str, object], json.loads(model_bytes.decode("utf-8")))


def _to_int_tuple(value: np.ndarray) -> tuple[int, ...]:
    flattened = np.asarray(value, dtype=np.int64).reshape(-1)
    return tuple(int(item) for item in flattened)


def _evaluate_fake_onnx(
    payload: dict[str, object], input_array: np.ndarray
) -> np.ndarray:
    graph = cast(dict[str, object], payload["graph"])
    initializer_items = cast(list[dict[str, object]], graph["initializer"])
    values: dict[str, np.ndarray] = {"input": np.asarray(input_array, dtype=np.float32)}

    for item in initializer_items:
        dtype = np.dtype(cast(str, item["dtype"]))
        values[cast(str, item["name"])] = np.asarray(item["array"], dtype=dtype)

    nodes = cast(list[dict[str, object]], graph["nodes"])
    for node in nodes:
        op_type = cast(str, node["op_type"])
        inputs = [values[name] for name in cast(list[str], node["inputs"])]
        attrs = cast(dict[str, object], node.get("attrs", {}))
        output_name = cast(list[str], node["outputs"])[0]

        if op_type == "MatMul":
            values[output_name] = np.matmul(inputs[0], inputs[1])
        elif op_type == "Add":
            values[output_name] = inputs[0] + inputs[1]
        elif op_type == "Mul":
            values[output_name] = inputs[0] * inputs[1]
        elif op_type == "Tanh":
            values[output_name] = np.tanh(inputs[0])
        elif op_type == "Sigmoid":
            values[output_name] = 1.0 / (1.0 + np.exp(-inputs[0]))
        elif op_type == "Relu":
            values[output_name] = np.maximum(inputs[0], 0)
        elif op_type == "Elu":
            alpha = float(cast(float, attrs.get("alpha", 1.0)))
            values[output_name] = np.where(
                inputs[0] > 0,
                inputs[0],
                alpha * (np.exp(inputs[0]) - 1.0),
            )
        elif op_type == "Softmax":
            axis = int(cast(int, attrs.get("axis", 1)))
            shifted = cast(
                np.ndarray,
                inputs[0] - np.max(inputs[0], axis=axis, keepdims=True),
            )
            exponentiated = np.exp(shifted)
            denominator = cast(
                np.ndarray, np.sum(exponentiated, axis=axis, keepdims=True)
            )
            values[output_name] = cast(np.ndarray, exponentiated / denominator)
        elif op_type == "Reshape":
            target_shape = _to_int_tuple(inputs[1])
            values[output_name] = np.reshape(inputs[0], target_shape)
        elif op_type == "Transpose":
            perm = attrs.get("perm")
            if perm is None:
                values[output_name] = np.transpose(inputs[0])
            else:
                values[output_name] = np.transpose(
                    inputs[0], axes=tuple(cast(list[int], perm))
                )
        elif op_type == "Pad":
            pads = list(_to_int_tuple(inputs[1]))
            rank = len(inputs[0].shape)
            begins = pads[:rank]
            ends = pads[rank:]
            pad_width = list(zip(begins, ends))
            constant_array = np.asarray(inputs[2], dtype=np.float32).reshape(())
            constant_value = float(constant_array.item())
            values[output_name] = np.pad(
                inputs[0],
                pad_width=pad_width,
                mode="constant",
                constant_values=constant_value,
            )
        elif op_type == "ReduceMean":
            keepdims = bool(int(cast(int, attrs.get("keepdims", 1))))
            if len(inputs) == 1:
                values[output_name] = np.mean(inputs[0], keepdims=keepdims)
            else:
                axes_values = _to_int_tuple(inputs[1])
                values[output_name] = np.mean(
                    inputs[0],
                    axis=axes_values,
                    keepdims=keepdims,
                )
        elif op_type == "Cast":
            values[output_name] = inputs[0].astype(np.float32)
        elif op_type == "Squeeze":
            if len(inputs) == 1:
                values[output_name] = np.squeeze(inputs[0])
            else:
                axes_values = _to_int_tuple(inputs[1])
                values[output_name] = np.squeeze(inputs[0], axis=axes_values)
        else:
            raise AssertionError("unsupported fake op: %s" % op_type)

    output_name = cast(list[dict[str, object]], graph["outputs"])[0]["name"]
    return values[cast(str, output_name)]


class TestMlNnOnnxPrimitives(unittest.TestCase):
    def _patched_import(self):
        fake_onnx = _fake_onnx_module()
        original_import = importlib.import_module

        def side_effect(name: str, package: str | None = None) -> ModuleType | object:
            del package
            if name == "onnx":
                return fake_onnx
            return original_import(name)

        return mock.patch("importlib.import_module", side_effect=side_effect)

    def test_primitive_fixture_converts_deterministically_and_matches_reference(self):
        module = _get_converter_module()

        model = PrimitiveNetwork(
            layers=[
                FeedForwardLayer(
                    weights=np.array(
                        [
                            [0.1, -0.4, 0.8, 1.0],
                            [0.2, 0.5, -0.2, 0.1],
                            [-0.6, 0.7, 0.3, -0.9],
                        ],
                        dtype=np.float32,
                    ),
                    bias=np.array([0.4, -0.2, 0.3, 0.1], dtype=np.float32),
                    activation_fn=relu,
                ),
                BatchNormLayer(
                    beta=np.array([0.01, -0.02, 0.03, 0.04], dtype=np.float32),
                    gamma=np.array([1.1, 0.9, 1.2, 0.7], dtype=np.float32),
                    mean=np.array([0.3, -0.1, 0.5, 0.2], dtype=np.float32),
                    inv_std=np.array([0.8, 1.1, 0.6, 0.9], dtype=np.float32),
                    activation_fn=tanh,
                ),
                ReshapeLayer(newshape=(2, 2, 2), order="C"),
                TransposeLayer(axes=(1, 0, 2)),
                PadLayer(width=1, axes=(2,), value=0.25),
                AverageLayer(axis=2, keepdims=False),
                FeedForwardLayer(
                    weights=np.array(
                        [[1.2, -0.4, 0.5], [0.3, 0.7, -1.1]],
                        dtype=np.float32,
                    ),
                    bias=np.array([0.2, -0.3, 0.1], dtype=np.float32),
                    activation_fn=softmax,
                ),
            ]
        )
        input_data = np.array(
            [[1.0, -0.5, 0.3], [0.2, 0.4, -1.0]],
            dtype=np.float32,
        )
        expected = model.process(input_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            source_rel = "fixtures/primitives.pkl"
            target_rel = "fixtures/primitives.onnx"
            source_path = os.path.join(temp_dir, "fixtures", "primitives.pkl")
            target_path = os.path.join(temp_dir, "fixtures", "primitives.onnx")
            os.makedirs(os.path.dirname(source_path), exist_ok=True)
            with open(source_path, "wb") as outfile:
                pickle.dump(model, outfile)

            manifest = _single_convertible_manifest(
                module=module,
                source_rel=source_rel,
                target_rel=target_rel,
                source_path=source_path,
            )

            with self._patched_import():
                first = module.convert_manifest_entries(
                    manifest=manifest,
                    models_dir=temp_dir,
                    run_onnx_checker=False,
                )
                second = module.convert_manifest_entries(
                    manifest=manifest,
                    models_dir=temp_dir,
                    run_onnx_checker=False,
                )

            first_result = cast(list[dict[str, object]], first["results"])[0]
            second_result = cast(list[dict[str, object]], second["results"])[0]
            self.assertEqual(first_result["status"], "converted")
            self.assertEqual(second_result["status"], "converted")
            self.assertEqual(first_result["output_hash"], second_result["output_hash"])

            with open(target_path, "rb") as infile:
                first_bytes = infile.read()
            with open(target_path, "rb") as infile:
                second_bytes = infile.read()
            self.assertEqual(first_bytes, second_bytes)

            model_payload = _model_payload(first_bytes)
            graph_payload = cast(dict[str, object], model_payload["graph"])
            graph_inputs = cast(list[dict[str, object]], graph_payload["inputs"])
            graph_outputs = cast(list[dict[str, object]], graph_payload["outputs"])
            self.assertEqual(
                graph_inputs[0]["shape"],
                ["input_dim_0", "input_dim_1"],
            )
            self.assertEqual(graph_outputs[0]["shape"], ["output_0_dim_0"])

            observed = _evaluate_fake_onnx(model_payload, input_data)
            assert_parity_close(
                observed,
                expected,
                family="primitives",
                layer_group="default",
                context="fixture=primitives",
            )

            op_types = [
                cast(str, node["op_type"])
                for node in cast(
                    list[dict[str, object]],
                    cast(dict[str, object], model_payload["graph"])["nodes"],
                )
            ]
            self.assertIn("Reshape", op_types)
            self.assertIn("Transpose", op_types)
            self.assertIn("Pad", op_types)
            self.assertIn("ReduceMean", op_types)
            self.assertIn("Softmax", op_types)
            self.assertIn("Squeeze", op_types)

    def test_supported_activation_mappings_emit_expected_ops(self):
        module = _get_converter_module()
        input_data = np.array(
            [[0.2, -0.1], [1.0, 0.5]],
            dtype=np.float32,
        )
        cases: list[tuple[str, ActivationFn, str | None]] = [
            ("linear", linear, None),
            ("tanh", tanh, "Tanh"),
            ("sigmoid", sigmoid, "Sigmoid"),
            ("relu", relu, "Relu"),
            ("elu", elu, "Elu"),
            ("softmax", softmax, "Softmax"),
        ]

        for activation_name, activation_fn, expected_op in cases:
            with self.subTest(activation=activation_name):
                model = PrimitiveNetwork(
                    layers=[
                        FeedForwardLayer(
                            weights=np.array(
                                [[1.0, -0.5], [0.3, 0.7]],
                                dtype=np.float32,
                            ),
                            bias=np.array([0.1, -0.2], dtype=np.float32),
                            activation_fn=activation_fn,
                        )
                    ]
                )
                expected = model.process(input_data)

                with tempfile.TemporaryDirectory() as temp_dir:
                    source_rel = "fixtures/%s.pkl" % activation_name
                    target_rel = "fixtures/%s.onnx" % activation_name
                    source_path = os.path.join(
                        temp_dir, "fixtures", "%s.pkl" % activation_name
                    )
                    target_path = os.path.join(
                        temp_dir, "fixtures", "%s.onnx" % activation_name
                    )
                    os.makedirs(os.path.dirname(source_path), exist_ok=True)
                    with open(source_path, "wb") as outfile:
                        pickle.dump(model, outfile)

                    manifest = _single_convertible_manifest(
                        module=module,
                        source_rel=source_rel,
                        target_rel=target_rel,
                        source_path=source_path,
                    )

                    with self._patched_import():
                        report = module.convert_manifest_entries(
                            manifest=manifest,
                            models_dir=temp_dir,
                            run_onnx_checker=False,
                        )

                    result = cast(list[dict[str, object]], report["results"])[0]
                    self.assertEqual(result["status"], "converted")
                    with open(target_path, "rb") as infile:
                        model_bytes = infile.read()

                model_payload = _model_payload(model_bytes)
                observed = _evaluate_fake_onnx(model_payload, input_data)
                assert_parity_close(
                    observed,
                    expected,
                    family="primitives",
                    layer_group="default",
                    context="activation=%s" % activation_name,
                )

                op_types = [
                    cast(str, node["op_type"])
                    for node in cast(
                        list[dict[str, object]],
                        cast(dict[str, object], model_payload["graph"])["nodes"],
                    )
                ]
                activation_ops = {"Tanh", "Sigmoid", "Relu", "Elu", "Softmax"}
                if expected_op is None:
                    self.assertFalse(activation_ops.intersection(op_types))
                else:
                    self.assertIn(expected_op, op_types)

    def test_output_squeeze_parity_matches_scalar_reference(self):
        module = _get_converter_module()
        model = PrimitiveNetwork(
            layers=[
                AverageLayer(axis=None, keepdims=False),
            ]
        )
        input_data = np.array(
            [[1.0, 3.0, -2.0], [2.0, 0.0, 4.0]],
            dtype=np.float32,
        )
        expected = model.process(input_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            source_rel = "fixtures/squeeze_scalar.pkl"
            target_rel = "fixtures/squeeze_scalar.onnx"
            source_path = os.path.join(temp_dir, "fixtures", "squeeze_scalar.pkl")
            target_path = os.path.join(temp_dir, "fixtures", "squeeze_scalar.onnx")
            os.makedirs(os.path.dirname(source_path), exist_ok=True)
            with open(source_path, "wb") as outfile:
                pickle.dump(model, outfile)

            manifest = _single_convertible_manifest(
                module=module,
                source_rel=source_rel,
                target_rel=target_rel,
                source_path=source_path,
            )

            with self._patched_import():
                report = module.convert_manifest_entries(
                    manifest=manifest,
                    models_dir=temp_dir,
                    run_onnx_checker=False,
                )

            result = cast(list[dict[str, object]], report["results"])[0]
            self.assertEqual(result["status"], "converted")
            with open(target_path, "rb") as infile:
                model_bytes = infile.read()

        model_payload = _model_payload(model_bytes)
        observed = _evaluate_fake_onnx(model_payload, input_data)
        self.assertEqual(np.asarray(observed).ndim, 0)
        self.assertEqual(np.asarray(expected).ndim, 0)
        assert_parity_close(
            observed,
            expected,
            family="primitives",
            layer_group="default",
            context="fixture=squeeze_scalar",
        )

        op_types = [
            cast(str, node["op_type"])
            for node in cast(
                list[dict[str, object]],
                cast(dict[str, object], model_payload["graph"])["nodes"],
            )
        ]
        self.assertIn("ReduceMean", op_types)
        self.assertIn("Squeeze", op_types)

    def test_unknown_activation_returns_actionable_conversion_error(self):
        module = _get_converter_module()

        model = PrimitiveNetwork(
            layers=[
                FeedForwardLayer(
                    weights=np.array([[1.0], [2.0]], dtype=np.float32),
                    bias=np.array([0.5], dtype=np.float32),
                    activation_fn=unsupported_activation,
                )
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            source_rel = "fixtures/unsupported_activation.pkl"
            target_rel = "fixtures/unsupported_activation.onnx"
            source_path = os.path.join(
                temp_dir, "fixtures", "unsupported_activation.pkl"
            )
            os.makedirs(os.path.dirname(source_path), exist_ok=True)
            with open(source_path, "wb") as outfile:
                pickle.dump(model, outfile)

            manifest = _single_convertible_manifest(
                module=module,
                source_rel=source_rel,
                target_rel=target_rel,
                source_path=source_path,
            )

            with self._patched_import():
                report = module.convert_manifest_entries(
                    manifest=manifest,
                    models_dir=temp_dir,
                    run_onnx_checker=False,
                )

            result = cast(list[dict[str, object]], report["results"])[0]
            self.assertEqual(result["status"], "conversion_error")
            self.assertIn("UnsupportedActivationError", cast(str, result["error"]))
            self.assertIn("unsupported_activation", cast(str, result["error"]))
            self.assertIn("supported activations", cast(str, result["error"]))

    def test_unsupported_reshape_layout_returns_actionable_conversion_error(self):
        module = _get_converter_module()

        model = PrimitiveNetwork(
            layers=[
                ReshapeLayer(newshape=(4, 1), order="F"),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            source_rel = "fixtures/unsupported_layout.pkl"
            target_rel = "fixtures/unsupported_layout.onnx"
            source_path = os.path.join(temp_dir, "fixtures", "unsupported_layout.pkl")
            os.makedirs(os.path.dirname(source_path), exist_ok=True)
            with open(source_path, "wb") as outfile:
                pickle.dump(model, outfile)

            manifest = _single_convertible_manifest(
                module=module,
                source_rel=source_rel,
                target_rel=target_rel,
                source_path=source_path,
            )

            with self._patched_import():
                report = module.convert_manifest_entries(
                    manifest=manifest,
                    models_dir=temp_dir,
                    run_onnx_checker=False,
                )

            result = cast(list[dict[str, object]], report["results"])[0]
            self.assertEqual(result["status"], "conversion_error")
            self.assertIn("UnsupportedLayoutError", cast(str, result["error"]))
            self.assertIn("order='C'", cast(str, result["error"]))

    def test_unsupported_layer_returns_actionable_converter_unavailable(self):
        module = _get_converter_module()

        model = PrimitiveNetwork(
            layers=[
                UnsupportedLayer(),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            source_rel = "fixtures/unsupported_layer.pkl"
            target_rel = "fixtures/unsupported_layer.onnx"
            source_path = os.path.join(temp_dir, "fixtures", "unsupported_layer.pkl")
            os.makedirs(os.path.dirname(source_path), exist_ok=True)
            with open(source_path, "wb") as outfile:
                pickle.dump(model, outfile)

            manifest = _single_convertible_manifest(
                module=module,
                source_rel=source_rel,
                target_rel=target_rel,
                source_path=source_path,
            )

            with self._patched_import():
                report = module.convert_manifest_entries(
                    manifest=manifest,
                    models_dir=temp_dir,
                    run_onnx_checker=False,
                )

            result = cast(list[dict[str, object]], report["results"])[0]
            self.assertEqual(result["status"], "converter_unavailable")
            self.assertIn("NotImplementedError", cast(str, result["error"]))
            self.assertIn("UnsupportedLayer", cast(str, result["error"]))


if __name__ == "__main__":
    _ = unittest.main()
