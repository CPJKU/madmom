#!/usr/bin/env python
from __future__ import absolute_import, annotations, division, print_function

import argparse
import glob
import hashlib
import importlib
import importlib.machinery
import json
import os
import pickle
import re
import sys
import types
from contextlib import contextmanager
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
from typing import Optional, Protocol, cast

import numpy as np


ManifestEntry = dict[str, object]
ManifestData = dict[str, object]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_MODELS_DIR = os.path.join(REPO_ROOT, "madmom", "models")
DEFAULT_SCHEMA_PATH = os.path.join(DEFAULT_MODELS_DIR, "manifest.schema.json")
DEFAULT_OPSET = 18
MODE_DRY_RUN_INVENTORY = "dry-run-inventory"
MODE_CONVERT = "convert"
MODE_CHECK = "check"

RESULT_CONVERTED = "converted"
RESULT_SKIPPED_OUT_OF_SCOPE = "skipped_out_of_scope"
RESULT_CORRUPT_SOURCE = "corrupt_source"
RESULT_CONVERTER_UNAVAILABLE = "converter_unavailable"
RESULT_CONVERSION_ERROR = "conversion_error"
RESULT_CHECK_FAILED = "check_failed"


class CliArgs(argparse.Namespace):
    dry_run_inventory: bool = False
    convert: bool = False
    check: bool = False
    models_dir: str = DEFAULT_MODELS_DIR
    schema: str = DEFAULT_SCHEMA_PATH
    manifest: str | None = None
    opset: int = DEFAULT_OPSET


ConverterFn = Callable[[str, ManifestEntry, int], object]


class _SerializableModel(Protocol):
    def SerializeToString(self, deterministic: bool = False) -> object: ...


class _OnnxChecker(Protocol):
    def check_model(self, model_path: str) -> None: ...


class _OnnxModule(Protocol):
    checker: _OnnxChecker


class _OnnxTensorProto(Protocol):
    FLOAT: int
    INT64: int


class _OnnxNumpyHelper(Protocol):
    def from_array(self, array: np.ndarray, name: str | None = None) -> object: ...


class _OnnxGraphHelper(Protocol):
    def make_tensor_value_info(
        self, name: str, elem_type: int, shape: object
    ) -> object: ...

    def make_node(
        self,
        op_type: str,
        inputs: Sequence[str],
        outputs: Sequence[str],
        name: str | None = None,
        **kwargs: object,
    ) -> object: ...

    def make_graph(
        self,
        nodes: Sequence[object],
        name: str,
        inputs: Sequence[object],
        outputs: Sequence[object],
        initializer: Sequence[object] | None = None,
    ) -> object: ...

    def make_model(
        self,
        graph: object,
        producer_name: str,
        opset_imports: Sequence[object],
    ) -> object: ...


class _OnnxBuilderModule(_OnnxModule, Protocol):
    TensorProto: _OnnxTensorProto
    helper: _OnnxGraphHelper
    numpy_helper: _OnnxNumpyHelper


class UnsupportedActivationError(ValueError):
    pass


class UnsupportedLayoutError(ValueError):
    pass


class SequentialLayer(object):
    def __init__(self, layers: Sequence[object]):
        self.layers = list(layers)


class ParallelLayer(object):
    def __init__(self, layers: Sequence[object]):
        self.layers = list(layers)


def _infer_layer_input_rank(layer: object) -> int | None:
    layer_type = layer.__class__.__name__

    if layer_type in {"SequentialLayer", "MultiTaskLayer"}:
        sub_layers = getattr(layer, "layers", None)
        if isinstance(sub_layers, list):
            for sub_layer in sub_layers:
                inferred = _infer_layer_input_rank(sub_layer)
                if inferred is not None:
                    return inferred
        return None

    if layer_type == "ParallelLayer":
        sub_layers = getattr(layer, "layers", None)
        if isinstance(sub_layers, list):
            inferred_values = [
                inferred
                for inferred in (
                    _infer_layer_input_rank(sub_layer) for sub_layer in sub_layers
                )
                if inferred is not None
            ]
            if inferred_values:
                return max(inferred_values)
        return None

    if layer_type == "BidirectionalLayer":
        fwd_layer = getattr(layer, "fwd_layer", None)
        if fwd_layer is not None:
            return _infer_layer_input_rank(fwd_layer)
        return None

    if layer_type == "TCNLayer":
        blocks = getattr(layer, "tcn_blocks", None)
        if isinstance(blocks, list) and blocks:
            return _infer_layer_input_rank(blocks[0])
        return 2

    if layer_type == "TCNBlock":
        dilated_conv = getattr(layer, "dilated_conv", None)
        if isinstance(dilated_conv, list) and dilated_conv:
            first_conv = dilated_conv[0]
        else:
            first_conv = dilated_conv
        if first_conv is None:
            return 2
        return _infer_layer_input_rank(first_conv)

    if layer_type == "ConvolutionalLayer":
        weights = np.asarray(getattr(layer, "weights", []), dtype=np.float32)
        if weights.ndim == 4 and int(weights.shape[0]) > 1:
            return 3
        return 2

    if layer_type in {
        "FeedForwardLayer",
        "RecurrentLayer",
        "LSTMLayer",
        "GRULayer",
    }:
        return 2

    return None


def _infer_graph_input_rank(layers: Sequence[object]) -> int:
    if not layers:
        return 2

    for layer in layers:
        inferred = _infer_layer_input_rank(layer)
        if inferred is not None:
            return inferred
    return 2


def _normalized_sys_path_entry(path: str) -> str:
    return os.path.normcase(os.path.abspath(path if path else os.getcwd()))


@contextmanager
def _temporary_repo_root_on_syspath() -> Iterator[None]:
    repo_root_normalized = _normalized_sys_path_entry(REPO_ROOT)
    has_repo_root = any(
        _normalized_sys_path_entry(path) == repo_root_normalized for path in sys.path
    )
    added_repo_root = False
    if not has_repo_root:
        sys.path.insert(0, REPO_ROOT)
        added_repo_root = True

    try:
        yield
    finally:
        if added_repo_root:
            for index, path in enumerate(sys.path):
                if _normalized_sys_path_entry(path) == repo_root_normalized:
                    del sys.path[index]
                    break


@contextmanager
def _temporary_local_madmom_package() -> Iterator[None]:
    package_name = "madmom"
    if package_name in sys.modules:
        yield
        return

    package_dir = os.path.join(REPO_ROOT, package_name)
    module = types.ModuleType(package_name)
    module.__package__ = package_name
    module.__path__ = [package_dir]  # type: ignore[attr-defined]
    module.__file__ = os.path.join(package_dir, "__init__.py")
    module.__spec__ = importlib.machinery.ModuleSpec(
        name=package_name,
        loader=None,
        is_package=True,
    )
    sys.modules[package_name] = module

    try:
        yield
    finally:
        current = sys.modules.get(package_name)
        if current is module:
            del sys.modules[package_name]


@contextmanager
def _temporary_numpy_shape_base_alias() -> Iterator[None]:
    legacy_module_name = "numpy.lib.shape_base"
    if legacy_module_name in sys.modules:
        yield
        return

    try:
        replacement_module = importlib.import_module("numpy.lib._shape_base_impl")
    except ImportError:
        yield
        return

    sys.modules[legacy_module_name] = replacement_module
    try:
        yield
    finally:
        current = sys.modules.get(legacy_module_name)
        if current is replacement_module:
            del sys.modules[legacy_module_name]


def _as_dict(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return None


def _as_list(value: object) -> list[object] | None:
    if isinstance(value, list):
        return cast(list[object], value)
    return None


def _relative_posix_path(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def _iter_model_pickles(models_dir: str) -> list[str]:
    pattern = os.path.join(models_dir, "**", "*.pkl")
    return sorted(glob.glob(pattern, recursive=True))


def _sha256_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as infile:
        while True:
            data = infile.read(1024 * 1024)
            if not data:
                break
            digest.update(data)
    return "sha256:%s" % digest.hexdigest()


def _classify_by_path(
    source_pkl: str, model_hash: str, target_onnx: str, opset: int
) -> ManifestEntry:
    basename = os.path.basename(source_pkl).lower()

    if source_pkl.startswith("patterns/"):
        return {
            "source_pkl": source_pkl,
            "target_onnx": None,
            "hash": model_hash,
            "opset": None,
            "model_type": "gmm_pattern_model",
            "status": "out_of_scope",
            "rationale": "GMM pattern artifact is outside the NN conversion scope.",
            "conversion_mode": "none",
            "convertible": False,
        }

    if source_pkl.startswith("chords/") and "crf" in basename:
        return {
            "source_pkl": source_pkl,
            "target_onnx": None,
            "hash": model_hash,
            "opset": None,
            "model_type": "conditional_random_field",
            "status": "out_of_scope",
            "rationale": "CRF artifact is not an nn-convertible model.",
            "conversion_mode": "none",
            "convertible": False,
        }

    if source_pkl.startswith("notes/") and "cnn" in basename:
        return {
            "source_pkl": source_pkl,
            "target_onnx": target_onnx,
            "hash": model_hash,
            "opset": opset,
            "model_type": "wrapped_nn_processor",
            "status": "nn_convertible",
            "rationale": "Notes CNN artifact is a SequentialProcessor wrapper; convert only the wrapped NN core.",
            "conversion_mode": "wrapped_nn_core",
            "convertible": True,
        }

    nn_family_prefixes = (
        "beats/",
        "chroma/",
        "chords/",
        "downbeats/",
        "key/",
        "notes/",
        "onsets/",
    )
    if source_pkl.startswith(nn_family_prefixes):
        return {
            "source_pkl": source_pkl,
            "target_onnx": target_onnx,
            "hash": model_hash,
            "opset": opset,
            "model_type": "neural_network",
            "status": "nn_convertible",
            "rationale": "Artifact contains NN layers and is in conversion scope.",
            "conversion_mode": "direct_nn",
            "convertible": True,
        }

    return {
        "source_pkl": source_pkl,
        "target_onnx": None,
        "hash": model_hash,
        "opset": None,
        "model_type": "unsupported_artifact",
        "status": "out_of_scope",
        "rationale": "Artifact does not expose an NN model supported by this migration.",
        "conversion_mode": "none",
        "convertible": False,
    }


def classify_artifact(
    source_file: str, models_dir: str, opset: int = DEFAULT_OPSET
) -> ManifestEntry:
    source_pkl = _relative_posix_path(source_file, models_dir)
    target_onnx = source_pkl[:-4] + ".onnx"
    model_hash = _sha256_digest(source_file)
    return _classify_by_path(source_pkl, model_hash, target_onnx, opset)


def build_supported_artifact_matrix(
    models_dir: str = DEFAULT_MODELS_DIR, opset: int = DEFAULT_OPSET
) -> ManifestData:
    artifacts: list[ManifestEntry] = []
    for source_file in _iter_model_pickles(models_dir):
        artifacts.append(classify_artifact(source_file, models_dir, opset=opset))
    return {"schema_version": 1, "artifacts": artifacts}


def _json_type_matches(value: object, json_type: str) -> bool:
    if json_type == "object":
        return isinstance(value, dict)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "null":
        return value is None
    return True


def validate_manifest_shape(manifest: object, schema: object) -> list[str]:
    errors: list[str] = []
    manifest_dict = _as_dict(manifest)
    schema_dict = _as_dict(schema)
    if manifest_dict is None:
        return ["manifest must be a JSON object"]
    if schema_dict is None:
        return ["schema must be a JSON object"]

    required_keys = _as_list(schema_dict.get("required"))
    if required_keys is not None:
        for required_key in required_keys:
            if isinstance(required_key, str) and required_key not in manifest_dict:
                errors.append("missing top-level key: %s" % required_key)

    artifacts = _as_list(manifest_dict.get("artifacts"))
    if artifacts is None:
        errors.append("top-level artifacts must be a list")
        return errors

    properties = _as_dict(schema_dict.get("properties"))
    artifacts_schema = _as_dict(properties.get("artifacts") if properties else None)
    item_schema = _as_dict(artifacts_schema.get("items") if artifacts_schema else None)

    required = _as_list(item_schema.get("required") if item_schema else None)
    property_map = _as_dict(item_schema.get("properties") if item_schema else None)
    allow_extra_value = item_schema.get("additionalProperties") if item_schema else True
    allow_extra = bool(allow_extra_value)

    for index, artifact in enumerate(artifacts):
        prefix = "artifacts[%d]" % index
        artifact_dict = _as_dict(artifact)
        if artifact_dict is None:
            errors.append("%s must be an object" % prefix)
            continue

        if required is not None:
            for required_key in required:
                if isinstance(required_key, str) and required_key not in artifact_dict:
                    errors.append("%s missing key: %s" % (prefix, required_key))

        if not allow_extra and property_map is not None:
            for key in artifact_dict:
                if key not in property_map:
                    errors.append("%s has unexpected key: %s" % (prefix, key))

        if property_map is None:
            continue

        for key, prop_value in property_map.items():
            prop = _as_dict(prop_value)
            if prop is None or key not in artifact_dict:
                continue
            value = artifact_dict[key]

            expected_type = prop.get("type")
            if isinstance(expected_type, list):
                valid = False
                for type_value in cast(list[object], expected_type):
                    if isinstance(type_value, str) and _json_type_matches(
                        value, type_value
                    ):
                        valid = True
                if not valid:
                    errors.append("%s.%s has invalid type" % (prefix, key))
            elif isinstance(expected_type, str):
                if not _json_type_matches(value, expected_type):
                    errors.append("%s.%s has invalid type" % (prefix, key))

            enum_values = _as_list(prop.get("enum"))
            if enum_values is not None and value not in enum_values:
                errors.append("%s.%s has unsupported value" % (prefix, key))

            pattern_value = prop.get("pattern")
            if isinstance(pattern_value, str) and isinstance(value, str):
                if re.match(pattern_value, value) is None:
                    errors.append(
                        "%s.%s does not match expected pattern" % (prefix, key)
                    )

        status = artifact_dict.get("status")
        convertible = artifact_dict.get("convertible")
        if status == "nn_convertible" and convertible is not True:
            errors.append(
                "%s.convertible must be true for nn_convertible status" % prefix
            )
        if status == "out_of_scope" and convertible is not False:
            errors.append(
                "%s.convertible must be false for out_of_scope status" % prefix
            )

    return errors


def validate_conversion_report_shape(report: object) -> list[str]:
    errors: list[str] = []
    report_dict = _as_dict(report)
    if report_dict is None:
        return ["report must be a JSON object"]

    results = _as_list(report_dict.get("results"))
    if results is None:
        return ["top-level results must be a list"]

    mode_value = report_dict.get("mode")
    if mode_value is not None and mode_value != MODE_CONVERT:
        errors.append("mode must be 'convert' when provided")

    for index, result in enumerate(results):
        prefix = "results[%d]" % index
        result_dict = _as_dict(result)
        if result_dict is None:
            errors.append("%s must be an object" % prefix)
            continue

        required_keys = [
            "source_pkl",
            "target_onnx",
            "source_hash",
            "status",
            "output_hash",
            "checker_invoked",
            "error",
        ]
        for required_key in required_keys:
            if required_key not in result_dict:
                errors.append("%s missing key: %s" % (prefix, required_key))

        source_pkl = result_dict.get("source_pkl")
        if not isinstance(source_pkl, str):
            errors.append("%s.source_pkl has invalid type" % prefix)

        target_onnx = result_dict.get("target_onnx")
        if target_onnx is not None and not isinstance(target_onnx, str):
            errors.append("%s.target_onnx has invalid type" % prefix)

        source_hash = result_dict.get("source_hash")
        if not isinstance(source_hash, str):
            errors.append("%s.source_hash has invalid type" % prefix)

        status = result_dict.get("status")
        if not isinstance(status, str):
            errors.append("%s.status has invalid type" % prefix)

        output_hash = result_dict.get("output_hash")
        if output_hash is not None and not isinstance(output_hash, str):
            errors.append("%s.output_hash has invalid type" % prefix)

        checker_invoked = result_dict.get("checker_invoked")
        if not isinstance(checker_invoked, bool):
            errors.append("%s.checker_invoked has invalid type" % prefix)

        error_value = result_dict.get("error")
        if error_value is not None and not isinstance(error_value, str):
            errors.append("%s.error has invalid type" % prefix)

    return errors


def validate_check_input_shape(manifest: object, schema: object) -> list[str]:
    manifest_dict = _as_dict(manifest)
    if manifest_dict is None:
        return ["manifest must be a JSON object"]

    artifacts = _as_list(manifest_dict.get("artifacts"))
    if artifacts is not None:
        return validate_manifest_shape(manifest, schema)

    results = _as_list(manifest_dict.get("results"))
    if results is not None:
        return validate_conversion_report_shape(manifest)

    return ["manifest must contain an artifacts or results list"]


def _load_json(path: str) -> object:
    with open(path, "r") as infile:
        return cast(object, json.load(infile))


def _write_json(data: object, path: str) -> None:
    with open(path, "w") as outfile:
        json.dump(data, outfile, indent=2, sort_keys=True)
        _ = outfile.write("\n")


def determine_mode(cli_args: CliArgs) -> str:
    selected: list[str] = []
    if cli_args.dry_run_inventory:
        selected.append(MODE_DRY_RUN_INVENTORY)
    if cli_args.convert:
        selected.append(MODE_CONVERT)
    if cli_args.check:
        selected.append(MODE_CHECK)

    if len(selected) > 1:
        raise SystemExit(
            "use exactly one of --dry-run-inventory, --convert, or --check"
        )
    if not selected:
        return MODE_DRY_RUN_INVENTORY
    return selected[0]


def _serialize_onnx_model_bytes(model: object) -> bytes:
    if isinstance(model, bytes):
        return model
    if isinstance(model, bytearray):
        return bytes(model)

    serializable = cast(_SerializableModel, model)
    try:
        serialize = serializable.SerializeToString
    except AttributeError as error:
        raise TypeError(
            "converter must return ONNX bytes or a serializable model"
        ) from error
    if not callable(serialize):
        raise TypeError("converter must return ONNX bytes or a serializable model")

    serialized: object
    try:
        serialized = serialize(deterministic=True)
    except TypeError:
        serialized = serialize()

    if isinstance(serialized, bytes):
        return serialized
    if isinstance(serialized, bytearray):
        return bytes(serialized)
    raise TypeError("SerializeToString() must return bytes")


def _artifact_source_sort_key(artifact: object) -> str:
    artifact_dict = _as_dict(artifact)
    if artifact_dict is None:
        return ""
    source_pkl = artifact_dict.get("source_pkl")
    if isinstance(source_pkl, str):
        return source_pkl
    return ""


def write_deterministic_onnx(model: object, output_path: str) -> str:
    onnx_bytes = _serialize_onnx_model_bytes(model)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as outfile:
        _ = outfile.write(onnx_bytes)
    return "sha256:%s" % hashlib.sha256(onnx_bytes).hexdigest()


def run_onnx_checker_if_available(model_path: str) -> bool:
    try:
        imported_module = importlib.import_module("onnx")
        onnx_module = cast(_OnnxModule, cast(object, imported_module))
    except ImportError:
        return False

    onnx_module.checker.check_model(model_path)
    return True


def _import_onnx_for_conversion() -> _OnnxBuilderModule:
    try:
        imported_module = importlib.import_module("onnx")
    except ImportError as error:
        raise NotImplementedError(
            "primitive NN conversion requires optional dependency 'onnx'"
        ) from error
    return cast(_OnnxBuilderModule, cast(object, imported_module))


class _PrimitiveGraphBuilder:
    def __init__(self, onnx_module: _OnnxBuilderModule, opset: int):
        self.onnx_module = onnx_module
        self.opset = opset
        self.nodes: list[object] = []
        self.initializers: list[object] = []
        self.current_tensor = "input"
        self.current_rank: int | None = 3
        self._counter = 0

    def _name(self, stem: str) -> str:
        name = "%s_%d" % (stem, self._counter)
        self._counter += 1
        return name

    def add_initializer(self, stem: str, array: np.ndarray) -> str:
        name = self._name(stem)
        initializer = self.onnx_module.numpy_helper.from_array(array, name=name)
        self.initializers.append(initializer)
        return name

    def add_node(
        self,
        op_type: str,
        inputs: Sequence[str],
        stem: str,
        **attrs: object,
    ) -> str:
        output = self._name(stem)
        node = self.onnx_module.helper.make_node(
            op_type,
            list(inputs),
            [output],
            name=output,
            **attrs,
        )
        self.nodes.append(node)
        self.current_tensor = output
        return output


def _layer_label(index: int, layer: object) -> str:
    return "%s[%d]" % (layer.__class__.__name__, index)


def _require_attr(layer: object, name: str, label: str) -> object:
    if not hasattr(layer, name):
        raise UnsupportedLayoutError(
            "%s is missing required attribute '%s'" % (label, name)
        )
    return getattr(layer, name)


def _as_float32_array(value: object, what: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise UnsupportedLayoutError("%s must be numeric" % what) from error
    return array


def _as_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UnsupportedLayoutError("%s must be an int" % what)
    return value


def _as_positive_int(value: object, what: str) -> int:
    int_value = _as_int(value, what)
    if int_value <= 0:
        raise UnsupportedLayoutError("%s must be > 0" % what)
    return int_value


def _as_int_pair(value: object, what: str) -> tuple[int, int]:
    if isinstance(value, bool):
        raise UnsupportedLayoutError("%s must be int or length-2 tuple/list" % what)
    if isinstance(value, int):
        pair = (value, value)
    elif isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise UnsupportedLayoutError("%s must contain exactly 2 ints" % what)
        first = _as_positive_int(value[0], "%s[0]" % what)
        second = _as_positive_int(value[1], "%s[1]" % what)
        pair = (first, second)
    else:
        raise UnsupportedLayoutError("%s must be int or length-2 tuple/list" % what)

    if pair[0] <= 0 or pair[1] <= 0:
        raise UnsupportedLayoutError("%s values must be > 0" % what)
    return pair


def _add_axes_initializer(
    builder: _PrimitiveGraphBuilder, stem: str, axes: Sequence[int]
) -> str:
    return builder.add_initializer(stem, np.asarray(list(axes), dtype=np.int64))


def _append_unsqueeze(
    builder: _PrimitiveGraphBuilder,
    axes: Sequence[int],
    stem: str,
) -> None:
    axes_name = _add_axes_initializer(builder, "%s_axes" % stem, axes)
    _ = builder.add_node("Unsqueeze", [builder.current_tensor, axes_name], stem)
    if builder.current_rank is not None:
        builder.current_rank += len(axes)


def _append_squeeze(
    builder: _PrimitiveGraphBuilder,
    axes: Sequence[int],
    stem: str,
) -> None:
    axes_name = _add_axes_initializer(builder, "%s_axes" % stem, axes)
    _ = builder.add_node("Squeeze", [builder.current_tensor, axes_name], stem)
    if builder.current_rank is not None:
        builder.current_rank -= len(axes)


def _prepare_spatial_input_for_nchw(
    builder: _PrimitiveGraphBuilder,
    label: str,
    what: str,
) -> bool:
    if builder.current_rank == 2:
        _append_unsqueeze(builder, [2], "%s_expand_channel" % what)
        if builder.current_rank != 3:
            raise UnsupportedLayoutError(
                "%s internal error while preparing channels" % label
            )
        channel_added = True
    elif builder.current_rank == 3:
        channel_added = False
    else:
        raise UnsupportedLayoutError(
            "%s requires rank-2 or rank-3 input in [time, freq, channel] layout" % label
        )

    _append_unsqueeze(builder, [0], "%s_add_batch" % what)
    _ = builder.add_node(
        "Transpose",
        [builder.current_tensor],
        "%s_to_nchw" % what,
        perm=[0, 3, 1, 2],
    )
    builder.current_rank = 4
    return channel_added


def _restore_spatial_output_from_nchw(
    builder: _PrimitiveGraphBuilder,
    *,
    channel_added: bool,
    what: str,
    remove_added_channel: bool = True,
) -> None:
    _ = builder.add_node(
        "Transpose",
        [builder.current_tensor],
        "%s_to_nhwc" % what,
        perm=[0, 2, 3, 1],
    )
    builder.current_rank = 4
    _append_squeeze(builder, [0], "%s_remove_batch" % what)
    builder.current_rank = 3
    if channel_added and remove_added_channel:
        _append_squeeze(builder, [2], "%s_remove_channel" % what)
        builder.current_rank = 2


def _normalize_axes(axis_values: Sequence[int], rank: int, label: str) -> list[int]:
    normalized: list[int] = []
    for axis in axis_values:
        normalized_axis = axis
        if axis < 0:
            normalized_axis = rank + axis
        if normalized_axis < 0 or normalized_axis >= rank:
            raise UnsupportedLayoutError(
                "%s axis %d is out of bounds for rank %d" % (label, axis, rank)
            )
        normalized.append(normalized_axis)
    if len(set(normalized)) != len(normalized):
        raise UnsupportedLayoutError("%s axes must be unique" % label)
    return normalized


def _activation_name(activation_fn: object, label: str) -> str | None:
    if activation_fn is None:
        return None
    activation_name = getattr(activation_fn, "__name__", None)
    if not isinstance(activation_name, str):
        raise UnsupportedActivationError(
            "%s activation must expose a __name__ attribute" % label
        )
    if activation_name in {"linear", "tanh", "sigmoid", "relu", "elu", "softmax"}:
        return activation_name
    raise UnsupportedActivationError(
        "%s activation '%s' is unsupported; supported activations: "
        "linear, tanh, sigmoid, relu, elu, softmax" % (label, activation_name)
    )


def _append_activation(
    builder: _PrimitiveGraphBuilder,
    layer: object,
    label: str,
) -> None:
    activation_name = _activation_name(
        _require_attr(layer, "activation_fn", label), label
    )
    if activation_name is None or activation_name == "linear":
        return

    if activation_name == "softmax":
        if builder.current_rank is not None and builder.current_rank < 2:
            raise UnsupportedLayoutError("%s softmax requires rank >= 2" % label)
        _ = builder.add_node("Softmax", [builder.current_tensor], "softmax", axis=-1)
        return

    op_map = {
        "tanh": "Tanh",
        "sigmoid": "Sigmoid",
        "relu": "Relu",
        "elu": "Elu",
    }
    op_type = op_map[activation_name]
    if op_type == "Elu":
        _ = builder.add_node(op_type, [builder.current_tensor], "elu", alpha=1.0)
    else:
        _ = builder.add_node(op_type, [builder.current_tensor], activation_name)


def _as_1d_float32_array(value: object, what: str) -> np.ndarray:
    array = _as_float32_array(value, what)
    if array.ndim != 1:
        raise UnsupportedLayoutError(
            "%s must be rank-1; got shape %s" % (what, tuple(array.shape))
        )
    return array


def _onnx_recurrent_activation_name(
    activation_fn: object,
    label: str,
    allowed: set[str],
) -> str:
    activation_name = _activation_name(activation_fn, label)
    if activation_name is None:
        raise UnsupportedActivationError(
            "%s recurrent activation must not be None" % label
        )

    if activation_name not in allowed:
        supported = ", ".join(sorted(allowed))
        raise UnsupportedActivationError(
            "%s recurrent activation '%s' is unsupported; supported activations: %s"
            % (label, activation_name, supported)
        )

    activation_map = {
        "tanh": "Tanh",
        "sigmoid": "Sigmoid",
        "relu": "Relu",
    }
    return activation_map[activation_name]


def _temporarily_convert_tensor(
    builder: _PrimitiveGraphBuilder,
    tensor_name: str,
    tensor_rank: int | None,
    axes: Sequence[int],
    stem: str,
    op_type: str,
) -> str:
    previous_tensor = builder.current_tensor
    previous_rank = builder.current_rank
    builder.current_tensor = tensor_name
    builder.current_rank = tensor_rank
    if op_type == "Unsqueeze":
        _append_unsqueeze(builder, axes, stem)
    else:
        _append_squeeze(builder, axes, stem)
    output = builder.current_tensor
    builder.current_tensor = previous_tensor
    builder.current_rank = previous_rank
    return output


def _unsqueeze_tensor(
    builder: _PrimitiveGraphBuilder,
    tensor_name: str,
    tensor_rank: int | None,
    axes: Sequence[int],
    stem: str,
) -> str:
    return _temporarily_convert_tensor(
        builder=builder,
        tensor_name=tensor_name,
        tensor_rank=tensor_rank,
        axes=axes,
        stem=stem,
        op_type="Unsqueeze",
    )


def _squeeze_tensor(
    builder: _PrimitiveGraphBuilder,
    tensor_name: str,
    tensor_rank: int | None,
    axes: Sequence[int],
    stem: str,
) -> str:
    return _temporarily_convert_tensor(
        builder=builder,
        tensor_name=tensor_name,
        tensor_rank=tensor_rank,
        axes=axes,
        stem=stem,
        op_type="Squeeze",
    )


def _append_multi_output_node(
    builder: _PrimitiveGraphBuilder,
    op_type: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    stem: str,
    **attrs: object,
) -> None:
    node_name = builder._name(stem)
    node = builder.onnx_module.helper.make_node(
        op_type,
        list(inputs),
        list(outputs),
        name=node_name,
        **attrs,
    )
    builder.nodes.append(node)


TensorReference = tuple[str, Optional[int]]
TensorReferences = list[TensorReference]


def _validate_state_size(
    state: np.ndarray,
    expected_size: int,
    what: str,
) -> None:
    if state.size != expected_size:
        raise UnsupportedLayoutError(
            "%s size mismatch: expected %d, got %d" % (what, expected_size, state.size)
        )


def _validate_gate_shapes(
    gate: object,
    *,
    gate_name: str,
    label: str,
    input_size: int,
    hidden_size: int,
    allow_peephole: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, object]:
    gate_label = "%s.%s" % (label, gate_name)
    weights = _as_float32_array(
        _require_attr(gate, "weights", gate_label), "%s.weights" % gate_label
    )
    recurrent_weights = _as_float32_array(
        _require_attr(gate, "recurrent_weights", gate_label),
        "%s.recurrent_weights" % gate_label,
    )
    bias = _as_1d_float32_array(
        _require_attr(gate, "bias", gate_label), "%s.bias" % gate_label
    )

    if weights.shape != (input_size, hidden_size):
        raise UnsupportedLayoutError(
            "%s.weights must have shape (%d, %d); got %s"
            % (gate_label, input_size, hidden_size, tuple(weights.shape))
        )
    if recurrent_weights.shape != (hidden_size, hidden_size):
        raise UnsupportedLayoutError(
            "%s.recurrent_weights must have shape (%d, %d); got %s"
            % (gate_label, hidden_size, hidden_size, tuple(recurrent_weights.shape))
        )
    if bias.size != hidden_size:
        raise UnsupportedLayoutError(
            "%s.bias size mismatch: expected %d, got %d"
            % (gate_label, hidden_size, bias.size)
        )

    peephole_value = getattr(gate, "peephole_weights", None)
    peephole_weights: np.ndarray | None
    if peephole_value is None:
        peephole_weights = None
    else:
        if not allow_peephole:
            raise UnsupportedLayoutError(
                "%s peephole weights are unsupported in this context" % gate_label
            )
        peephole_weights = _as_1d_float32_array(
            peephole_value, "%s.peephole_weights" % gate_label
        )
        if peephole_weights.size != hidden_size:
            raise UnsupportedLayoutError(
                "%s.peephole_weights size mismatch: expected %d, got %d"
                % (gate_label, hidden_size, peephole_weights.size)
            )

    activation_fn = _require_attr(gate, "activation_fn", gate_label)
    return weights, recurrent_weights, bias, peephole_weights, activation_fn


def _convert_feed_forward_layer(
    builder: _PrimitiveGraphBuilder,
    layer: object,
    label: str,
) -> None:
    weights = _as_float32_array(
        _require_attr(layer, "weights", label), "%s.weights" % label
    )
    if weights.ndim not in (1, 2):
        raise UnsupportedLayoutError(
            "%s weights must be rank-1 or rank-2; got rank %d" % (label, weights.ndim)
        )

    if builder.current_rank is not None and builder.current_rank < 1:
        raise UnsupportedLayoutError("%s input rank must be >= 1 for MatMul" % label)

    bias = _as_float32_array(_require_attr(layer, "bias", label), "%s.bias" % label)
    bias = bias.reshape(-1)

    weights_name = builder.add_initializer("weights", weights)
    bias_name = builder.add_initializer("bias", bias)
    matmul_output = builder.add_node(
        "MatMul",
        [builder.current_tensor, weights_name],
        "matmul",
    )
    _ = builder.add_node("Add", [matmul_output, bias_name], "bias_add")

    if builder.current_rank is not None:
        if weights.ndim == 1:
            builder.current_rank = builder.current_rank - 1
        else:
            builder.current_rank = builder.current_rank

    _append_activation(builder, layer, label)


def _convert_batch_norm_layer(
    builder: _PrimitiveGraphBuilder,
    layer: object,
    label: str,
) -> None:
    beta = _as_float32_array(_require_attr(layer, "beta", label), "%s.beta" % label)
    gamma = _as_float32_array(_require_attr(layer, "gamma", label), "%s.gamma" % label)
    mean = _as_float32_array(_require_attr(layer, "mean", label), "%s.mean" % label)
    inv_std = _as_float32_array(
        _require_attr(layer, "inv_std", label), "%s.inv_std" % label
    )

    scale = gamma * inv_std
    offset = beta - (mean * scale)

    scale_name = builder.add_initializer("batchnorm_scale", scale)
    offset_name = builder.add_initializer("batchnorm_offset", offset)
    scaled = builder.add_node(
        "Mul",
        [builder.current_tensor, scale_name],
        "batchnorm_mul",
    )
    _ = builder.add_node("Add", [scaled, offset_name], "batchnorm_add")
    _append_activation(builder, layer, label)


def _convert_reshape_layer(
    builder: _PrimitiveGraphBuilder,
    layer: object,
    label: str,
) -> None:
    order = _require_attr(layer, "order", label)
    if order != "C":
        raise UnsupportedLayoutError(
            "%s supports only order='C'; got order=%r" % (label, order)
        )

    newshape_value = _require_attr(layer, "newshape", label)
    if isinstance(newshape_value, int):
        newshape = [newshape_value]
    elif isinstance(newshape_value, (list, tuple)):
        newshape = list(newshape_value)
    else:
        raise UnsupportedLayoutError("%s newshape must be int, list, or tuple" % label)

    normalized_shape: list[int] = []
    for dim in newshape:
        if isinstance(dim, np.integer):
            normalized_dim: object = int(dim)
        else:
            normalized_dim = dim

        if isinstance(normalized_dim, bool) or not isinstance(normalized_dim, int):
            raise UnsupportedLayoutError("%s newshape values must be ints" % label)
        if normalized_dim == 0:
            raise UnsupportedLayoutError(
                "%s newshape with 0 is unsupported for ONNX parity" % label
            )
        normalized_shape.append(normalized_dim)

    shape_name = builder.add_initializer(
        "reshape_shape", np.asarray(normalized_shape, dtype=np.int64)
    )
    _ = builder.add_node("Reshape", [builder.current_tensor, shape_name], "reshape")
    builder.current_rank = len(normalized_shape)


def _convert_transpose_layer(
    builder: _PrimitiveGraphBuilder,
    layer: object,
    label: str,
) -> None:
    axes = _require_attr(layer, "axes", label)
    if axes is None:
        if builder.current_rank is None:
            raise UnsupportedLayoutError(
                "%s axes=None requires known rank for reverse permutation" % label
            )
        _ = builder.add_node("Transpose", [builder.current_tensor], "transpose")
        return

    if not isinstance(axes, (list, tuple)):
        raise UnsupportedLayoutError("%s axes must be list or tuple" % label)

    permutation: list[int] = []
    for axis in axes:
        if not isinstance(axis, int):
            raise UnsupportedLayoutError("%s axes must contain ints" % label)
        permutation.append(axis)

    if builder.current_rank is not None:
        permutation = _normalize_axes(permutation, builder.current_rank, label)
    if len(set(permutation)) != len(permutation):
        raise UnsupportedLayoutError("%s axes must be unique" % label)
    if sorted(permutation) != list(range(len(permutation))):
        raise UnsupportedLayoutError(
            "%s axes must be a full permutation of [0, ..., rank-1]" % label
        )

    _ = builder.add_node(
        "Transpose",
        [builder.current_tensor],
        "transpose",
        perm=permutation,
    )
    builder.current_rank = len(permutation)


def _convert_pad_layer(
    builder: _PrimitiveGraphBuilder,
    layer: object,
    label: str,
) -> None:
    if builder.current_rank is None:
        raise UnsupportedLayoutError(
            "%s requires known input rank for static pads" % label
        )

    width_value = _require_attr(layer, "width", label)
    if not isinstance(width_value, int):
        raise UnsupportedLayoutError("%s width must be an int" % label)
    if width_value < 0:
        raise UnsupportedLayoutError("%s width must be non-negative" % label)

    axes_value = _require_attr(layer, "axes", label)
    if not isinstance(axes_value, (list, tuple)):
        raise UnsupportedLayoutError("%s axes must be list or tuple" % label)

    axes: list[int] = []
    for axis in axes_value:
        if not isinstance(axis, int):
            raise UnsupportedLayoutError("%s axes must contain ints" % label)
        axes.append(axis)
    normalized_axes = _normalize_axes(axes, builder.current_rank, label)

    pads = np.zeros(builder.current_rank * 2, dtype=np.int64)
    for axis in normalized_axes:
        pads[axis] = width_value
        pads[builder.current_rank + axis] = width_value

    value_array = _as_float32_array(
        _require_attr(layer, "value", label), "%s.value" % label
    )
    if value_array.size != 1:
        raise UnsupportedLayoutError("%s value must be scalar" % label)

    pads_name = builder.add_initializer("pad_width", pads)
    value_name = builder.add_initializer(
        "pad_value", np.asarray(value_array.item(), dtype=np.float32)
    )
    _ = builder.add_node(
        "Pad",
        [builder.current_tensor, pads_name, value_name],
        "pad",
        mode="constant",
    )


def _convert_average_layer(
    builder: _PrimitiveGraphBuilder,
    layer: object,
    label: str,
) -> None:
    dtype_value = _require_attr(layer, "dtype", label)
    if dtype_value is not None:
        dtype_name: str | None = None
        if dtype_value is np.float32:
            dtype_name = "float32"
        elif isinstance(dtype_value, np.dtype):
            dtype_name = dtype_value.name
        elif isinstance(dtype_value, str):
            dtype_name = dtype_value

        if dtype_name not in {"float32", "single"}:
            raise UnsupportedLayoutError(
                "%s supports only dtype=None or dtype=np.float32" % label
            )
        _ = builder.add_node(
            "Cast",
            [builder.current_tensor],
            "average_cast",
            to=builder.onnx_module.TensorProto.FLOAT,
        )

    axis_value = _require_attr(layer, "axis", label)
    keepdims_value = _require_attr(layer, "keepdims", label)
    keepdims = bool(keepdims_value)

    axes: list[int] | None
    if axis_value is None:
        axes = None
    elif isinstance(axis_value, int):
        axes = [axis_value]
    elif isinstance(axis_value, (list, tuple)):
        axes = []
        for axis in axis_value:
            if not isinstance(axis, int):
                raise UnsupportedLayoutError("%s axis values must be ints" % label)
            axes.append(axis)
    else:
        raise UnsupportedLayoutError(
            "%s axis must be None, int, list, or tuple" % label
        )

    normalized_axes: list[int] | None = axes
    if axes is not None and builder.current_rank is not None:
        normalized_axes = _normalize_axes(axes, builder.current_rank, label)

    reduce_inputs = [builder.current_tensor]
    if normalized_axes is not None:
        axes_name = builder.add_initializer(
            "reduce_axes", np.asarray(normalized_axes, dtype=np.int64)
        )
        reduce_inputs.append(axes_name)

    _ = builder.add_node(
        "ReduceMean",
        reduce_inputs,
        "reduce_mean",
        keepdims=1 if keepdims else 0,
    )

    if builder.current_rank is not None:
        if normalized_axes is None:
            builder.current_rank = builder.current_rank if keepdims else 0
        else:
            reduced_rank = builder.current_rank - len(normalized_axes)
            builder.current_rank = builder.current_rank if keepdims else reduced_rank


def _convert_convolutional_layer(
    builder: _PrimitiveGraphBuilder,
    layer: object,
    label: str,
) -> None:
    weights = _as_float32_array(
        _require_attr(layer, "weights", label), "%s.weights" % label
    )
    if weights.ndim != 4:
        raise UnsupportedLayoutError(
            "%s weights must have rank 4 [channel, feature, time, freq]" % label
        )

    num_channels = weights.shape[0]
    bias = _as_float32_array(_require_attr(layer, "bias", label), "%s.bias" % label)
    bias = bias.reshape(-1)
    output_features = weights.shape[1]
    if bias.size == 1 and output_features > 1:
        bias = np.repeat(bias, output_features)
    elif bias.size != output_features:
        raise UnsupportedLayoutError(
            "%s bias size (%d) must be 1 or match number of feature maps (%d)"
            % (label, bias.size, output_features)
        )

    stride_value = _require_attr(layer, "stride", label)
    if stride_value in (None, 1, (1, 1)):
        strides = (1, 1)
    else:
        strides = _as_int_pair(stride_value, "%s.stride" % label)

    pad_value = _require_attr(layer, "pad", label)
    if not isinstance(pad_value, str):
        raise UnsupportedLayoutError("%s pad must be a string" % label)
    if pad_value not in {"valid", "same"}:
        raise UnsupportedLayoutError(
            "%s supports only pad='valid' or pad='same'; got %r" % (label, pad_value)
        )

    if builder.current_rank == 2 and num_channels != 1:
        raise UnsupportedLayoutError(
            "%s rank-2 input implies one channel, but weights expect %d channels"
            % (label, num_channels)
        )
    if builder.current_rank not in (2, 3):
        raise UnsupportedLayoutError(
            "%s requires rank-2 or rank-3 input in [time, freq, channel] layout" % label
        )

    channel_added = _prepare_spatial_input_for_nchw(builder, label, "conv")
    if channel_added and num_channels != 1:
        raise UnsupportedLayoutError(
            "%s rank-2 input can only be used with single-channel weights" % label
        )

    transformed_weights = np.flip(weights, axis=(2, 3)).transpose(1, 0, 2, 3)
    weights_name = builder.add_initializer("conv_weights", transformed_weights)
    bias_name = builder.add_initializer("conv_bias", bias)

    conv_attrs: dict[str, object] = {
        "strides": [strides[0], strides[1]],
    }
    if pad_value == "same":
        conv_attrs["auto_pad"] = "SAME_UPPER"

    _ = builder.add_node(
        "Conv",
        [builder.current_tensor, weights_name, bias_name],
        "conv",
        **conv_attrs,
    )
    _restore_spatial_output_from_nchw(
        builder,
        channel_added=channel_added,
        what="conv",
        remove_added_channel=False,
    )
    builder.current_rank = 3
    _append_activation(builder, layer, label)


def _convert_max_pool_layer(
    builder: _PrimitiveGraphBuilder,
    layer: object,
    label: str,
) -> None:
    axis_value = _require_attr(layer, "axis", label)
    stride_value = _require_attr(layer, "stride", label)

    def normalize_pool_pair_value(value: object) -> object:
        if isinstance(value, (bool, int, list, tuple)):
            return value

        value_array = np.asarray(value)
        if value_array.ndim == 0:
            return value_array.item()
        if value_array.ndim == 1:
            return value_array.tolist()
        return value

    if axis_value is not None:
        if stride_value is not None:
            raise UnsupportedLayoutError(
                "%s does not support axis pooling when stride is set" % label
            )

        if isinstance(axis_value, int):
            axes = [axis_value]
        elif isinstance(axis_value, (list, tuple)):
            axes = [_as_int(axis, "%s.axis" % label) for axis in axis_value]
        else:
            raise UnsupportedLayoutError(
                "%s axis must be an int, tuple, list, or None" % label
            )

        if builder.current_rank is not None:
            axes = _normalize_axes(axes, builder.current_rank, label)
        axes_name = _add_axes_initializer(builder, "pool_axis", axes)
        _ = builder.add_node(
            "ReduceMax",
            [builder.current_tensor, axes_name],
            "pool_axis_reduce",
            keepdims=0,
        )
        if builder.current_rank is not None:
            builder.current_rank -= len(axes)
        return

    size_value = _require_attr(layer, "size", label)
    size_value = normalize_pool_pair_value(size_value)
    size = _as_int_pair(size_value, "%s.size" % label)

    if stride_value is None:
        stride = size
    else:
        stride_value = normalize_pool_pair_value(stride_value)
        stride = _as_int_pair(stride_value, "%s.stride" % label)

    channel_added = _prepare_spatial_input_for_nchw(builder, label, "pool")
    _ = builder.add_node(
        "MaxPool",
        [builder.current_tensor],
        "max_pool",
        kernel_shape=[size[0], size[1]],
        strides=[stride[0], stride[1]],
    )
    _restore_spatial_output_from_nchw(
        builder,
        channel_added=channel_added,
        what="pool",
        remove_added_channel=True,
    )


def _convert_tcn_block(
    builder: _PrimitiveGraphBuilder,
    layer: object,
    label: str,
    layer_index: int,
    squeeze_outputs: bool = True,
) -> TensorReferences:
    dilated_conv = _require_attr(layer, "dilated_conv", label)
    dilation_rate = _require_attr(layer, "dilation_rate", label)

    if isinstance(dilated_conv, list):
        convs = dilated_conv
        if isinstance(dilation_rate, (list, tuple)):
            rates = list(dilation_rate)
        else:
            rates = [dilation_rate] * len(convs)
    else:
        convs = [dilated_conv]
        rates = [dilation_rate]

    if len(rates) != len(convs):
        raise UnsupportedLayoutError(
            "%s dilation_rate length (%d) must match dilated_conv length (%d)"
            % (label, len(rates), len(convs))
        )

    original_input_tensor = builder.current_tensor
    original_input_rank = builder.current_rank

    if builder.current_rank == 2:
        _append_unsqueeze(builder, [1], "tcn_block_input_expand")
        builder.current_rank = 3
        channel_added = True
    elif builder.current_rank == 3:
        channel_added = False
    else:
        raise UnsupportedLayoutError(
            "%s requires rank-2 or rank-3 input in [time, freq, channel] layout" % label
        )

    res_path_base = builder.current_tensor
    res_rank_base = builder.current_rank

    _append_unsqueeze(builder, [0], "tcn_block_add_batch")
    _ = builder.add_node(
        "Transpose",
        [builder.current_tensor],
        "tcn_block_to_nchw",
        perm=[0, 3, 1, 2],
    )
    builder.current_rank = 4
    nchw_input_tensor = builder.current_tensor

    conv_outputs = []
    for i, (conv, rate) in enumerate(zip(convs, rates)):
        rate_int = _as_positive_int(rate, "%s.dilation_rate[%d]" % (label, i))
        weights = _as_float32_array(
            _require_attr(conv, "weights", "%s.conv[%d]" % (label, i)),
            "%s.weights" % label,
        )
        if weights.ndim != 4:
            raise UnsupportedLayoutError(
                "%s.conv[%d] weights must be rank-4" % (label, i)
            )

        # Weights in madmom TCN are (C_in, C_out, 1, K)
        # Transpose to (C_out, C_in, K, 1) for Conv along H (time) in NCHW
        transformed_weights = weights.transpose(1, 0, 3, 2)
        # flip along K axis (axis 2) to match scipy.ndimage.convolve
        transformed_weights = np.flip(transformed_weights, axis=2)

        bias = _as_float32_array(
            _require_attr(conv, "bias", "%s.conv[%d]" % (label, i)), "%s.bias" % label
        )
        bias = bias.reshape(-1)

        weights_name = builder.add_initializer(
            "tcn_conv_weights_%d_%d" % (layer_index, i), transformed_weights
        )
        bias_name = builder.add_initializer(
            "tcn_conv_bias_%d_%d" % (layer_index, i), bias
        )

        kernel_size = transformed_weights.shape[2]
        left_pad = 2 * rate_int
        right_pad = max((kernel_size - 3) * rate_int, 0)

        conv_output = builder.add_node(
            "Conv",
            [nchw_input_tensor, weights_name, bias_name],
            "tcn_conv_%d_%d" % (layer_index, i),
            dilations=[rate_int, 1],
            pads=[left_pad, 0, right_pad, 0],
        )
        _append_activation(builder, conv, label + ".conv[%d]" % i)
        conv_outputs.append(builder.current_tensor)

    if len(conv_outputs) > 1:
        out = builder.add_node(
            "Concat", conv_outputs, "tcn_conv_concat_%d" % layer_index, axis=1
        )
    else:
        out = conv_outputs[0]

    builder.current_tensor = out
    _restore_spatial_output_from_nchw(
        builder,
        channel_added=channel_added,
        what="tcn_block_out",
        remove_added_channel=False,
    )
    builder.current_rank = 3

    _append_activation(builder, layer, label)
    activated_out = builder.current_tensor

    def convert_tcn_branch_projection(branch_layer: object, branch_label: str) -> None:
        branch_weights = _as_float32_array(
            _require_attr(branch_layer, "weights", branch_label),
            "%s.weights" % branch_label,
        )
        if branch_weights.ndim == 4:
            _convert_convolutional_layer(builder, branch_layer, branch_label)
            return
        if branch_weights.ndim in (1, 2):
            _convert_feed_forward_layer(builder, branch_layer, branch_label)
            return
        raise UnsupportedLayoutError(
            "%s weights must have rank 4 [channel, feature, time, freq] or rank-1/rank-2 feed-forward projection"
            % branch_label
        )

    skip_conv = getattr(layer, "skip_conv", None)
    if skip_conv is not None:
        convert_tcn_branch_projection(skip_conv, label + ".skip_conv")

    final_skip_out = builder.current_tensor
    final_skip_rank = builder.current_rank

    residual_conv = getattr(layer, "residual_conv", None)
    if residual_conv is not None:
        builder.current_tensor = res_path_base
        builder.current_rank = res_rank_base
        convert_tcn_branch_projection(residual_conv, label + ".residual_conv")
        res_path = builder.current_tensor
    else:
        res_path = res_path_base

    final_out_res = builder.add_node(
        "Add", [res_path, final_skip_out], "tcn_residual_add_%d" % layer_index
    )

    def _squeeze_to_rank2(
        tensor_name: str, tensor_rank: int | None, stem: str
    ) -> TensorReference:
        if tensor_rank == 3:
            return _squeeze_tensor(builder, tensor_name, 3, [1], stem), 2
        return tensor_name, tensor_rank

    if squeeze_outputs:
        res_ref = _squeeze_to_rank2(final_out_res, 3, "tcn_block_res_squeeze")
        skip_ref = _squeeze_to_rank2(
            final_skip_out, final_skip_rank, "tcn_block_skip_squeeze"
        )
    else:
        res_ref = (final_out_res, 3)
        skip_ref = (final_skip_out, final_skip_rank)

    return [res_ref, skip_ref]


def _convert_tcn_layer(
    builder: _PrimitiveGraphBuilder,
    layer: object,
    label: str,
    input_values: TensorReferences,
    recurrent_state_interfaces: list[tuple[str, str, np.ndarray]],
    layer_counter: list[int],
) -> TensorReferences:
    tcn_blocks = _require_layer_list(layer, label, "tcn_blocks")
    skip_connections = bool(getattr(layer, "skip_connections", False))

    current_data = input_values[0]
    all_skips = []

    for i, block in enumerate(tcn_blocks):
        _set_builder_input(builder, current_data)
        block_outputs = _convert_tcn_block(
            builder,
            block,
            label + ".block[%d]" % i,
            layer_counter[0],
            squeeze_outputs=False,
        )
        layer_counter[0] += 1
        current_data = block_outputs[0]
        all_skips.append(block_outputs[1])

    if len(all_skips) > 1:
        current_skip = all_skips[0][0]
        for i in range(1, len(all_skips)):
            current_skip = builder.add_node(
                "Add",
                [current_skip, all_skips[i][0]],
                "tcn_skip_sum_%d_%d" % (layer_counter[0], i),
            )
        skip_total = (current_skip, all_skips[0][1])
    else:
        skip_total = all_skips[0]

    _set_builder_input(builder, current_data)
    _append_activation(builder, layer, label)
    current_data = (builder.current_tensor, builder.current_rank)

    if skip_connections:
        return [current_data, skip_total]
    else:
        return [current_data]


def _convert_stride_layer(
    builder: _PrimitiveGraphBuilder,
    layer: object,
    label: str,
) -> None:
    block_size_value = _require_attr(layer, "block_size", label)
    if not isinstance(block_size_value, (bool, int)):
        block_size_array = np.asarray(block_size_value)
        if block_size_array.ndim == 0:
            block_size_value = block_size_array.item()
    block_size = _as_positive_int(block_size_value, "%s.block_size" % label)
    if builder.current_rank is not None and builder.current_rank < 1:
        raise UnsupportedLayoutError("%s requires rank >= 1 input" % label)

    sliced_segments: list[str] = []
    source_tensor = builder.current_tensor
    max_end = np.asarray(np.iinfo(np.int64).max, dtype=np.int64)
    axis_zero = builder.add_initializer("stride_axis", np.asarray([0], dtype=np.int64))
    unit_step = builder.add_initializer("stride_step", np.asarray([1], dtype=np.int64))
    unsqueeze_axis = _add_axes_initializer(builder, "stride_unsqueeze_axis", [1])

    for offset in range(block_size):
        starts_name = builder.add_initializer(
            "stride_starts", np.asarray([offset], dtype=np.int64)
        )
        if offset == block_size - 1:
            ends_array = np.asarray([max_end.item()], dtype=np.int64)
        else:
            ends_array = np.asarray([-(block_size - offset - 1)], dtype=np.int64)
        ends_name = builder.add_initializer("stride_ends", ends_array)
        sliced = builder.add_node(
            "Slice",
            [source_tensor, starts_name, ends_name, axis_zero, unit_step],
            "stride_slice",
        )
        segment = builder.add_node(
            "Unsqueeze",
            [sliced, unsqueeze_axis],
            "stride_segment",
        )
        sliced_segments.append(segment)

    stacked = builder.add_node("Concat", sliced_segments, "stride_concat", axis=1)
    shape = builder.add_node("Shape", [stacked], "stride_shape")
    first_dim_index = builder.add_initializer(
        "stride_first_dim", np.asarray([0], dtype=np.int64)
    )
    first_dim = builder.add_node(
        "Gather",
        [shape, first_dim_index],
        "stride_first_dim_value",
        axis=0,
    )
    minus_one = builder.add_initializer(
        "stride_minus_one", np.asarray([-1], dtype=np.int64)
    )
    target_shape = builder.add_node(
        "Concat",
        [first_dim, minus_one],
        "stride_target_shape",
        axis=0,
    )
    _ = builder.add_node("Reshape", [stacked, target_shape], "stride_flatten")
    builder.current_rank = 2


def _convert_recurrent_layer(
    builder: _PrimitiveGraphBuilder,
    layer: object,
    label: str,
    layer_index: int,
    recurrent_state_interfaces: list[tuple[str, str, np.ndarray]],
) -> None:
    weights = _as_float32_array(
        _require_attr(layer, "weights", label), "%s.weights" % label
    )
    recurrent_weights = _as_float32_array(
        _require_attr(layer, "recurrent_weights", label), "%s.recurrent_weights" % label
    )
    bias = _as_1d_float32_array(_require_attr(layer, "bias", label), "%s.bias" % label)
    init = _as_1d_float32_array(_require_attr(layer, "init", label), "%s.init" % label)

    if weights.ndim != 2:
        raise UnsupportedLayoutError(
            "%s.weights must be rank-2; got rank %d" % (label, weights.ndim)
        )
    input_size, hidden_size = weights.shape
    if recurrent_weights.shape != (hidden_size, hidden_size):
        raise UnsupportedLayoutError(
            "%s.recurrent_weights must have shape (%d, %d); got %s"
            % (label, hidden_size, hidden_size, tuple(recurrent_weights.shape))
        )
    if bias.size != hidden_size:
        raise UnsupportedLayoutError(
            "%s.bias size mismatch: expected %d, got %d"
            % (label, hidden_size, bias.size)
        )
    _validate_state_size(init, hidden_size, "%s.init" % label)

    activation_name = _onnx_recurrent_activation_name(
        _require_attr(layer, "activation_fn", label),
        label,
        allowed={"tanh", "sigmoid", "relu"},
    )

    x_name = _unsqueeze_tensor(
        builder,
        builder.current_tensor,
        builder.current_rank,
        [1],
        "recurrent_input_batch_%d" % layer_index,
    )
    state_input_name = "state_%d_hidden_in" % layer_index
    initial_h_name = _unsqueeze_tensor(
        builder,
        state_input_name,
        1,
        [0, 1],
        "recurrent_state_batch_%d" % layer_index,
    )

    w_name = builder.add_initializer(
        "recurrent_w_%d" % layer_index,
        np.expand_dims(weights.T, axis=0),
    )
    r_name = builder.add_initializer(
        "recurrent_r_%d" % layer_index,
        np.expand_dims(recurrent_weights.T, axis=0),
    )
    b_name = builder.add_initializer(
        "recurrent_b_%d" % layer_index,
        np.expand_dims(
            np.concatenate(
                [
                    bias,
                    np.zeros(hidden_size, dtype=np.float32),
                ]
            ),
            axis=0,
        ),
    )

    sequence_output_name = builder._name("recurrent_sequence_%d" % layer_index)
    hidden_output_name = builder._name("recurrent_hidden_%d" % layer_index)
    _append_multi_output_node(
        builder,
        "RNN",
        [x_name, w_name, r_name, b_name, "", initial_h_name],
        [sequence_output_name, hidden_output_name],
        "recurrent_node_%d" % layer_index,
        activations=[activation_name],
        hidden_size=hidden_size,
    )

    builder.current_tensor = _squeeze_tensor(
        builder,
        sequence_output_name,
        4,
        [1, 2],
        "recurrent_sequence_squeeze_%d" % layer_index,
    )
    builder.current_rank = 2

    state_output_name = _squeeze_tensor(
        builder,
        hidden_output_name,
        3,
        [0, 1],
        "recurrent_state_squeeze_%d" % layer_index,
    )
    recurrent_state_interfaces.append(
        (state_input_name, state_output_name, np.asarray(init, dtype=np.float32))
    )


def _convert_lstm_layer(
    builder: _PrimitiveGraphBuilder,
    layer: object,
    label: str,
    layer_index: int,
    recurrent_state_interfaces: list[tuple[str, str, np.ndarray]],
) -> None:
    input_gate = _require_attr(layer, "input_gate", label)
    forget_gate = _require_attr(layer, "forget_gate", label)
    output_gate = _require_attr(layer, "output_gate", label)
    cell = _require_attr(layer, "cell", label)

    input_weights = _as_float32_array(
        _require_attr(input_gate, "weights", "%s.input_gate" % label),
        "%s.input_gate.weights" % label,
    )
    if input_weights.ndim != 2:
        raise UnsupportedLayoutError(
            "%s.input_gate.weights must be rank-2; got rank %d"
            % (label, input_weights.ndim)
        )
    input_size, hidden_size = input_weights.shape

    input_params = _validate_gate_shapes(
        input_gate,
        gate_name="input_gate",
        label=label,
        input_size=input_size,
        hidden_size=hidden_size,
        allow_peephole=True,
    )
    forget_params = _validate_gate_shapes(
        forget_gate,
        gate_name="forget_gate",
        label=label,
        input_size=input_size,
        hidden_size=hidden_size,
        allow_peephole=True,
    )
    cell_params = _validate_gate_shapes(
        cell,
        gate_name="cell",
        label=label,
        input_size=input_size,
        hidden_size=hidden_size,
        allow_peephole=False,
    )
    output_params = _validate_gate_shapes(
        output_gate,
        gate_name="output_gate",
        label=label,
        input_size=input_size,
        hidden_size=hidden_size,
        allow_peephole=True,
    )

    input_activation = _onnx_recurrent_activation_name(
        input_params[4],
        "%s.input_gate" % label,
        allowed={"sigmoid", "tanh", "relu"},
    )
    forget_activation = _onnx_recurrent_activation_name(
        forget_params[4],
        "%s.forget_gate" % label,
        allowed={"sigmoid", "tanh", "relu"},
    )
    output_activation = _onnx_recurrent_activation_name(
        output_params[4],
        "%s.output_gate" % label,
        allowed={"sigmoid", "tanh", "relu"},
    )
    if input_activation != forget_activation or input_activation != output_activation:
        raise UnsupportedActivationError(
            "%s requires identical input/forget/output gate activations; got %s, %s, %s"
            % (label, input_activation, forget_activation, output_activation)
        )

    cell_activation = _onnx_recurrent_activation_name(
        cell_params[4],
        "%s.cell" % label,
        allowed={"tanh", "sigmoid", "relu"},
    )
    state_activation = _onnx_recurrent_activation_name(
        _require_attr(layer, "activation_fn", label),
        label,
        allowed={"tanh", "sigmoid", "relu"},
    )

    init = _as_1d_float32_array(_require_attr(layer, "init", label), "%s.init" % label)
    cell_init = _as_1d_float32_array(
        _require_attr(layer, "cell_init", label), "%s.cell_init" % label
    )
    _validate_state_size(init, hidden_size, "%s.init" % label)
    _validate_state_size(cell_init, hidden_size, "%s.cell_init" % label)

    x_name = _unsqueeze_tensor(
        builder,
        builder.current_tensor,
        builder.current_rank,
        [1],
        "lstm_input_batch_%d" % layer_index,
    )
    hidden_state_input_name = "state_%d_hidden_in" % layer_index
    cell_state_input_name = "state_%d_cell_in" % layer_index
    initial_h_name = _unsqueeze_tensor(
        builder,
        hidden_state_input_name,
        1,
        [0, 1],
        "lstm_hidden_batch_%d" % layer_index,
    )
    initial_c_name = _unsqueeze_tensor(
        builder,
        cell_state_input_name,
        1,
        [0, 1],
        "lstm_cell_batch_%d" % layer_index,
    )

    gate_weights = [
        input_params[0].T,
        output_params[0].T,
        forget_params[0].T,
        cell_params[0].T,
    ]
    recurrent_gate_weights = [
        input_params[1].T,
        output_params[1].T,
        forget_params[1].T,
        cell_params[1].T,
    ]
    gate_biases = [
        input_params[2],
        output_params[2],
        forget_params[2],
        cell_params[2],
    ]

    w_name = builder.add_initializer(
        "lstm_w_%d" % layer_index,
        np.expand_dims(np.concatenate(gate_weights, axis=0), axis=0),
    )
    r_name = builder.add_initializer(
        "lstm_r_%d" % layer_index,
        np.expand_dims(np.concatenate(recurrent_gate_weights, axis=0), axis=0),
    )
    b_name = builder.add_initializer(
        "lstm_b_%d" % layer_index,
        np.expand_dims(
            np.concatenate(
                [
                    np.concatenate(gate_biases, axis=0),
                    np.zeros(hidden_size * 4, dtype=np.float32),
                ]
            ),
            axis=0,
        ),
    )

    peepholes = [input_params[3], output_params[3], forget_params[3]]
    has_peepholes = any(item is not None for item in peepholes)
    recurrent_inputs = [
        x_name,
        w_name,
        r_name,
        b_name,
        "",
        initial_h_name,
        initial_c_name,
    ]
    if has_peepholes:
        p_values: list[np.ndarray] = []
        for item in peepholes:
            if item is None:
                p_values.append(np.zeros(hidden_size, dtype=np.float32))
            else:
                p_values.append(item)
        p_name = builder.add_initializer(
            "lstm_p_%d" % layer_index,
            np.expand_dims(np.concatenate(p_values, axis=0), axis=0),
        )
        recurrent_inputs.append(p_name)

    sequence_output_name = builder._name("lstm_sequence_%d" % layer_index)
    hidden_output_name = builder._name("lstm_hidden_%d" % layer_index)
    cell_output_name = builder._name("lstm_cell_%d" % layer_index)
    _append_multi_output_node(
        builder,
        "LSTM",
        recurrent_inputs,
        [sequence_output_name, hidden_output_name, cell_output_name],
        "lstm_node_%d" % layer_index,
        activations=[input_activation, cell_activation, state_activation],
        hidden_size=hidden_size,
    )

    builder.current_tensor = _squeeze_tensor(
        builder,
        sequence_output_name,
        4,
        [1, 2],
        "lstm_sequence_squeeze_%d" % layer_index,
    )
    builder.current_rank = 2

    hidden_state_output = _squeeze_tensor(
        builder,
        hidden_output_name,
        3,
        [0, 1],
        "lstm_hidden_squeeze_%d" % layer_index,
    )
    cell_state_output = _squeeze_tensor(
        builder,
        cell_output_name,
        3,
        [0, 1],
        "lstm_cell_squeeze_%d" % layer_index,
    )

    recurrent_state_interfaces.append(
        (
            hidden_state_input_name,
            hidden_state_output,
            np.asarray(init, dtype=np.float32),
        )
    )
    recurrent_state_interfaces.append(
        (
            cell_state_input_name,
            cell_state_output,
            np.asarray(cell_init, dtype=np.float32),
        )
    )


def _convert_gru_layer(
    builder: _PrimitiveGraphBuilder,
    layer: object,
    label: str,
    layer_index: int,
    recurrent_state_interfaces: list[tuple[str, str, np.ndarray]],
) -> None:
    reset_gate = _require_attr(layer, "reset_gate", label)
    update_gate = _require_attr(layer, "update_gate", label)
    cell = _require_attr(layer, "cell", label)

    reset_weights = _as_float32_array(
        _require_attr(reset_gate, "weights", "%s.reset_gate" % label),
        "%s.reset_gate.weights" % label,
    )
    if reset_weights.ndim != 2:
        raise UnsupportedLayoutError(
            "%s.reset_gate.weights must be rank-2; got rank %d"
            % (label, reset_weights.ndim)
        )
    input_size, hidden_size = reset_weights.shape

    reset_params = _validate_gate_shapes(
        reset_gate,
        gate_name="reset_gate",
        label=label,
        input_size=input_size,
        hidden_size=hidden_size,
        allow_peephole=False,
    )
    update_params = _validate_gate_shapes(
        update_gate,
        gate_name="update_gate",
        label=label,
        input_size=input_size,
        hidden_size=hidden_size,
        allow_peephole=False,
    )
    cell_params = _validate_gate_shapes(
        cell,
        gate_name="cell",
        label=label,
        input_size=input_size,
        hidden_size=hidden_size,
        allow_peephole=False,
    )

    gate_activation_reset = _onnx_recurrent_activation_name(
        reset_params[4],
        "%s.reset_gate" % label,
        allowed={"sigmoid", "tanh", "relu"},
    )
    gate_activation_update = _onnx_recurrent_activation_name(
        update_params[4],
        "%s.update_gate" % label,
        allowed={"sigmoid", "tanh", "relu"},
    )
    if gate_activation_reset != "Sigmoid" or gate_activation_update != "Sigmoid":
        raise UnsupportedActivationError(
            "%s requires Sigmoid reset/update gate activations; got %s and %s"
            % (label, gate_activation_reset, gate_activation_update)
        )

    candidate_activation = _onnx_recurrent_activation_name(
        cell_params[4],
        "%s.cell" % label,
        allowed={"tanh", "sigmoid", "relu"},
    )

    init = _as_1d_float32_array(_require_attr(layer, "init", label), "%s.init" % label)
    _validate_state_size(init, hidden_size, "%s.init" % label)

    x_name = _unsqueeze_tensor(
        builder,
        builder.current_tensor,
        builder.current_rank,
        [1],
        "gru_input_batch_%d" % layer_index,
    )
    state_input_name = "state_%d_hidden_in" % layer_index
    initial_h_name = _unsqueeze_tensor(
        builder,
        state_input_name,
        1,
        [0, 1],
        "gru_state_batch_%d" % layer_index,
    )

    z_input_weights = -update_params[0].T
    z_recurrent_weights = -update_params[1].T
    z_bias = -update_params[2]

    w_name = builder.add_initializer(
        "gru_w_%d" % layer_index,
        np.expand_dims(
            np.concatenate(
                [
                    z_input_weights,
                    reset_params[0].T,
                    cell_params[0].T,
                ],
                axis=0,
            ),
            axis=0,
        ),
    )
    r_name = builder.add_initializer(
        "gru_r_%d" % layer_index,
        np.expand_dims(
            np.concatenate(
                [
                    z_recurrent_weights,
                    reset_params[1].T,
                    cell_params[1].T,
                ],
                axis=0,
            ),
            axis=0,
        ),
    )
    b_name = builder.add_initializer(
        "gru_b_%d" % layer_index,
        np.expand_dims(
            np.concatenate(
                [
                    np.concatenate(
                        [
                            z_bias,
                            reset_params[2],
                            cell_params[2],
                        ],
                        axis=0,
                    ),
                    np.zeros(hidden_size * 3, dtype=np.float32),
                ]
            ),
            axis=0,
        ),
    )

    sequence_output_name = builder._name("gru_sequence_%d" % layer_index)
    hidden_output_name = builder._name("gru_hidden_%d" % layer_index)
    _append_multi_output_node(
        builder,
        "GRU",
        [x_name, w_name, r_name, b_name, "", initial_h_name],
        [sequence_output_name, hidden_output_name],
        "gru_node_%d" % layer_index,
        activations=[gate_activation_reset, candidate_activation],
        hidden_size=hidden_size,
        linear_before_reset=1,
    )

    builder.current_tensor = _squeeze_tensor(
        builder,
        sequence_output_name,
        4,
        [1, 2],
        "gru_sequence_squeeze_%d" % layer_index,
    )
    builder.current_rank = 2

    state_output_name = _squeeze_tensor(
        builder,
        hidden_output_name,
        3,
        [0, 1],
        "gru_state_squeeze_%d" % layer_index,
    )
    recurrent_state_interfaces.append(
        (state_input_name, state_output_name, np.asarray(init, dtype=np.float32))
    )


def _append_output_squeeze(
    builder: _PrimitiveGraphBuilder,
    tensor_name: str,
    tensor_rank: int | None,
    stem: str,
) -> TensorReference:
    previous_tensor = builder.current_tensor
    previous_rank = builder.current_rank
    builder.current_tensor = tensor_name
    builder.current_rank = tensor_rank
    output_name = builder.add_node("Squeeze", [builder.current_tensor], stem)
    builder.current_rank = None
    builder.current_tensor = previous_tensor
    builder.current_rank = previous_rank
    return output_name, None


def _symbolic_shape(rank: int, prefix: str) -> list[str]:
    if rank < 0:
        raise ValueError("rank must be >= 0")
    return ["%s_dim_%d" % (prefix, index) for index in range(rank)]


def _checker_compatible_shape(rank: int | None, prefix: str) -> list[str]:
    if rank is None:
        return ["%s_dim_0" % prefix]
    return _symbolic_shape(rank, prefix)


def _require_single_tensor(
    values: TensorReferences,
    label: str,
    what: str,
) -> TensorReference:
    if len(values) != 1:
        raise UnsupportedLayoutError(
            "%s requires a single input tensor for %s; got %d tensors"
            % (label, what, len(values))
        )
    return values[0]


def _set_builder_input(
    builder: _PrimitiveGraphBuilder,
    tensor: TensorReference,
) -> None:
    builder.current_tensor = tensor[0]
    builder.current_rank = tensor[1]


def _require_layer_list(layer: object, label: str, attr_name: str) -> list[object]:
    value = _require_attr(layer, attr_name, label)
    if not isinstance(value, (list, tuple)):
        raise UnsupportedLayoutError(
            "%s.%s must be a list or tuple of layers" % (label, attr_name)
        )
    return list(value)


def _reverse_tensor_along_axis(
    builder: _PrimitiveGraphBuilder,
    tensor_name: str,
    tensor_rank: int | None,
    axis: int,
    stem: str,
) -> TensorReference:
    if tensor_rank is not None and (axis < 0 or axis >= tensor_rank):
        raise UnsupportedLayoutError(
            "cannot reverse axis %d for rank %d tensor" % (axis, tensor_rank)
        )

    starts_name = builder.add_initializer(
        "%s_starts" % stem,
        np.asarray([-1], dtype=np.int64),
    )
    ends_name = builder.add_initializer(
        "%s_ends" % stem,
        np.asarray([np.iinfo(np.int64).min], dtype=np.int64),
    )
    axes_name = builder.add_initializer(
        "%s_axes" % stem,
        np.asarray([axis], dtype=np.int64),
    )
    steps_name = builder.add_initializer(
        "%s_steps" % stem,
        np.asarray([-1], dtype=np.int64),
    )

    previous_tensor = builder.current_tensor
    previous_rank = builder.current_rank
    builder.current_tensor = tensor_name
    builder.current_rank = tensor_rank
    output_name = builder.add_node(
        "Slice",
        [tensor_name, starts_name, ends_name, axes_name, steps_name],
        stem,
    )
    builder.current_tensor = previous_tensor
    builder.current_rank = previous_rank
    return output_name, tensor_rank


def _convert_layer_graph(
    builder: _PrimitiveGraphBuilder,
    layer: object,
    input_values: TensorReferences,
    recurrent_state_interfaces: list[tuple[str, str, np.ndarray]],
    layer_counter: list[int],
) -> TensorReferences:
    layer_index = layer_counter[0]
    layer_counter[0] += 1
    label = _layer_label(layer_index, layer)
    layer_type = layer.__class__.__name__

    if layer_type == "SequentialLayer":
        layers = _require_layer_list(layer, label, "layers")
        current_values = list(input_values)
        for sub_layer in layers:
            current_values = _convert_layer_graph(
                builder,
                sub_layer,
                current_values,
                recurrent_state_interfaces,
                layer_counter,
            )
        return current_values

    if layer_type == "ParallelLayer":
        layers = _require_layer_list(layer, label, "layers")
        outputs: TensorReferences = []
        for branch_index, sub_layer in enumerate(layers):
            branch_outputs = _convert_layer_graph(
                builder,
                sub_layer,
                list(input_values),
                recurrent_state_interfaces,
                layer_counter,
            )
            if len(branch_outputs) != 1:
                raise UnsupportedLayoutError(
                    "%s branch %d must produce exactly one tensor; got %d"
                    % (label, branch_index, len(branch_outputs))
                )
            outputs.append(branch_outputs[0])
        return outputs

    if layer_type == "MultiTaskLayer":
        layers = _require_layer_list(layer, label, "layers")
        mapping_value = _require_attr(layer, "mapping", label)
        if mapping_value is None:
            mapping = None
        elif isinstance(mapping_value, dict):
            mapping = mapping_value
        else:
            raise UnsupportedLayoutError("%s.mapping must be a dict or None" % label)

        outputs = []
        for task_index, sub_layer in enumerate(layers):
            if mapping is None:
                input_index = task_index
            else:
                try:
                    mapping_index = mapping[task_index]
                except KeyError as error:
                    raise UnsupportedLayoutError(
                        "%s.mapping is missing task index %d" % (label, task_index)
                    ) from error
                input_index = _as_int(
                    mapping_index,
                    "%s.mapping[%d]" % (label, task_index),
                )

            if input_index < 0 or input_index >= len(input_values):
                raise UnsupportedLayoutError(
                    "%s task %d selects input index %d, but only %d inputs are available"
                    % (label, task_index, input_index, len(input_values))
                )

            task_outputs = _convert_layer_graph(
                builder,
                sub_layer,
                [input_values[input_index]],
                recurrent_state_interfaces,
                layer_counter,
            )
            if len(task_outputs) != 1:
                raise UnsupportedLayoutError(
                    "%s task %d must produce exactly one tensor; got %d"
                    % (label, task_index, len(task_outputs))
                )
            outputs.append(task_outputs[0])
        return outputs

    if layer_type == "BidirectionalLayer":
        source_tensor_name, source_rank = _require_single_tensor(
            input_values,
            label,
            "bidirectional input",
        )

        fwd_layer = _require_attr(layer, "fwd_layer", label)
        bwd_layer = _require_attr(layer, "bwd_layer", label)

        fwd_outputs = _convert_layer_graph(
            builder,
            fwd_layer,
            [(source_tensor_name, source_rank)],
            recurrent_state_interfaces,
            layer_counter,
        )
        if len(fwd_outputs) != 1:
            raise UnsupportedLayoutError(
                "%s.fwd_layer must produce exactly one tensor; got %d"
                % (label, len(fwd_outputs))
            )

        reversed_input_name, reversed_input_rank = _reverse_tensor_along_axis(
            builder,
            source_tensor_name,
            source_rank,
            axis=0,
            stem="bidirectional_input_reverse_%d" % layer_index,
        )
        bwd_outputs = _convert_layer_graph(
            builder,
            bwd_layer,
            [(reversed_input_name, reversed_input_rank)],
            recurrent_state_interfaces,
            layer_counter,
        )
        if len(bwd_outputs) != 1:
            raise UnsupportedLayoutError(
                "%s.bwd_layer must produce exactly one tensor; got %d"
                % (label, len(bwd_outputs))
            )

        bwd_name, bwd_rank = bwd_outputs[0]
        fwd_name, fwd_rank = fwd_outputs[0]
        if fwd_rank is not None and fwd_rank < 2:
            raise UnsupportedLayoutError(
                "%s output rank must be >= 2 for bidirectional concatenation" % label
            )
        if fwd_rank is not None and bwd_rank is not None and fwd_rank != bwd_rank:
            raise UnsupportedLayoutError(
                "%s forward/backward rank mismatch: %d vs %d"
                % (label, fwd_rank, bwd_rank)
            )

        reversed_bwd_name, _ = _reverse_tensor_along_axis(
            builder,
            bwd_name,
            bwd_rank,
            axis=0,
            stem="bidirectional_output_reverse_%d" % layer_index,
        )

        previous_tensor = builder.current_tensor
        previous_rank = builder.current_rank
        builder.current_tensor = fwd_name
        builder.current_rank = fwd_rank
        output_name = builder.add_node(
            "Concat",
            [fwd_name, reversed_bwd_name],
            "bidirectional_concat_%d" % layer_index,
            axis=1,
        )
        output_rank = builder.current_rank
        builder.current_tensor = previous_tensor
        builder.current_rank = previous_rank
        return [(output_name, output_rank)]

    tensor_input = _require_single_tensor(input_values, label, layer_type)
    _set_builder_input(builder, tensor_input)

    if layer_type == "FeedForwardLayer":
        _convert_feed_forward_layer(builder, layer, label)
        return [(builder.current_tensor, builder.current_rank)]
    if layer_type == "BatchNormLayer":
        _convert_batch_norm_layer(builder, layer, label)
        return [(builder.current_tensor, builder.current_rank)]
    if layer_type == "ReshapeLayer":
        _convert_reshape_layer(builder, layer, label)
        return [(builder.current_tensor, builder.current_rank)]
    if layer_type == "TransposeLayer":
        _convert_transpose_layer(builder, layer, label)
        return [(builder.current_tensor, builder.current_rank)]
    if layer_type == "PadLayer":
        _convert_pad_layer(builder, layer, label)
        return [(builder.current_tensor, builder.current_rank)]
    if layer_type == "AverageLayer":
        _convert_average_layer(builder, layer, label)
        return [(builder.current_tensor, builder.current_rank)]
    if layer_type == "ConvolutionalLayer":
        _convert_convolutional_layer(builder, layer, label)
        return [(builder.current_tensor, builder.current_rank)]
    if layer_type == "MaxPoolLayer":
        _convert_max_pool_layer(builder, layer, label)
        return [(builder.current_tensor, builder.current_rank)]
    if layer_type == "StrideLayer":
        _convert_stride_layer(builder, layer, label)
        return [(builder.current_tensor, builder.current_rank)]
    if layer_type == "RecurrentLayer":
        _convert_recurrent_layer(
            builder,
            layer,
            label,
            layer_index,
            recurrent_state_interfaces,
        )
        return [(builder.current_tensor, builder.current_rank)]
    if layer_type == "LSTMLayer":
        _convert_lstm_layer(
            builder,
            layer,
            label,
            layer_index,
            recurrent_state_interfaces,
        )
        return [(builder.current_tensor, builder.current_rank)]
    if layer_type == "GRULayer":
        _convert_gru_layer(
            builder,
            layer,
            label,
            layer_index,
            recurrent_state_interfaces,
        )
        return [(builder.current_tensor, builder.current_rank)]
    if layer_type == "TCNBlock":
        return _convert_tcn_block(builder, layer, label, layer_index)
    if layer_type == "TCNLayer":
        return _convert_tcn_layer(
            builder,
            layer,
            label,
            input_values,
            recurrent_state_interfaces,
            layer_counter,
        )

    raise NotImplementedError(
        "layer %s is not implemented; converter supports structural layers "
        "SequentialLayer, ParallelLayer, MultiTaskLayer, BidirectionalLayer "
        "and primitive layers FeedForwardLayer, BatchNormLayer, ReshapeLayer, "
        "TransposeLayer, PadLayer, AverageLayer, ConvolutionalLayer, "
        "MaxPoolLayer, StrideLayer, RecurrentLayer, LSTMLayer, GRULayer, "
        "TCNBlock, and TCNLayer" % layer_type
    )


def _build_onnx_model_from_primitive_layers(
    layers: Sequence[object],
    opset: int,
) -> object:
    if opset < 13:
        raise NotImplementedError("primitive conversion requires opset >= 13")

    onnx_module = _import_onnx_for_conversion()
    builder = _PrimitiveGraphBuilder(onnx_module=onnx_module, opset=opset)
    input_rank = _infer_graph_input_rank(layers)
    builder.current_rank = input_rank
    recurrent_state_interfaces: list[tuple[str, str, np.ndarray]] = []

    layer_counter = [0]
    outputs: TensorReferences = [("input", input_rank)]
    for layer in layers:
        outputs = _convert_layer_graph(
            builder,
            layer,
            outputs,
            recurrent_state_interfaces,
            layer_counter,
        )

    if not outputs:
        raise UnsupportedLayoutError("network produced no outputs")

    input_info = onnx_module.helper.make_tensor_value_info(
        "input",
        onnx_module.TensorProto.FLOAT,
        _symbolic_shape(input_rank, "input"),
    )

    graph_inputs = [input_info]
    graph_outputs = []
    for output_index, (output_name, output_rank) in enumerate(outputs):
        final_name, final_rank = _append_output_squeeze(
            builder, output_name, output_rank, "output_%d_squeeze" % output_index
        )
        graph_outputs.append(
            onnx_module.helper.make_tensor_value_info(
                final_name,
                onnx_module.TensorProto.FLOAT,
                _checker_compatible_shape(final_rank, "output_%d" % output_index),
            )
        )
    for state_input_name, state_output_name, state_init in recurrent_state_interfaces:
        if state_init.ndim != 1:
            raise UnsupportedLayoutError(
                "state initializer for '%s' must be rank-1; got %s"
                % (state_input_name, tuple(state_init.shape))
            )
        hidden_size = int(state_init.size)
        graph_inputs.append(
            onnx_module.helper.make_tensor_value_info(
                state_input_name,
                onnx_module.TensorProto.FLOAT,
                [hidden_size],
            )
        )
        graph_outputs.append(
            onnx_module.helper.make_tensor_value_info(
                state_output_name,
                onnx_module.TensorProto.FLOAT,
                [hidden_size],
            )
        )

    graph = onnx_module.helper.make_graph(
        builder.nodes,
        "madmom_primitive_network",
        graph_inputs,
        graph_outputs,
        initializer=builder.initializers,
    )

    make_opsetid = getattr(onnx_module.helper, "make_opsetid", None)
    if make_opsetid is None:
        make_opsetid = getattr(onnx_module.helper, "make_operatorsetid")
    opset_import = make_opsetid("", opset)
    model = onnx_module.helper.make_model(
        graph,
        producer_name="madmom-model-converter",
        opset_imports=[opset_import],
    )
    model.ir_version = 8
    return model


def _extract_convertible_layers(
    model_object: object, entry: ManifestEntry
) -> list[object]:
    conversion_mode = entry.get("conversion_mode")
    if conversion_mode in (None, "direct_nn"):
        layers = getattr(model_object, "layers", None)
        if isinstance(layers, list):
            return list(layers)
        raise NotImplementedError(
            "direct_nn conversion requires an object with a list-like 'layers' attribute"
        )

    if conversion_mode == "wrapped_nn_core":

        def _looks_like_layer(obj: object) -> bool:
            layer_name = obj.__class__.__name__
            return layer_name.endswith("Layer") or layer_name in {
                "TCNBlock",
                "TCNLayer",
            }

        def _extract_wrapped_node(obj: object) -> object | None:
            layers = getattr(obj, "layers", None)
            if isinstance(layers, list):
                return SequentialLayer(layers)

            processors = getattr(obj, "processors", None)
            if isinstance(processors, (list, tuple)):
                converted_children = []
                for processor in processors:
                    converted = _extract_wrapped_node(processor)
                    if converted is not None:
                        converted_children.append(converted)
                if not converted_children:
                    return None

                if obj.__class__.__name__ == "ParallelProcessor":
                    return ParallelLayer(converted_children)

                return SequentialLayer(converted_children)

            if _looks_like_layer(obj):
                return obj

            return None

        extracted = _extract_wrapped_node(model_object)
        if extracted is not None:
            if extracted.__class__.__name__ == "SequentialLayer":
                return list(cast(Sequence[object], getattr(extracted, "layers")))
            return [extracted]

        raise NotImplementedError(
            "wrapped_nn_core conversion could not find a nested object with 'layers' attribute"
        )

    raise NotImplementedError("unsupported conversion mode: %r" % conversion_mode)


def _convert_pickled_model_to_onnx(
    model_object: object,
    entry: ManifestEntry,
    opset: int,
) -> object:
    layers = _extract_convertible_layers(model_object, entry)
    return _build_onnx_model_from_primitive_layers(layers, opset=opset)


def _default_converter(source_file: str, entry: ManifestEntry, opset: int) -> object:
    with _temporary_repo_root_on_syspath():
        with _temporary_local_madmom_package():
            with _temporary_numpy_shape_base_alias():
                with open(source_file, "rb") as infile:
                    try:
                        model_object = pickle.load(infile, encoding="latin1")
                    except TypeError:
                        model_object = pickle.load(infile)
    return _convert_pickled_model_to_onnx(
        model_object=model_object,
        entry=entry,
        opset=opset,
    )


def _error_result(
    *,
    entry: ManifestEntry,
    status: str,
    error: Exception,
) -> ManifestEntry:
    return {
        "source_pkl": entry["source_pkl"],
        "target_onnx": entry["target_onnx"],
        "source_hash": entry["hash"],
        "status": status,
        "output_hash": None,
        "checker_invoked": False,
        "error": "%s: %s" % (error.__class__.__name__, error),
    }


def convert_manifest_entries(
    manifest: ManifestData,
    models_dir: str = DEFAULT_MODELS_DIR,
    converter: ConverterFn | None = None,
    run_onnx_checker: bool = True,
) -> ManifestData:
    artifacts = _as_list(manifest.get("artifacts"))
    if artifacts is None:
        raise ValueError("manifest must contain an artifacts list")

    converter_fn = converter if converter is not None else _default_converter
    results: list[ManifestEntry] = []
    sorted_artifacts = sorted(artifacts, key=_artifact_source_sort_key)
    for artifact in sorted_artifacts:
        entry = _as_dict(artifact)
        if entry is None:
            raise ValueError("manifest artifacts must only contain objects")

        source_pkl = cast(Optional[str], entry.get("source_pkl"))
        target_onnx = cast(Optional[str], entry.get("target_onnx"))
        source_hash = cast(Optional[str], entry.get("hash"))
        status = entry.get("status")
        convertible = entry.get("convertible")
        opset_value = entry.get("opset")
        opset = opset_value if isinstance(opset_value, int) else DEFAULT_OPSET

        if source_pkl is None or source_hash is None:
            raise ValueError("manifest artifact is missing required source fields")

        if status != "nn_convertible" or convertible is not True or target_onnx is None:
            results.append(
                {
                    "source_pkl": source_pkl,
                    "target_onnx": target_onnx,
                    "source_hash": source_hash,
                    "status": RESULT_SKIPPED_OUT_OF_SCOPE,
                    "output_hash": None,
                    "checker_invoked": False,
                    "error": None,
                }
            )
            continue

        source_file = os.path.join(models_dir, source_pkl.replace("/", os.sep))
        output_path = os.path.join(models_dir, target_onnx.replace("/", os.sep))

        try:
            converted_model = converter_fn(source_file, entry, opset)
            output_hash = write_deterministic_onnx(converted_model, output_path)
        except (pickle.UnpicklingError, EOFError) as error:
            results.append(
                _error_result(entry=entry, status=RESULT_CORRUPT_SOURCE, error=error)
            )
            continue
        except NotImplementedError as error:
            results.append(
                _error_result(
                    entry=entry, status=RESULT_CONVERTER_UNAVAILABLE, error=error
                )
            )
            continue
        except Exception as error:
            results.append(
                _error_result(entry=entry, status=RESULT_CONVERSION_ERROR, error=error)
            )
            continue

        checker_invoked = False
        try:
            if run_onnx_checker:
                checker_invoked = run_onnx_checker_if_available(output_path)
        except Exception as error:
            results.append(
                _error_result(entry=entry, status=RESULT_CHECK_FAILED, error=error)
            )
            continue

        results.append(
            {
                "source_pkl": source_pkl,
                "target_onnx": target_onnx,
                "source_hash": source_hash,
                "status": RESULT_CONVERTED,
                "output_hash": output_hash,
                "checker_invoked": checker_invoked,
                "error": None,
            }
        )

    return {
        "mode": MODE_CONVERT,
        "schema_version": manifest.get("schema_version", 1),
        "results": results,
    }


def _emit_json(data: object, output_path: str | None) -> None:
    if output_path:
        _write_json(data, output_path)
        return
    json.dump(data, sys.stdout, indent=2, sort_keys=True)
    _ = sys.stdout.write("\n")


def _conversion_exit_code(report: ManifestData) -> int:
    results = _as_list(report.get("results"))
    if results is None:
        return 1
    non_success = {
        RESULT_CORRUPT_SOURCE,
        RESULT_CONVERTER_UNAVAILABLE,
        RESULT_CONVERSION_ERROR,
        RESULT_CHECK_FAILED,
    }
    for item in results:
        item_dict = _as_dict(item)
        if item_dict is None:
            return 1
        status = item_dict.get("status")
        if isinstance(status, str) and status in non_success:
            return 1
    return 0


def parse_args(args: Sequence[str] | None = None) -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Inventory, check, and convert shipped pickle artifacts for NN-to-ONNX migration."
    )
    _ = parser.add_argument(
        "--dry-run-inventory",
        action="store_true",
        help="enumerate all shipped pickle artifacts and classify conversion scope",
    )
    _ = parser.add_argument(
        "--convert",
        action="store_true",
        help="run conversion for nn_convertible artifacts and emit structured results",
    )
    _ = parser.add_argument(
        "--check",
        action="store_true",
        help="validate an existing manifest JSON against the schema",
    )
    _ = parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help="directory containing shipped model pickle files",
    )
    _ = parser.add_argument(
        "--schema", default=DEFAULT_SCHEMA_PATH, help="manifest schema path"
    )
    _ = parser.add_argument(
        "--manifest",
        default=None,
        help="output path for dry-run/convert JSON; input path for --check",
    )
    _ = parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help="target ONNX opset for nn_convertible artifacts",
    )
    namespace = CliArgs()
    _ = parser.parse_args(args=args, namespace=namespace)
    return namespace


def main(args: Sequence[str] | None = None) -> int:
    cli_args = parse_args(args=args)

    mode = determine_mode(cli_args)

    if mode == MODE_CHECK:
        if cli_args.manifest is None:
            raise SystemExit("--check requires --manifest")
        schema = _load_json(cli_args.schema)
        manifest = _load_json(cli_args.manifest)
        errors = validate_check_input_shape(manifest, schema)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        return 0

    if mode == MODE_CONVERT:
        manifest = build_supported_artifact_matrix(
            models_dir=cli_args.models_dir, opset=cli_args.opset
        )
        report = convert_manifest_entries(
            manifest=manifest,
            models_dir=cli_args.models_dir,
            run_onnx_checker=True,
        )
        _emit_json(report, cli_args.manifest)
        return _conversion_exit_code(report)

    manifest = build_supported_artifact_matrix(
        models_dir=cli_args.models_dir, opset=cli_args.opset
    )
    _emit_json(manifest, cli_args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
