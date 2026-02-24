from __future__ import absolute_import, division, print_function

import importlib

import numpy as np

from ...processors import Processor


CPU_EXECUTION_PROVIDER = "CPUExecutionProvider"


def _import_onnxruntime():
    try:
        return importlib.import_module("onnxruntime")
    except ImportError as error:
        raise ImportError(
            "onnxruntime is required for madmom.ml.nn runtime; install the "
            "optional dependency and load ONNX model files"
        ) from error


def _validate_providers(providers):
    if providers is None:
        return [CPU_EXECUTION_PROVIDER]
    if not isinstance(providers, (list, tuple)):
        raise ValueError("providers must be a list or tuple")
    normalized = list(providers)
    if any(provider != CPU_EXECUTION_PROVIDER for provider in normalized):
        raise ValueError(
            "only CPUExecutionProvider is supported for madmom.ml.nn runtime"
        )
    return [CPU_EXECUTION_PROVIDER]


def _create_session_options(ort, intra_op_num_threads=2):
    session_options = ort.SessionOptions()
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = intra_op_num_threads
    session_options.inter_op_num_threads = 1
    session_options.enable_mem_pattern = True
    session_options.enable_cpu_mem_arena = True
    return session_options


def _state_shape(input_meta):
    shape = []
    for dim in input_meta.shape:
        if isinstance(dim, np.integer):
            dim = int(dim)
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(
                "state input '%s' must have static positive dimensions; got %s"
                % (input_meta.name, input_meta.shape)
            )
        shape.append(dim)
    if not shape:
        raise ValueError("state input '%s' shape must not be empty" % input_meta.name)
    return tuple(shape)


class OnnxNeuralNetworkRuntime(Processor):
    def __init__(self, model_file, providers=None, intra_op_num_threads=2):
        self._model_source = model_file
        self._ort = _import_onnxruntime()
        self.providers = _validate_providers(providers)
        self._intra_op_num_threads = intra_op_num_threads
        self._configure_session()

    def _configure_session(self):
        self.session_options = _create_session_options(
            self._ort, intra_op_num_threads=self._intra_op_num_threads
        )
        self.session = self._ort.InferenceSession(
            self._model_source,
            sess_options=self.session_options,
            providers=self.providers,
        )
        self._inputs = list(self.session.get_inputs())
        self._overridable_inputs = list(self.session.get_overridable_initializers())
        self._outputs = list(self.session.get_outputs())
        if not self._inputs:
            raise ValueError("ONNX model must expose at least one input")
        if not self._outputs:
            raise ValueError("ONNX model must expose at least one output")
        self.data_input_name = self._inputs[0].name

        state_input_meta = list(self._inputs[1:])
        existing_state_names = {state_input.name for state_input in state_input_meta}
        for state_input in self._overridable_inputs:
            if state_input.name == self.data_input_name:
                continue
            if state_input.name in existing_state_names:
                continue
            state_input_meta.append(state_input)
            existing_state_names.add(state_input.name)

        self.state_input_meta = state_input_meta
        self.state_input_names = [
            state_input.name for state_input in self.state_input_meta
        ]
        self._state_shapes = {
            state_input.name: _state_shape(state_input)
            for state_input in self.state_input_meta
        }
        overridable_initializers = set(
            initializer.name
            for initializer in self.session.get_overridable_initializers()
        )
        self._state_inputs_with_default_initializer = {
            name for name in self.state_input_names if name in overridable_initializers
        }

        state_count = len(self.state_input_names)
        if state_count > len(self._outputs):
            raise ValueError(
                "ONNX model has %d state inputs but only %d outputs"
                % (state_count, len(self._outputs))
            )
        split_index = len(self._outputs) - state_count
        self.prediction_output_names = [
            output.name for output in self._outputs[:split_index]
        ]
        self.state_output_names = [
            output.name for output in self._outputs[split_index:]
        ]
        if not self.prediction_output_names:
            raise ValueError("ONNX model must expose at least one prediction output")

        self._initial_state = {
            state_input_name: np.zeros(state_shape, dtype=np.float32)
            for state_input_name, state_shape in self._state_shapes.items()
            if state_input_name not in self._state_inputs_with_default_initializer
        }
        self._state = {}
        self.reset()

    def __getstate__(self):
        state_values = {
            state_input_name: (
                None
                if self._state[state_input_name] is None
                else np.array(
                    self._state[state_input_name], dtype=np.float32, copy=True
                )
            )
            for state_input_name in self.state_input_names
        }
        return {
            "model_source": self._model_source,
            "providers": list(self.providers),
            "intra_op_num_threads": self._intra_op_num_threads,
            "state_values": state_values,
        }

    def __setstate__(self, state):
        self._model_source = state["model_source"]
        self._ort = _import_onnxruntime()
        self.providers = _validate_providers(state["providers"])
        self._intra_op_num_threads = state.get("intra_op_num_threads", 2)
        self._configure_session()

        state_values = state.get("state_values", {})
        for state_input_name in self.state_input_names:
            restored_state = state_values.get(
                state_input_name, self._state[state_input_name]
            )
            if restored_state is None:
                self._state[state_input_name] = None
                continue
            restored_state = np.asarray(restored_state, dtype=np.float32)
            expected_shape = self._state_shapes[state_input_name]
            if restored_state.shape != expected_shape:
                raise ValueError(
                    "restored state shape mismatch for '%s': expected %s, got %s"
                    % (state_input_name, expected_shape, restored_state.shape)
                )
            self._state[state_input_name] = np.array(
                restored_state,
                dtype=np.float32,
                copy=True,
            )

    def reset(self):
        self._state = {}
        for state_input_name in self.state_input_names:
            if state_input_name in self._state_inputs_with_default_initializer:
                self._state[state_input_name] = None
            else:
                self._state[state_input_name] = np.array(
                    self._initial_state[state_input_name],
                    dtype=np.float32,
                    copy=True,
                )

    def process(self, data, reset=True, **kwargs):
        del kwargs
        if reset:
            self.reset()

        input_data = np.asarray(data, dtype=np.float32)
        if input_data.ndim < 2:
            input_data = np.array(input_data, subok=True, copy=False, ndmin=2)

        ort_inputs = {self.data_input_name: input_data}
        for state_input_name in self.state_input_names:
            state_input_value = self._state[state_input_name]
            if state_input_value is not None:
                ort_inputs[state_input_name] = state_input_value

        raw_outputs = self.session.run(
            self.prediction_output_names + self.state_output_names,
            ort_inputs,
        )
        output_map = dict(
            zip(self.prediction_output_names + self.state_output_names, raw_outputs)
        )

        for index, state_input_name in enumerate(self.state_input_names):
            state_output_name = self.state_output_names[index]
            state_output = np.asarray(output_map[state_output_name], dtype=np.float32)
            expected_shape = self._state_shapes[state_input_name]
            if state_output.shape != expected_shape:
                raise ValueError(
                    "state output shape mismatch for '%s': expected %s, got %s"
                    % (state_input_name, expected_shape, state_output.shape)
                )
            self._state[state_input_name] = state_output

        predictions = [
            np.asarray(output_map[name]) for name in self.prediction_output_names
        ]
        if len(predictions) == 1:
            pred = predictions[0]
            if pred.ndim > 1:
                return pred.squeeze()
            return pred
        return tuple(p.squeeze() if p.ndim > 1 else p for p in predictions)
