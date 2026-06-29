from dataclasses import dataclass

import onnx


@dataclass
class DimInfo:
    name: str  # symbolic name (e.g. "seq_len")
    input_name: str  # which graph input it appears on
    axis: int  # axis index


@dataclass
class ShapeAnalysis:
    dynamic_dims: list[DimInfo]
    all_static: bool
    memory_pressure_flags: list[str]


def analyze(model: onnx.ModelProto) -> ShapeAnalysis:
    dynamic_dims = _collect_dynamic_dims(model)
    memory_flags = _check_memory_pressure(model, {d.name for d in dynamic_dims})
    return ShapeAnalysis(
        dynamic_dims=dynamic_dims,
        all_static=len(dynamic_dims) == 0,
        memory_pressure_flags=memory_flags,
    )


def _collect_dynamic_dims(model: onnx.ModelProto) -> list[DimInfo]:
    dims: list[DimInfo] = []
    for inp in model.graph.input:
        tensor_type = inp.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        for axis, dim in enumerate(tensor_type.shape.dim):
            if dim.dim_param:
                dims.append(DimInfo(name=dim.dim_param, input_name=inp.name, axis=axis))
    return dims


def _check_memory_pressure(model: onnx.ModelProto, dynamic_names: set[str]) -> list[str]:
    flags: list[str] = []

    if dynamic_names:
        flags.append(
            f"Dynamic dims {sorted(dynamic_names)} cause XNNPACK weight repacking on "
            "every shape change. Consider bucketing to a fixed set of shapes."
        )

    # Flag large static concat outputs (rough heuristic: many large initializers)
    large_inits = [i for i in model.graph.initializer if _tensor_numel(i) > 10_000_000]
    if large_inits:
        flags.append(
            f"{len(large_inits)} initializer(s) exceed 10M elements — "
            "ensure they fit comfortably in device RAM."
        )

    return flags


def _tensor_numel(tensor: onnx.TensorProto) -> int:
    n = 1
    for d in tensor.dims:
        n *= d
    return n
