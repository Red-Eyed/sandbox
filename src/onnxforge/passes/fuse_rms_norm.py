"""Fuse standalone RMSNorm arithmetic into RMSNormalization (standard opset 23).

nn.RMSNorm decomposes under dynamo into explicit arithmetic:

    Pow(x, 2) → ReduceMean(axes) → Add(eps) → Sqrt → Reciprocal
                                                             ↓
    x ──────────────────────────────────────────────→ Mul(x, rsqrt)
                                                             ↓
    gamma ──────────────────────────────────────────→ Mul(norm, gamma)  ← optional learnable scale

Seven nodes become one RMSNormalization (standard ONNX opset 23).

For models at opset < 23 the pass is gated out by Pass.min_std_opset — use
FuseSkipRMSNormPass instead, which handles Add+RMSNorm pairs via the
com.microsoft SkipSimplifiedLayerNormalization op and has no opset floor.

This pass runs BEFORE OnnxSlimPass — onnxslim folds the gamma initializer into
the downstream MatMul weights, making the pattern unrecoverable afterward.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import helper, numpy_helper

from onnxforge.passes.base import Pass, PassResult

_RMSNORM_OPSET = 23


@dataclass
class _Match:
    pow_node: onnx.NodeProto
    reduce_mean: onnx.NodeProto
    add_eps: onnx.NodeProto
    sqrt_node: onnx.NodeProto
    reciprocal: onnx.NodeProto
    mul_norm: onnx.NodeProto
    mul_gamma: onnx.NodeProto | None
    x: str
    scale_name: str  # initializer name; all-ones tensor created if gamma absent
    epsilon: float
    axis: int
    out_name: str  # tensor the fused node must produce


class FuseRMSNormPass(Pass):
    name = "fuse_rms_norm"
    description = "Fuse Pow→ReduceMean→Add→Sqrt→Reciprocal→Mul[→Mul(gamma)] → RMSNormalization"
    min_std_opset = _RMSNORM_OPSET  # RMSNormalization added in opset 23

    def is_applicable(self, model: onnx.ModelProto) -> bool:
        return any(n.op_type == "Pow" for n in model.graph.node)

    def apply(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, PassResult]:
        model = copy.deepcopy(model)
        graph = model.graph
        before = len(graph.node)

        fused = 0
        changed = True
        while changed:
            changed = _fuse_one(graph)
            if changed:
                fused += 1

        after = len(graph.node)
        return model, PassResult(
            applied=fused > 0,
            description=f"{fused} RMSNorm subgraph(s) fused into RMSNormalization"
            if fused
            else "no RMSNorm patterns found",
            nodes_before=before,
            nodes_after=after,
        )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _fuse_one(graph: onnx.GraphProto) -> bool:
    inits = {i.name: i for i in graph.initializer}
    producers = {o: n for n in graph.node for o in n.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)

    for node in graph.node:
        if node.op_type != "Pow":
            continue
        match = _try_match(node, producers, consumers, inits, graph)
        if match is None:
            continue
        _apply_match(graph, match)
        return True

    return False


def _try_match(
    pow_node: onnx.NodeProto,
    producers: dict[str, onnx.NodeProto],
    consumers: dict[str, list[onnx.NodeProto]],
    inits: dict[str, onnx.TensorProto],
    graph: onnx.GraphProto,
) -> _Match | None:
    x = pow_node.input[0]

    exp = _scalar_const(pow_node.input[1], inits)
    if exp is None or abs(exp - 2.0) > 1e-6:
        return None

    reduce_mean = _sole_consumer(pow_node.output[0], consumers)
    if reduce_mean is None or reduce_mean.op_type != "ReduceMean":
        return None
    axis = _reduce_axis(reduce_mean, inits)
    if axis is None:
        return None

    add_eps = _sole_consumer(reduce_mean.output[0], consumers)
    if add_eps is None or add_eps.op_type != "Add":
        return None
    eps = _other_scalar_const(add_eps, reduce_mean.output[0], inits)
    if eps is None:
        return None

    sqrt_node = _sole_consumer(add_eps.output[0], consumers)
    if sqrt_node is None or sqrt_node.op_type != "Sqrt":
        return None

    reciprocal = _sole_consumer(sqrt_node.output[0], consumers)
    if reciprocal is None or reciprocal.op_type != "Reciprocal":
        return None

    mul_norm = _sole_consumer(reciprocal.output[0], consumers)
    if mul_norm is None or mul_norm.op_type != "Mul":
        return None
    rsqrt = reciprocal.output[0]
    if set(mul_norm.input) != {x, rsqrt}:
        return None

    # Optional: Mul(norm, gamma_initializer)
    mul_gamma: onnx.NodeProto | None = None
    scale_name: str | None = None
    out_name = mul_norm.output[0]

    norm_consumers = consumers.get(mul_norm.output[0], [])
    if len(norm_consumers) == 1 and norm_consumers[0].op_type == "Mul":
        candidate = norm_consumers[0]
        gamma = _const_input_other_than(candidate, mul_norm.output[0], inits)
        if gamma is not None:
            mul_gamma = candidate
            scale_name = gamma
            out_name = candidate.output[0]

    if scale_name is None:
        scale_name = _ensure_ones_scale(graph, inits, x, axis)
        if scale_name is None:
            return None

    return _Match(
        pow_node=pow_node,
        reduce_mean=reduce_mean,
        add_eps=add_eps,
        sqrt_node=sqrt_node,
        reciprocal=reciprocal,
        mul_norm=mul_norm,
        mul_gamma=mul_gamma,
        x=x,
        scale_name=scale_name,
        epsilon=eps,
        axis=axis,
        out_name=out_name,
    )


def _apply_match(graph: onnx.GraphProto, match: _Match) -> None:
    rms_node = helper.make_node(
        "RMSNormalization",
        inputs=[match.x, match.scale_name],
        outputs=[match.out_name],
        name=f"rms_norm_{match.pow_node.name or ''}",
        axis=match.axis,
        epsilon=match.epsilon,
    )

    drop = {
        id(match.pow_node),
        id(match.reduce_mean),
        id(match.add_eps),
        id(match.sqrt_node),
        id(match.reciprocal),
        id(match.mul_norm),
    }
    if match.mul_gamma is not None:
        drop.add(id(match.mul_gamma))

    new_nodes: list[onnx.NodeProto] = []
    inserted = False
    for n in graph.node:
        if id(n) == id(match.pow_node):
            new_nodes.append(rms_node)
            inserted = True
        elif id(n) not in drop:
            new_nodes.append(n)
    assert inserted

    del graph.node[:]
    graph.node.extend(new_nodes)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sole_consumer(
    tensor: str, consumers: dict[str, list[onnx.NodeProto]]
) -> onnx.NodeProto | None:
    nodes = consumers.get(tensor, [])
    return nodes[0] if len(nodes) == 1 else None


def _scalar_const(name: str, inits: dict[str, onnx.TensorProto]) -> float | None:
    if name not in inits:
        return None
    arr = numpy_helper.to_array(inits[name])
    return float(arr.flat[0]) if arr.size == 1 else None


def _other_scalar_const(
    node: onnx.NodeProto, known_input: str, inits: dict[str, onnx.TensorProto]
) -> float | None:
    for inp in node.input:
        if inp != known_input:
            val = _scalar_const(inp, inits)
            if val is not None:
                return val
    return None


def _const_input_other_than(
    node: onnx.NodeProto, known_input: str, inits: dict[str, onnx.TensorProto]
) -> str | None:
    for inp in node.input:
        if inp != known_input and inp in inits:
            return inp
    return None


def _reduce_axis(rm: onnx.NodeProto, inits: dict[str, onnx.TensorProto]) -> int | None:
    # Opset 18+: axes is an optional second input tensor.
    if len(rm.input) > 1 and rm.input[1]:
        arr = numpy_helper.to_array(inits[rm.input[1]]) if rm.input[1] in inits else None
        if arr is None or arr.size != 1:
            return None
        return int(arr.flat[0])
    # Older opset: axes attribute.
    for attr in rm.attribute:
        if attr.name == "axes" and len(attr.ints) == 1:
            return int(attr.ints[0])
    return None


def _ensure_ones_scale(
    graph: onnx.GraphProto,
    inits: dict[str, onnx.TensorProto],
    x: str,
    axis: int,
) -> str | None:
    """Create an all-ones scale when the model has no explicit gamma.

    Infers the scale dimension from graph value_info shape annotations.
    Returns None if the static dimension cannot be determined.
    """
    shape_map = {vi.name: vi.type.tensor_type.shape for vi in graph.value_info}
    shape = shape_map.get(x)
    if shape is None:
        return None
    dims = list(shape.dim)
    idx = axis if axis >= 0 else len(dims) + axis
    if idx < 0 or idx >= len(dims) or not dims[idx].HasField("dim_value"):
        return None
    size = dims[idx].dim_value
    name = f"__ones_scale_{x}"
    if name not in inits:
        ones = np.ones(size, dtype=np.float32)
        graph.initializer.append(numpy_helper.from_array(ones, name=name))
    return name
