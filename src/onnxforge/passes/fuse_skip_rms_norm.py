"""Fuse residual Add + RMSNorm arithmetic into SkipSimplifiedLayerNormalization.

In a pre-norm transformer the residual stream feeds both the next sub-layer's
normalization and the following skip connection:

    h = a + b                          ← Add (residual)
    h → Pow(h,2) → ReduceMean → Add(eps) → Sqrt → Reciprocal → Mul(h, rsqrt)
                                                                      ↓
    gamma ────────────────────────────────────────────────────→ Mul(norm, gamma)
    h ─────────────────────────────────────────────────────────────→ next Add

com.microsoft SkipSimplifiedLayerNormalization fuses the whole thing:

    SkipSimplifiedLayerNormalization(a, b, gamma, epsilon=eps)
        output[0] = rms_norm(a + b) * gamma
        output[3] = a + b          ← 4th output wired back as the residual

Analogous to FuseSkipLayerNormPass but for RMSNorm.  Works at any standard
opset — SkipSimplifiedLayerNormalization is a com.microsoft contrib op with no
standard-domain opset requirement.  The pass adds the com.microsoft opset
import when absent.

Runs BEFORE OnnxSlimPass so the gamma weight is still a distinct initializer.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import onnx
from onnx import helper, numpy_helper

from onnxforge.passes.base import Pass, PassResult

_MS_DOMAIN = "com.microsoft"


@dataclass
class _Match:
    add_node: onnx.NodeProto  # the residual Add
    pow_node: onnx.NodeProto
    reduce_mean: onnx.NodeProto
    add_eps: onnx.NodeProto
    sqrt_node: onnx.NodeProto
    reciprocal: onnx.NodeProto
    mul_norm: onnx.NodeProto
    mul_gamma: onnx.NodeProto | None
    a: str  # add.input[0]
    b: str  # add.input[1]
    scale_name: str
    epsilon: float
    axis: int
    norm_out: str  # tensor the fused node produces
    sum_out: str | None  # if residual stream still needed, emit 4th output here


class FuseSkipRMSNormPass(Pass):
    name = "fuse_skip_rms_norm"
    description = (
        "Fuse residual Add + RMSNorm → SkipSimplifiedLayerNormalization (4th-output rewire)"
    )
    # SkipSimplifiedLayerNormalization is a com.microsoft op with no standard
    # opset requirement — min_std_opset stays at the default of 1.

    def is_applicable(self, model: onnx.ModelProto) -> bool:
        has_pow = any(n.op_type == "Pow" for n in model.graph.node)
        has_add = any(n.op_type == "Add" for n in model.graph.node)
        return has_pow and has_add

    def apply(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, PassResult]:
        model = copy.deepcopy(model)

        if not any((i.domain or "") == _MS_DOMAIN for i in model.opset_import):
            entry = model.opset_import.add()
            entry.domain = _MS_DOMAIN
            entry.version = 1

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
            description=f"{fused} Add+RMSNorm pair(s) fused into SkipSimplifiedLayerNormalization"
            if fused
            else "no Add+RMSNorm pairs found",
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
    graph_outputs = {o.name for o in graph.output}

    for node in graph.node:
        if node.op_type != "Pow":
            continue
        match = _try_match(node, producers, consumers, inits, graph_outputs)
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
    graph_outputs: set[str],
) -> _Match | None:
    h = pow_node.input[0]

    # h must be produced by an Add (the residual addition).
    add_node = producers.get(h)
    if add_node is None or add_node.op_type != "Add":
        return None
    a, b = add_node.input[0], add_node.input[1]

    exp = _scalar_const(pow_node.input[1], inits)
    if exp is None or abs(exp - 2.0) > 1e-6:
        return None

    # Pow must have exactly one consumer (ReduceMean).
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
    if set(mul_norm.input) != {h, rsqrt}:
        return None

    # Optional: Mul(norm, gamma_initializer).
    mul_gamma: onnx.NodeProto | None = None
    scale_name: str | None = None
    norm_out = mul_norm.output[0]

    norm_consumers = consumers.get(mul_norm.output[0], [])
    if len(norm_consumers) == 1 and norm_consumers[0].op_type == "Mul":
        candidate = norm_consumers[0]
        gamma = _const_input_other_than(candidate, mul_norm.output[0], inits)
        if gamma is not None:
            mul_gamma = candidate
            scale_name = gamma
            norm_out = candidate.output[0]

    if scale_name is None:
        return None  # no gamma found; this pass requires an explicit scale

    # Determine whether the residual h = a+b is still needed by other nodes
    # after the fusion.  The Add node disappears, so anything else consuming h
    # must be served by SkipSimplifiedLayerNormalization's 4th output.
    rms_nodes = {
        id(add_node),
        id(pow_node),
        id(reduce_mean),
        id(add_eps),
        id(sqrt_node),
        id(reciprocal),
        id(mul_norm),
    }
    if mul_gamma is not None:
        rms_nodes.add(id(mul_gamma))

    other_consumers_of_h = [c for c in consumers.get(h, []) if id(c) not in rms_nodes]
    sum_out: str | None = None
    if other_consumers_of_h or h in graph_outputs:
        sum_out = h

    return _Match(
        add_node=add_node,
        pow_node=pow_node,
        reduce_mean=reduce_mean,
        add_eps=add_eps,
        sqrt_node=sqrt_node,
        reciprocal=reciprocal,
        mul_norm=mul_norm,
        mul_gamma=mul_gamma,
        a=a,
        b=b,
        scale_name=scale_name,
        epsilon=eps,
        axis=axis,
        norm_out=norm_out,
        sum_out=sum_out,
    )


def _apply_match(graph: onnx.GraphProto, match: _Match) -> None:
    outputs = [match.norm_out]
    if match.sum_out is not None:
        outputs += ["", "", match.sum_out]

    skip_rms = helper.make_node(
        "SkipSimplifiedLayerNormalization",
        inputs=[match.a, match.b, match.scale_name],
        outputs=outputs,
        name=f"skip_rms_norm_{match.pow_node.name or ''}",
        domain=_MS_DOMAIN,
        epsilon=match.epsilon,
    )

    drop = {
        id(match.add_node),
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
        if id(n) == id(match.add_node):
            new_nodes.append(skip_rms)
            inserted = True
        elif id(n) not in drop:
            new_nodes.append(n)
    assert inserted

    del graph.node[:]
    graph.node.extend(new_nodes)


# ---------------------------------------------------------------------------
# helpers (shared pattern with fuse_rms_norm — keep in sync)
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
    if len(rm.input) > 1 and rm.input[1]:
        arr = numpy_helper.to_array(inits[rm.input[1]]) if rm.input[1] in inits else None
        if arr is None or arr.size != 1:
            return None
        return int(arr.flat[0])
    for attr in rm.attribute:
        if attr.name == "axes" and len(attr.ints) == 1:
            return int(attr.ints[0])
    return None
