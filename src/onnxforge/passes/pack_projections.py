"""Pack parallel projections that share an activation into one GEMM.

In a transformer block several MatMuls read the *same* normalized activation:

    Q = LN(x) @ Wq      K = LN(x) @ Wk      V = LN(x) @ Wv      (attention)
    g = LN(x) @ Wg      u = LN(x) @ Wu                          (gated FF)

Each is a separate small GEMM. Concatenating the weights along the output axis
turns them into one larger GEMM followed by a ``Split`` — fewer kernel launches
and better cache reuse on both MLAS and XNNPACK, with identical numerics.

This pass runs BEFORE ``ort_offline`` so ORT sees the packed form. It only fires
when every sibling weight is a constant initializer sharing the contraction dim
and the shared activation is the first MatMul input (``act @ W``).
"""

from __future__ import annotations

import copy

import numpy as np
import onnx
from onnx import helper, numpy_helper

from onnxforge.passes.base import Pass, PassResult


class PackProjectionsPass(Pass):
    name = "pack_projections"
    description = "Pack parallel act@W projections (QKV, gated-FF) into one GEMM + Split"

    def apply(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, PassResult]:
        model = copy.deepcopy(model)
        graph = model.graph
        before = len(graph.node)

        packed = 0
        changed = True
        while changed:
            changed = _pack_one(graph)
            if changed:
                packed += 1

        after = len(graph.node)
        return model, PassResult(
            applied=packed > 0,
            description=f"{packed} projection group(s) packed" if packed else "no packable groups",
            nodes_before=before,
            nodes_after=after,
        )


def _pack_one(graph: onnx.GraphProto) -> bool:
    inits = {i.name: i for i in graph.initializer}
    groups = _projection_groups(graph, inits)

    for act, matmuls in groups.items():
        weights = [_weight_input(mm, inits) for mm in matmuls]
        arrays = [numpy_helper.to_array(inits[w]) for w in weights]

        # Every weight must be 2-D and share the contraction (first) dim.
        if any(a.ndim != 2 for a in arrays):
            continue
        if len({a.shape[0] for a in arrays}) != 1:
            continue

        _do_pack(graph, act, matmuls, weights, arrays)
        return True

    return False


def _projection_groups(
    graph: onnx.GraphProto, inits: dict[str, onnx.TensorProto]
) -> dict[str, list[onnx.NodeProto]]:
    """Group act@W MatMuls (W constant, act = input[0]) by shared activation."""
    groups: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        act, weight = node.input[0], node.input[1]
        if act in inits or weight not in inits:
            continue
        groups.setdefault(act, []).append(node)
    return {act: mms for act, mms in groups.items() if len(mms) >= 2}


def _weight_input(matmul: onnx.NodeProto, inits: dict[str, onnx.TensorProto]) -> str:
    return matmul.input[1]


def _do_pack(
    graph: onnx.GraphProto,
    act: str,
    matmuls: list[onnx.NodeProto],
    weights: list[str],
    arrays: list[np.ndarray],
) -> None:
    sizes = [a.shape[1] for a in arrays]
    packed_w = np.concatenate(arrays, axis=1)

    base = matmuls[0].name or weights[0]
    packed_w_name = f"packed_w_{base}"
    packed_out = f"packed_out_{base}"
    split_sizes_name = f"packed_split_{base}"

    graph.initializer.append(numpy_helper.from_array(packed_w, name=packed_w_name))
    graph.initializer.append(
        numpy_helper.from_array(np.array(sizes, dtype=np.int64), name=split_sizes_name)
    )

    matmul_node = helper.make_node(
        "MatMul", inputs=[act, packed_w_name], outputs=[packed_out], name=f"packed_mm_{base}"
    )
    split_node = helper.make_node(
        "Split",
        inputs=[packed_out, split_sizes_name],
        outputs=[mm.output[0] for mm in matmuls],  # reuse names → consumers unchanged
        name=f"packed_split_node_{base}",
        axis=-1,
    )

    drop = {id(mm) for mm in matmuls}
    new_nodes: list[onnx.NodeProto] = []
    inserted = False
    for n in graph.node:
        if id(n) in drop:
            if not inserted:
                new_nodes.extend([matmul_node, split_node])
                inserted = True
            continue
        new_nodes.append(n)

    del graph.node[:]
    graph.node.extend(new_nodes)
