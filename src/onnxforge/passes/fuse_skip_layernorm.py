"""Fuse residual ``Add`` + ``LayerNormalization`` into ``SkipLayerNormalization``.

ORT's offline fuser only fires when the residual ``Add`` output has a single
consumer. In a pre-norm transformer the residual stream reuses that sum:

    h = x + attn(LN(x))     # `h` feeds BOTH the next LN and the next residual Add

so 7 of 8 LayerNorms in a 4-layer encoder never fuse. ``SkipLayerNormalization``
exposes an optional 4th output ``input_skip_bias_sum`` (= input + skip + bias)
for exactly this case: we route the residual stream through it and delete the
standalone ``Add``.

This pass runs AFTER ``ort_offline`` — it relies on the ``com.microsoft`` opset
import that ORT adds, and mops up the Add+LN pairs ORT left behind.
"""

from __future__ import annotations

import copy

import onnx
from onnx import helper

from onnxforge.passes.base import Pass, PassResult

_MS_DOMAIN = "com.microsoft"


class FuseSkipLayerNormPass(Pass):
    name = "fuse_skip_layernorm"
    description = (
        "Fuse residual Add + LayerNormalization → SkipLayerNormalization (4th-output rewire)"
    )

    def is_applicable(self, model: onnx.ModelProto) -> bool:
        has_ms = any((i.domain or "") == _MS_DOMAIN for i in model.opset_import)
        has_ln = any(n.op_type == "LayerNormalization" for n in model.graph.node)
        return has_ms and has_ln

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
            description=f"{fused} Add+LN pair(s) fused into SkipLayerNormalization"
            if fused
            else "no fusable Add+LN pairs",
            nodes_before=before,
            nodes_after=after,
        )


def _epsilon(ln: onnx.NodeProto) -> float:
    for attr in ln.attribute:
        if attr.name == "epsilon":
            return attr.f
    return 1e-5


def _graph_output_names(graph: onnx.GraphProto) -> set[str]:
    return {o.name for o in graph.output}


def _fuse_one(graph: onnx.GraphProto) -> bool:
    producers = {o: n for n in graph.node for o in n.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    graph_outputs = _graph_output_names(graph)

    for ln in graph.node:
        if ln.op_type != "LayerNormalization":
            continue

        add = producers.get(ln.input[0])
        if add is None or add.op_type != "Add":
            continue

        # LN must own its scale; beta is optional in both LN and SkipLayerNorm.
        if len(ln.input) < 2:
            continue
        gamma = ln.input[1]
        beta = ln.input[2] if len(ln.input) > 2 else ""

        skip_sum = add.output[0]
        sum_consumers = [c for c in consumers.get(skip_sum, []) if c is not ln]
        sum_is_graph_output = skip_sum in graph_outputs

        a, b = add.input[0], add.input[1]
        outputs = [ln.output[0]]
        # Only materialize the 4th (sum) output when something still needs the residual.
        if sum_consumers or sum_is_graph_output:
            outputs += ["", "", skip_sum]

        skip_ln = helper.make_node(
            "SkipLayerNormalization",
            inputs=[a, b, gamma, beta] if beta else [a, b, gamma],
            outputs=outputs,
            name=f"fused_skipln_{ln.name or ''}",
            domain=_MS_DOMAIN,
            epsilon=_epsilon(ln),
        )

        skip = {id(add), id(ln)}
        new_nodes: list[onnx.NodeProto] = []
        for n in graph.node:
            if id(n) == id(add):
                new_nodes.append(skip_ln)
            elif id(n) not in skip:
                new_nodes.append(n)

        del graph.node[:]
        graph.node.extend(new_nodes)
        return True

    return False
