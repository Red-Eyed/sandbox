"""Fold the SDPA K^T cluster into a single Transpose.

When torch.export decomposes F.scaled_dot_product_attention at opset 17, the
K.transpose(-2, -1) inside SDPA expands to a 7-node cluster per attention layer:

    K [B, H, S, d]           # head-split Transpose output (perm=[0,2,1,3])
    → Reshape([?, S, d])      # merge B and H, needs runtime shape
    → Transpose([0, 2, 1])    # 3-D K^T: [?, d, S]
    → Reshape([B, H, d, S])   # restore 4-D
    → K^T [B, H, d, S]

This is equivalent to a single Transpose(perm=[0, 1, 3, 2]) applied directly
to the 4-D K. The cluster exists only because einops rearrange with a dynamic
seq-len dimension forces runtime Shape extraction to build the reshape targets.
"""

from __future__ import annotations

import copy

import onnx
from onnx import helper

from onnxforge.passes.base import Pass, PassResult


class FoldTransposePass(Pass):
    name = "fold_transpose"
    description = "Fold K^T cluster (Reshape→Transpose([0,2,1])→Reshape) → Transpose([0,1,3,2])"

    def apply(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, PassResult]:
        model = copy.deepcopy(model)
        graph = model.graph
        before = len(graph.node)

        fused = 0
        changed = True
        while changed:
            producers = {n.output[0]: n for n in graph.node if n.output}
            use_count = _count_uses(graph)
            changed = _fold_one(graph, producers, use_count)
            if changed:
                fused += 1

        after = len(graph.node)
        return model, PassResult(
            applied=fused > 0,
            description=f"{fused} K^T cluster(s) folded" if fused else "no K^T clusters found",
            nodes_before=before,
            nodes_after=after,
        )


def _count_uses(graph: onnx.GraphProto) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                counts[inp] = counts.get(inp, 0) + 1
    return counts


def _get_perm(node: onnx.NodeProto) -> list[int]:
    for attr in node.attribute:
        if attr.name == "perm":
            return list(attr.ints)
    return []


def _fold_one(
    graph: onnx.GraphProto,
    producers: dict[str, onnx.NodeProto],
    use_count: dict[str, int],
) -> bool:
    for node in graph.node:
        if node.op_type != "Transpose" or _get_perm(node) != [0, 2, 1]:
            continue

        # Input must come from a Reshape
        reshape1 = producers.get(node.input[0])
        if reshape1 is None or reshape1.op_type != "Reshape":
            continue

        # The Reshape's data input must come from a [B,S,H,d]→[B,H,S,d] Transpose
        k_4d = reshape1.input[0]
        head_split = producers.get(k_4d)
        if head_split is None or head_split.op_type != "Transpose":
            continue
        if _get_perm(head_split) != [0, 2, 1, 3]:
            continue

        # 3-D Transpose output must be consumed only by a single Reshape
        if use_count.get(node.output[0], 0) != 1:
            continue
        reshape2_list = [n for n in graph.node if node.output[0] in n.input]
        if len(reshape2_list) != 1 or reshape2_list[0].op_type != "Reshape":
            continue
        reshape2 = reshape2_list[0]

        # Replace the 3-node cluster (reshape1, 3-D transpose, reshape2) with
        # a single Transpose(perm=[0,1,3,2]) inserted at reshape1's position.
        new_t = helper.make_node(
            "Transpose",
            inputs=[k_4d],
            outputs=list(reshape2.output),
            name=f"folded_kt_{node.name or ''}",
            perm=[0, 1, 3, 2],
        )
        skip = {id(node), id(reshape2)}
        new_nodes: list[onnx.NodeProto] = []
        for n in graph.node:
            if id(n) == id(reshape1):
                new_nodes.append(new_t)
            elif id(n) not in skip:
                new_nodes.append(n)

        del graph.node[:]
        graph.node.extend(new_nodes)
        return True

    return False
