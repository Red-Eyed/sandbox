import copy

import onnx
from onnx import helper

from onnx_speedup.passes.base import Pass, PassResult


class FuseGemmPass(Pass):
    name = "fuse_gemm"
    description = "Fuse MatMul+Add(bias) → Gemm"

    def apply(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, PassResult]:
        model = copy.deepcopy(model)
        graph = model.graph
        before = len(graph.node)

        initializer_names = {i.name for i in graph.initializer}

        fusions = 0
        changed = True
        while changed:
            changed = _fuse_one(graph, initializer_names, fusions)
            if changed:
                fusions += 1

        after = len(graph.node)
        return model, PassResult(
            applied=fusions > 0,
            description=f"{fusions} MatMul+Add fusion(s) → Gemm"
            if fusions > 0
            else "no MatMul+Add pairs found",
            nodes_before=before,
            nodes_after=after,
        )


def _count_output_uses(graph: onnx.GraphProto) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                counts[inp] = counts.get(inp, 0) + 1
    return counts


def _build_rank_map(graph: onnx.GraphProto) -> dict[str, int]:
    """Maps tensor names to their known rank (number of dims)."""
    ranks: dict[str, int] = {}
    for init in graph.initializer:
        ranks[init.name] = len(init.dims)
    for inp in graph.input:
        tt = inp.type.tensor_type
        if tt.HasField("shape"):
            ranks[inp.name] = len(tt.shape.dim)
    for vi in graph.value_info:
        tt = vi.type.tensor_type
        if tt.HasField("shape"):
            ranks[vi.name] = len(tt.shape.dim)
    return ranks


def _fuse_one(
    graph: onnx.GraphProto,
    initializer_names: set[str],
    fusion_idx: int,
) -> bool:
    """Fuse a single MatMul+Add pair. Returns True if a fusion was made."""
    output_use_count = _count_output_uses(graph)
    rank_map = _build_rank_map(graph)
    nodes = list(graph.node)
    matmul_by_output: dict[str, onnx.NodeProto] = {
        n.output[0]: n for n in nodes if n.op_type == "MatMul" and len(n.output) == 1
    }

    for add_node in nodes:
        if add_node.op_type != "Add":
            continue
        a, b = add_node.input[0], add_node.input[1]
        matmul, bias_name = _find_matmul_and_bias(
            a, b, matmul_by_output, initializer_names, output_use_count, rank_map
        )
        if matmul is None:
            continue

        gemm = helper.make_node(
            "Gemm",
            inputs=[matmul.input[0], matmul.input[1], bias_name],
            outputs=list(add_node.output),
            name=f"fused_Gemm_{fusion_idx}",
            transB=0,
            alpha=1.0,
            beta=1.0,
        )
        # Insert Gemm at MatMul's position to preserve topological order.
        # Drop the Add node (its output is now produced by Gemm directly).
        skip = {id(add_node)}
        new_nodes = []
        for n in graph.node:
            if id(n) == id(matmul):
                new_nodes.append(gemm)
            elif id(n) not in skip:
                new_nodes.append(n)

        del graph.node[:]
        graph.node.extend(new_nodes)
        return True

    return False


def _find_matmul_and_bias(
    a: str,
    b: str,
    matmul_by_output: dict[str, onnx.NodeProto],
    initializer_names: set[str],
    output_use_count: dict[str, int],
    rank_map: dict[str, int],
) -> tuple[onnx.NodeProto | None, str]:
    for matmul_out, bias in [(a, b), (b, a)]:
        if matmul_out not in matmul_by_output:
            continue
        if bias not in initializer_names:
            continue
        if output_use_count.get(matmul_out, 0) != 1:
            continue
        matmul = matmul_by_output[matmul_out]
        # Gemm requires rank-2 inputs; skip if first input rank is known and != 2.
        x_name = matmul.input[0]
        x_rank = rank_map.get(x_name)
        if x_rank is not None and x_rank != 2:
            continue
        return matmul, bias
    return None, ""
