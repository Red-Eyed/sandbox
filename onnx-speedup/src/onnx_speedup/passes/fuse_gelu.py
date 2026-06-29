"""
Fuse torch-decomposed GeLU into com.microsoft.Gelu.

Torch decomposition of GeLU(approximate='none'):
  x → Div(x, sqrt(2)) → Erf → Add(_, 1) → Mul(x, _) → Mul(_, 0.5)

We match this by walking the graph and checking the chain forward from Div nodes.
"""

import copy
import math

import onnx
from onnx import helper, numpy_helper

from onnx_speedup.passes.base import Pass, PassResult

_MS_DOMAIN = "com.microsoft"
_SQRT2 = math.sqrt(2.0)
_SQRT2_TOL = 1e-4


class FuseGeluPass(Pass):
    name = "fuse_gelu"
    description = "Fuse decomposed GeLU → com.microsoft.Gelu"

    def apply(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, PassResult]:
        model = copy.deepcopy(model)
        graph = model.graph
        before = len(graph.node)

        initializers = {i.name: i for i in graph.initializer}
        consumers = _build_consumer_map(graph)
        output_use_count = _count_uses(graph)

        fusions = 0
        nodes_to_remove: set[int] = set()
        # Maps first-chain-node-id → replacement Gelu node (for in-place insertion)
        replacements: dict[int, onnx.NodeProto] = {}

        for node in list(graph.node):
            if id(node) in nodes_to_remove:
                continue
            chain = _match_gelu_chain(node, consumers, initializers, output_use_count)
            if chain is None:
                continue
            gelu_input, chain_ids, chain_output = chain
            gelu = helper.make_node(
                "Gelu",
                inputs=[gelu_input],
                outputs=[chain_output],
                name=f"fused_Gelu_{fusions}",
                domain=_MS_DOMAIN,
            )
            nodes_to_remove.update(chain_ids)
            # Insert at the position of the first chain node (the Div)
            replacements[chain_ids[0]] = gelu
            fusions += 1

        if fusions > 0:
            new_nodes = []
            for n in graph.node:
                nid = id(n)
                if nid in replacements:
                    new_nodes.append(replacements[nid])
                elif nid not in nodes_to_remove:
                    new_nodes.append(n)
            del graph.node[:]
            graph.node.extend(new_nodes)
            _ensure_ms_domain(model)

        after = len(graph.node)
        return model, PassResult(
            applied=fusions > 0,
            description=f"{fusions} GeLU chain(s) fused to com.microsoft.Gelu"
            if fusions > 0
            else "no GeLU pattern found",
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


def _build_consumer_map(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    """Maps each tensor name to the list of nodes that consume it."""
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for n in graph.node:
        for inp in n.input:
            if inp:
                consumers.setdefault(inp, []).append(n)
    return consumers


def _single_consumer(
    tensor: str,
    consumers: dict[str, list[onnx.NodeProto]],
    use_count: dict[str, int],
) -> onnx.NodeProto | None:
    """Return the single consumer of tensor, or None if use_count != 1."""
    if use_count.get(tensor, 0) != 1:
        return None
    nodes = consumers.get(tensor, [])
    return nodes[0] if nodes else None


def _match_gelu_chain(
    div_node: onnx.NodeProto,
    consumers: dict[str, list[onnx.NodeProto]],
    initializers: dict[str, onnx.TensorProto],
    output_use_count: dict[str, int],
) -> tuple[str, list[int], str] | None:
    """Return (gelu_input, [chain node ids], chain_output) or None."""
    if div_node.op_type != "Div":
        return None

    gelu_input = div_node.input[0]
    divisor_name = div_node.input[1]
    if not _is_scalar_near(divisor_name, initializers, _SQRT2):
        return None

    div_out = div_node.output[0]
    erf_node = _single_consumer(div_out, consumers, output_use_count)
    if erf_node is None or erf_node.op_type != "Erf":
        return None

    erf_out = erf_node.output[0]
    add1_node = _single_consumer(erf_out, consumers, output_use_count)
    if add1_node is None or add1_node.op_type != "Add":
        return None
    if not _one_input_is_scalar(add1_node, initializers, 1.0):
        return None

    add1_out = add1_node.output[0]
    mul1_node = _single_consumer(add1_out, consumers, output_use_count)
    if mul1_node is None or mul1_node.op_type != "Mul":
        return None
    if gelu_input not in mul1_node.input:
        return None

    mul1_out = mul1_node.output[0]
    mul2_node = _single_consumer(mul1_out, consumers, output_use_count)
    if mul2_node is None or mul2_node.op_type != "Mul":
        return None
    if not _one_input_is_scalar(mul2_node, initializers, 0.5):
        return None

    chain_ids = [id(div_node), id(erf_node), id(add1_node), id(mul1_node), id(mul2_node)]
    return gelu_input, chain_ids, mul2_node.output[0]


def _is_scalar_near(name: str, initializers: dict[str, onnx.TensorProto], value: float) -> bool:
    if name not in initializers:
        return False
    t = initializers[name]
    arr = numpy_helper.to_array(t).flatten()
    return arr.size == 1 and abs(float(arr[0]) - value) < _SQRT2_TOL


def _one_input_is_scalar(
    node: onnx.NodeProto,
    initializers: dict[str, onnx.TensorProto],
    value: float,
) -> bool:
    for inp in node.input:
        if _is_scalar_near(inp, initializers, value):
            return True
    return False


def _ensure_ms_domain(model: onnx.ModelProto) -> None:
    for opset in model.opset_import:
        if opset.domain == _MS_DOMAIN:
            return
    opset = model.opset_import.add()
    opset.domain = _MS_DOMAIN
    opset.version = 1
