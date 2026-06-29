"""
Fuse torch-decomposed LayerNorm into LayerNormalization.

Torch export decomposes nn.LayerNorm to:
  ReduceMean(x, axes) → Sub(x, mean) → Pow(_, 2) → ReduceMean(_, axes)
  → Add(_, eps) → Sqrt → Div(sub_out, _) → Mul(_, weight) → Add(_, bias)

We walk each ReduceMean node and attempt to match the full chain forward.
On any mismatch we skip without partial rewrite.
"""

import copy

import onnx
from onnx import helper, numpy_helper

from onnx_speedup.passes.base import Pass, PassResult


class FuseLayerNormPass(Pass):
    name = "fuse_layernorm"
    description = "Fuse decomposed LayerNorm chain → LayerNormalization"

    def apply(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, PassResult]:
        model = copy.deepcopy(model)
        graph = model.graph
        before = len(graph.node)

        initializers = {i.name: i for i in graph.initializer}
        consumers = _build_consumer_map(graph)
        output_use_count = _count_uses(graph)

        fusions = 0
        nodes_to_remove: set[int] = set()
        # Maps first-chain-node-id → replacement LayerNormalization node
        replacements: dict[int, onnx.NodeProto] = {}

        for node in list(graph.node):
            if id(node) in nodes_to_remove or node.op_type != "ReduceMean":
                continue
            match = _match_ln_chain(node, consumers, initializers, output_use_count)
            if match is None:
                continue
            ln_input, axes, eps, weight_name, bias_name, chain_ids, ln_output = match
            ln_node = helper.make_node(
                "LayerNormalization",
                inputs=[ln_input, weight_name, bias_name],
                outputs=[ln_output],
                name=f"fused_LayerNorm_{fusions}",
                axis=axes[0] if len(axes) == 1 else -1,
                epsilon=eps,
            )
            nodes_to_remove.update(chain_ids)
            # Insert at the position of reduce_mean1 (first chain node) to keep topo order.
            replacements[chain_ids[0]] = ln_node
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

        after = len(graph.node)
        return model, PassResult(
            applied=fusions > 0,
            description=f"{fusions} LayerNorm chain(s) fused to LayerNormalization"
            if fusions > 0
            else "pattern not matched (see report)",
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
    if use_count.get(tensor, 0) != 1:
        return None
    nodes = consumers.get(tensor, [])
    return nodes[0] if nodes else None


def _match_ln_chain(
    reduce_mean1: onnx.NodeProto,
    consumers: dict[str, list[onnx.NodeProto]],
    initializers: dict[str, onnx.TensorProto],
    output_use_count: dict[str, int],
) -> tuple[str, list[int], float, str, str, list[int], str] | None:
    """
    Returns (ln_input, axes, eps, weight_name, bias_name, chain_node_ids, ln_output)
    or None on mismatch.
    """
    ln_input = reduce_mean1.input[0]
    axes = _get_axes(reduce_mean1)
    if not axes:
        return None

    mean_out = reduce_mean1.output[0]
    sub_node = _single_consumer(mean_out, consumers, output_use_count)
    if sub_node is None or sub_node.op_type != "Sub":
        return None
    if sub_node.input[0] != ln_input or sub_node.input[1] != mean_out:
        return None

    sub_out = sub_node.output[0]
    if output_use_count.get(sub_out, 0) != 2:  # consumed by Pow and Div
        return None

    # Find Pow among the two consumers of sub_out
    pow_node = next((n for n in consumers.get(sub_out, []) if n.op_type == "Pow"), None)
    if pow_node is None or pow_node.input[0] != sub_out:
        return None
    if not _is_scalar_value(pow_node.input[1], initializers, 2.0):
        return None

    pow_out = pow_node.output[0]
    reduce_mean2 = _single_consumer(pow_out, consumers, output_use_count)
    if reduce_mean2 is None or reduce_mean2.op_type != "ReduceMean":
        return None
    if _get_axes(reduce_mean2) != axes:
        return None

    var_out = reduce_mean2.output[0]
    add_eps_node = _single_consumer(var_out, consumers, output_use_count)
    if add_eps_node is None or add_eps_node.op_type != "Add":
        return None
    eps, eps_ok = _extract_scalar_from_add(add_eps_node, var_out, initializers)
    if not eps_ok:
        return None

    eps_out = add_eps_node.output[0]
    sqrt_node = _single_consumer(eps_out, consumers, output_use_count)
    if sqrt_node is None or sqrt_node.op_type != "Sqrt":
        return None

    sqrt_out = sqrt_node.output[0]
    div_node = _single_consumer(sqrt_out, consumers, output_use_count)
    if div_node is None or div_node.op_type != "Div":
        return None
    if div_node.input[0] != sub_out or div_node.input[1] != sqrt_out:
        return None

    div_out = div_node.output[0]
    mul_node = _single_consumer(div_out, consumers, output_use_count)
    if mul_node is None or mul_node.op_type != "Mul":
        return None
    weight_name = _other_input(mul_node, div_out)
    if weight_name is None or weight_name not in initializers:
        return None

    mul_out = mul_node.output[0]
    add_bias_node = _single_consumer(mul_out, consumers, output_use_count)
    if add_bias_node is None or add_bias_node.op_type != "Add":
        return None
    bias_name = _other_input(add_bias_node, mul_out)
    if bias_name is None or bias_name not in initializers:
        return None

    chain_ids = [
        id(reduce_mean1),
        id(sub_node),
        id(pow_node),
        id(reduce_mean2),
        id(add_eps_node),
        id(sqrt_node),
        id(div_node),
        id(mul_node),
        id(add_bias_node),
    ]
    return (
        ln_input,
        axes,
        eps,
        weight_name,
        bias_name,
        chain_ids,
        add_bias_node.output[0],
    )


def _get_axes(node: onnx.NodeProto) -> list[int]:
    for attr in node.attribute:
        if attr.name == "axes":
            return list(attr.ints)
    return []


def _is_scalar_value(name: str, initializers: dict[str, onnx.TensorProto], value: float) -> bool:
    if name not in initializers:
        return False
    arr = numpy_helper.to_array(initializers[name]).flatten()
    return arr.size == 1 and abs(float(arr[0]) - value) < 1e-5


def _extract_scalar_from_add(
    node: onnx.NodeProto,
    non_scalar_inp: str,
    initializers: dict[str, onnx.TensorProto],
) -> tuple[float, bool]:
    for inp in node.input:
        if inp == non_scalar_inp:
            continue
        if inp in initializers:
            arr = numpy_helper.to_array(initializers[inp]).flatten()
            if arr.size == 1:
                return float(arr[0]), True
    return 0.0, False


def _other_input(node: onnx.NodeProto, known_input: str) -> str | None:
    for inp in node.input:
        if inp != known_input:
            return inp
    return None
