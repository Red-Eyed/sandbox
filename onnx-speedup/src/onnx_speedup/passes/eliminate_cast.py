import onnx
from onnx import TensorProto, helper

from onnx_speedup.passes.base import Pass, PassResult

# These float-width pairs are considered redundant when they carry the same semantic width
# on ARM NEON/XNNPACK (f32 → f32 is always redundant; f16 → f16 likewise).
_NOOP_PAIRS: set[tuple[int, int]] = {
    (TensorProto.FLOAT, TensorProto.FLOAT),
    (TensorProto.FLOAT16, TensorProto.FLOAT16),
    (TensorProto.BFLOAT16, TensorProto.BFLOAT16),
    (TensorProto.DOUBLE, TensorProto.DOUBLE),
}


class EliminateCastPass(Pass):
    name = "eliminate_cast"
    description = "Remove Cast nodes whose input and output dtype are identical"

    def apply(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, PassResult]:
        graph = model.graph
        before = len(graph.node)

        # Build value_info dtype map (initializers + graph inputs + intermediate)
        dtype_map = _build_dtype_map(graph)

        # Identify noop Cast nodes
        noop_casts: dict[str, str] = {}  # cast_output → cast_input (bypass)
        for node in graph.node:
            if node.op_type != "Cast":
                continue
            to_attr = next((a for a in node.attribute if a.name == "to"), None)
            if to_attr is None:
                continue
            dst_dtype = to_attr.i
            src_dtype = dtype_map.get(node.input[0])
            if src_dtype is not None and (src_dtype, dst_dtype) in _NOOP_PAIRS:
                noop_casts[node.output[0]] = node.input[0]

        if not noop_casts:
            return model, PassResult(
                applied=False,
                description="no redundant Cast nodes found",
                nodes_before=before,
                nodes_after=before,
            )

        new_nodes = _rewrite_nodes(graph, noop_casts)
        new_graph = helper.make_graph(
            new_nodes,
            graph.name,
            list(graph.input),
            list(graph.output),
            list(graph.initializer),
        )
        new_model = helper.make_model(new_graph, opset_imports=list(model.opset_import))
        new_model.ir_version = model.ir_version
        after = len(new_nodes)
        removed = before - after
        return new_model, PassResult(
            applied=True,
            description=f"{removed} redundant Cast node(s) removed",
            nodes_before=before,
            nodes_after=after,
        )


def _build_dtype_map(graph: onnx.GraphProto) -> dict[str, int]:
    dtype_map: dict[str, int] = {}
    for init in graph.initializer:
        dtype_map[init.name] = init.data_type
    for inp in graph.input:
        tt = inp.type.tensor_type
        if tt.HasField("elem_type"):
            dtype_map[inp.name] = tt.elem_type
    for vi in graph.value_info:
        tt = vi.type.tensor_type
        if tt.HasField("elem_type"):
            dtype_map[vi.name] = tt.elem_type
    return dtype_map


def _rewrite_nodes(
    graph: onnx.GraphProto,
    noop_casts: dict[str, str],
) -> list[onnx.NodeProto]:
    new_nodes = []
    for node in graph.node:
        if node.op_type == "Cast" and node.output[0] in noop_casts:
            continue  # drop this node
        # Remap inputs that referenced a dropped Cast output
        new_inputs = [noop_casts.get(inp, inp) for inp in node.input]
        new_node = helper.make_node(
            node.op_type,
            inputs=new_inputs,
            outputs=list(node.output),
            name=node.name,
            domain=node.domain,
            **{a.name: _attr_value(a) for a in node.attribute},
        )
        new_nodes.append(new_node)
    return new_nodes


def _attr_value(attr: onnx.AttributeProto):
    if attr.type == onnx.AttributeProto.FLOAT:
        return attr.f
    if attr.type == onnx.AttributeProto.INT:
        return attr.i
    if attr.type == onnx.AttributeProto.STRING:
        return attr.s
    if attr.type == onnx.AttributeProto.TENSOR:
        return attr.t
    if attr.type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    if attr.type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    if attr.type == onnx.AttributeProto.STRINGS:
        return list(attr.strings)
    if attr.type == onnx.AttributeProto.GRAPH:
        return attr.g
    return attr
