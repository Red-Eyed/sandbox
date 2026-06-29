from collections import Counter
from dataclasses import dataclass, field

import onnx
from onnx import TensorProto

# Standard opset ops we know about — anything else in domain="" is flagged.
_KNOWN_OPS = {
    "Abs",
    "Acos",
    "Acosh",
    "Add",
    "And",
    "ArgMax",
    "ArgMin",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "AveragePool",
    "BatchNormalization",
    "BitShift",
    "Cast",
    "CastLike",
    "Ceil",
    "Clip",
    "Compress",
    "Concat",
    "ConcatFromSequence",
    "Constant",
    "ConstantOfShape",
    "Conv",
    "ConvInteger",
    "ConvTranspose",
    "Cos",
    "Cosh",
    "CumSum",
    "DepthToSpace",
    "DequantizeLinear",
    "Det",
    "Div",
    "Dropout",
    "DynamicQuantizeLinear",
    "Einsum",
    "Elu",
    "Equal",
    "Erf",
    "Exp",
    "Expand",
    "EyeLike",
    "Flatten",
    "Floor",
    "GRU",
    "Gather",
    "GatherElements",
    "GatherND",
    "Gemm",
    "GlobalAveragePool",
    "GlobalLpPool",
    "GlobalMaxPool",
    "Greater",
    "GreaterOrEqual",
    "HardSigmoid",
    "Hardmax",
    "Identity",
    "If",
    "IsInf",
    "IsNaN",
    "LRN",
    "LSTM",
    "LayerNormalization",
    "LeakyRelu",
    "Less",
    "LessOrEqual",
    "Log",
    "LogSoftmax",
    "Loop",
    "LpNormalization",
    "LpPool",
    "MatMul",
    "MatMulInteger",
    "Max",
    "MaxPool",
    "MaxUnpool",
    "Mean",
    "Min",
    "Mod",
    "Mul",
    "Multinomial",
    "Neg",
    "NonMaxSuppression",
    "NonZero",
    "Not",
    "OneHot",
    "Optional",
    "OptionalGetElement",
    "OptionalHasElement",
    "Or",
    "PRelu",
    "Pad",
    "Pow",
    "QLinearConv",
    "QLinearMatMul",
    "QuantizeLinear",
    "RNN",
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
    "Range",
    "Reciprocal",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceSumSquare",
    "Relu",
    "Reshape",
    "Resize",
    "ReverseSequence",
    "RoiAlign",
    "Round",
    "STFT",
    "Scan",
    "ScatterElements",
    "ScatterND",
    "Selu",
    "SequenceAt",
    "SequenceConstruct",
    "SequenceEmpty",
    "SequenceErase",
    "SequenceInsert",
    "SequenceLength",
    "Shape",
    "Shrink",
    "Sigmoid",
    "Sign",
    "Sin",
    "Sinh",
    "Size",
    "Slice",
    "Softmax",
    "Softplus",
    "Softsign",
    "SpaceToDepth",
    "Split",
    "SplitToSequence",
    "Sqrt",
    "Squeeze",
    "StringNormalizer",
    "Sub",
    "Sum",
    "Tan",
    "Tanh",
    "TfIdfVectorizer",
    "ThresholdedRelu",
    "Tile",
    "TopK",
    "Transpose",
    "Trilu",
    "Unique",
    "Unsqueeze",
    "Where",
    "Xor",
}

_LN_CHAIN = {"ReduceMean", "Sub", "Pow", "Add", "Sqrt", "Div", "Mul"}


@dataclass
class Finding:
    severity: str  # HIGH | MEDIUM | LOW
    title: str
    detail: str
    locations: list[str] = field(default_factory=list)
    guide: str | None = None


@dataclass
class GraphAnalysis:
    n_nodes: int
    n_params: int
    op_counts: dict[str, int]
    op_sequence_rle: list[str]
    dynamic_dims: list[str]
    unfused_patterns: list[Finding]
    red_flags: list[Finding]
    domain_ops: dict[str, list[str]]


def analyze(model: onnx.ModelProto) -> GraphAnalysis:
    graph = model.graph
    initializer_names = {i.name for i in graph.initializer}
    output_use_count = _build_output_use_count(graph)

    op_counts: dict[str, int] = Counter(n.op_type for n in graph.node)
    domain_ops = _collect_domain_ops(graph)
    n_params = _count_params(graph)
    dynamic_dims = _find_dynamic_dims(model)
    op_sequence_rle = _run_length_encode([n.op_type for n in graph.node])

    unfused_patterns: list[Finding] = []
    red_flags: list[Finding] = []

    _check_unfused_gemm(graph, initializer_names, output_use_count, unfused_patterns)
    _check_decomposed_layernorm(graph, red_flags)
    _check_redundant_cast(graph, red_flags)
    _check_layout_thrashing(graph, red_flags)
    _check_unknown_ops(graph, domain_ops, red_flags)

    return GraphAnalysis(
        n_nodes=len(graph.node),
        n_params=n_params,
        op_counts=dict(op_counts),
        op_sequence_rle=op_sequence_rle,
        dynamic_dims=dynamic_dims,
        unfused_patterns=unfused_patterns,
        red_flags=red_flags,
        domain_ops=domain_ops,
    )


def _build_output_use_count(graph: onnx.GraphProto) -> dict[str, int]:
    counts: dict[str, int] = Counter()
    for node in graph.node:
        for inp in node.input:
            if inp:
                counts[inp] += 1
    return dict(counts)


def _collect_domain_ops(graph: onnx.GraphProto) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for node in graph.node:
        domain = node.domain or ""
        if domain != "":
            result.setdefault(domain, []).append(node.op_type)
    return result


def _count_params(graph: onnx.GraphProto) -> int:
    total = 0
    for init in graph.initializer:
        count = 1
        for dim in init.dims:
            count *= dim
        total += count
    return total


def _find_dynamic_dims(model: onnx.ModelProto) -> list[str]:
    dynamic: list[str] = []
    for inp in model.graph.input:
        tensor_type = inp.type.tensor_type
        if tensor_type.HasField("shape"):
            for dim in tensor_type.shape.dim:
                if dim.dim_param and dim.dim_param not in dynamic:
                    dynamic.append(dim.dim_param)
    return dynamic


def _run_length_encode(seq: list[str]) -> list[str]:
    if not seq:
        return []
    result = []
    current, count = seq[0], 1
    for item in seq[1:]:
        if item == current:
            count += 1
        else:
            result.append(f"{current}×{count}" if count > 1 else current)
            current, count = item, 1
    result.append(f"{current}×{count}" if count > 1 else current)
    return result


def _check_unfused_gemm(
    graph: onnx.GraphProto,
    initializer_names: set[str],
    output_use_count: dict[str, int],
    findings: list[Finding],
) -> None:
    matmul_outputs: dict[str, str] = {}  # output_name → node_name
    for node in graph.node:
        if node.op_type == "MatMul":
            if len(node.output) == 1:
                matmul_outputs[node.output[0]] = node.name or node.op_type

    locations = []
    for node in graph.node:
        if node.op_type != "Add":
            continue
        for idx, inp in enumerate(node.input):
            if inp in matmul_outputs and output_use_count.get(inp, 0) == 1:
                bias_inp = node.input[1 - idx]
                if bias_inp in initializer_names:
                    locations.append(matmul_outputs[inp])

    if locations:
        findings.append(
            Finding(
                severity="MEDIUM",
                title="Unfused MatMul+Add (potential Gemm)",
                detail=f"Found {len(locations)} MatMul→Add pairs with constant bias. "
                "fuse_gemm pass will handle these automatically.",
                locations=locations,
            )
        )


def _check_decomposed_layernorm(graph: onnx.GraphProto, findings: list[Finding]) -> None:
    op_types = [n.op_type for n in graph.node]
    count = sum(1 for op in op_types if op == "ReduceMean")
    # Heuristic: if ReduceMean, Sub, Pow all appear, LN chain likely present
    if count > 0 and "Sub" in op_types and "Pow" in op_types and "Sqrt" in op_types:
        findings.append(
            Finding(
                severity="HIGH",
                title="Decomposed LayerNorm detected",
                detail=f"Found {count} ReduceMean node(s) with Sub/Pow/Sqrt chain — likely "
                "torch-decomposed LayerNorm. fuse_layernorm pass will attempt fusion.",
                guide="docs/layernorm_fusion.md",
            )
        )


def _check_redundant_cast(graph: onnx.GraphProto, findings: list[Finding]) -> None:
    float_types = {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE}
    locations = []
    for node in graph.node:
        if node.op_type != "Cast":
            continue
        to_attr = next((a for a in node.attribute if a.name == "to"), None)
        if to_attr and to_attr.i in float_types:
            locations.append(node.name or "Cast")

    if locations:
        findings.append(
            Finding(
                severity="LOW",
                title="Possible redundant Cast nodes (float-width only)",
                detail=f"{len(locations)} Cast node(s) change only float width. "
                "eliminate_cast pass will verify and remove unnecessary ones.",
                locations=locations,
            )
        )


def _check_layout_thrashing(graph: onnx.GraphProto, findings: list[Finding]) -> None:
    pairs = 0
    nodes = list(graph.node)
    for i in range(len(nodes) - 1):
        if nodes[i].op_type == "Reshape" and nodes[i + 1].op_type == "Transpose":
            pairs += 1
    if pairs > 2:
        findings.append(
            Finding(
                severity="MEDIUM",
                title="Repeated Reshape→Transpose pairs",
                detail=f"Found {pairs} consecutive Reshape→Transpose pairs — possible layout "
                "thrashing. Consider fusing or reordering in the PyTorch model.",
            )
        )


def _check_unknown_ops(
    graph: onnx.GraphProto,
    domain_ops: dict[str, list[str]],
    findings: list[Finding],
) -> None:
    unknown = [
        node.op_type
        for node in graph.node
        if (node.domain or "") == "" and node.op_type not in _KNOWN_OPS
    ]
    if unknown:
        findings.append(
            Finding(
                severity="HIGH",
                title="Unknown ops in standard domain",
                detail=f"Ops not in known opset: {', '.join(set(unknown))}. "
                "These may not be supported by XNNPACK.",
                locations=list(set(unknown)),
            )
        )
