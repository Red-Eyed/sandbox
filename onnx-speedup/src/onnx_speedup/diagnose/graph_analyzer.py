from collections import Counter
from dataclasses import dataclass, field

import onnx

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
    findings: list[Finding]
    domain_ops: dict[str, list[str]]


def analyze(model: onnx.ModelProto) -> GraphAnalysis:
    graph = model.graph

    op_counts: dict[str, int] = Counter(n.op_type for n in graph.node)
    domain_ops = _collect_domain_ops(graph)
    n_params = _count_params(graph)
    dynamic_dims = _find_dynamic_dims(model)
    op_sequence_rle = _run_length_encode([n.op_type for n in graph.node])

    findings: list[Finding] = []
    _check_layout_thrashing(graph, findings)
    _check_unknown_ops(graph, domain_ops, findings)

    return GraphAnalysis(
        n_nodes=len(graph.node),
        n_params=n_params,
        op_counts=dict(op_counts),
        op_sequence_rle=op_sequence_rle,
        dynamic_dims=dynamic_dims,
        findings=findings,
        domain_ops=domain_ops,
    )


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
