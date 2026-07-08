from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
from pydantic import BaseModel

from torchquant.config import Pt2eQuantizationConfig

_QUANTIZE_OP = "QuantizeLinear"
_DEQUANTIZE_OP = "DequantizeLinear"


class ModelStats(BaseModel):
    op_type_counts: dict[str, int]
    size_bytes: int


def compute_model_stats(model_path: str | Path) -> ModelStats:
    proto = onnx.load(str(model_path))
    op_type_counts = Counter(node.op_type for node in proto.graph.node)
    return ModelStats(
        op_type_counts=dict(op_type_counts), size_bytes=Path(model_path).stat().st_size
    )


class OutputDiff(BaseModel):
    output_name: str
    max_abs_diff: float
    mean_abs_diff: float


class ParityReport(BaseModel):
    """Raw numeric output diffs — domain-agnostic, no notion of labels or accuracy."""

    num_samples: int
    per_output: list[OutputDiff]


def compare_outputs(
    fp32_model: str | Path,
    int8_model: str | Path,
    samples: Iterable[dict[str, np.ndarray]],
) -> ParityReport:
    fp32_session = ort.InferenceSession(str(fp32_model), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(int8_model), providers=["CPUExecutionProvider"])
    # The two graphs come from different export pipelines (fp32 straight from the module,
    # int8 through prepare_pt2e/convert_pt2e), so their output tensor names differ even
    # for "the same" output — pair outputs by position, and label with the fp32 name.
    output_names = [output.name for output in fp32_session.get_outputs()]
    int8_output_names = [output.name for output in int8_session.get_outputs()]

    max_abs_diff = dict.fromkeys(output_names, 0.0)
    sum_abs_diff = dict.fromkeys(output_names, 0.0)
    num_samples = 0

    for feed in samples:
        fp32_outputs = cast("list[np.ndarray]", fp32_session.run(output_names, feed))
        int8_outputs = cast("list[np.ndarray]", int8_session.run(int8_output_names, feed))
        for name, fp32_out, int8_out in zip(output_names, fp32_outputs, int8_outputs, strict=True):
            abs_diff = np.abs(fp32_out.astype(np.float64) - int8_out.astype(np.float64))
            max_abs_diff[name] = max(max_abs_diff[name], float(abs_diff.max()))
            sum_abs_diff[name] += float(abs_diff.mean())
        num_samples += 1

    if num_samples == 0:
        raise ValueError("compare_outputs received no samples")

    per_output = [
        OutputDiff(
            output_name=name,
            max_abs_diff=max_abs_diff[name],
            mean_abs_diff=sum_abs_diff[name] / num_samples,
        )
        for name in output_names
    ]
    return ParityReport(num_samples=num_samples, per_output=per_output)


class OpQuantizationCoverage(BaseModel):
    """Op types whose input is a DequantizeLinear output — entry points, not full coverage."""

    quantized_op_counts: dict[str, int]
    unquantized_op_counts: dict[str, int]


def compute_op_quantization_coverage(model_path: str | Path) -> OpQuantizationCoverage:
    proto = onnx.load(str(model_path))
    producer_op_type = {
        output_name: node.op_type for node in proto.graph.node for output_name in node.output
    }

    quantized_counts: Counter[str] = Counter()
    unquantized_counts: Counter[str] = Counter()
    for node in proto.graph.node:
        if node.op_type in (_QUANTIZE_OP, _DEQUANTIZE_OP):
            continue
        consumes_dequantized = any(
            producer_op_type.get(input_name) == _DEQUANTIZE_OP for input_name in node.input
        )
        counts = quantized_counts if consumes_dequantized else unquantized_counts
        counts[node.op_type] += 1

    return OpQuantizationCoverage(
        quantized_op_counts=dict(quantized_counts),
        unquantized_op_counts=dict(unquantized_counts),
    )


class QuantizationReport(BaseModel):
    config: Pt2eQuantizationConfig
    fp32_stats: ModelStats
    int8_stats: ModelStats
    parity: ParityReport
    op_coverage: OpQuantizationCoverage

    @property
    def size_reduction_ratio(self) -> float:
        return self.fp32_stats.size_bytes / self.int8_stats.size_bytes

    def to_json_str(self) -> str:
        return self.model_dump_json(indent=2)

    def to_markdown(self) -> str:
        lines = [
            "# Quantization Report",
            "",
            "## Size",
            f"- FP32: {self.fp32_stats.size_bytes:,} bytes",
            f"- INT8: {self.int8_stats.size_bytes:,} bytes",
            f"- Reduction: {self.size_reduction_ratio:.2f}x",
            "",
            "## Parity",
            f"Compared on {self.parity.num_samples} samples.",
            "",
            "| output | max_abs_diff | mean_abs_diff |",
            "|---|---|---|",
        ]
        for output_diff in self.parity.per_output:
            lines.append(
                f"| {output_diff.output_name} | {output_diff.max_abs_diff:.6f} | "
                f"{output_diff.mean_abs_diff:.6f} |"
            )

        lines += [
            "",
            "## Op quantization coverage",
            "| op_type | quantized | unquantized |",
            "|---|---|---|",
        ]
        all_op_types = sorted(
            set(self.op_coverage.quantized_op_counts) | set(self.op_coverage.unquantized_op_counts)
        )
        for op_type in all_op_types:
            quantized = self.op_coverage.quantized_op_counts.get(op_type, 0)
            unquantized = self.op_coverage.unquantized_op_counts.get(op_type, 0)
            lines.append(f"| {op_type} | {quantized} | {unquantized} |")

        return "\n".join(lines)


def build_report(
    fp32_model: str | Path,
    int8_model: str | Path,
    config: Pt2eQuantizationConfig,
    parity_samples: Iterable[dict[str, np.ndarray]],
) -> QuantizationReport:
    return QuantizationReport(
        config=config,
        fp32_stats=compute_model_stats(fp32_model),
        int8_stats=compute_model_stats(int8_model),
        parity=compare_outputs(fp32_model, int8_model, parity_samples),
        op_coverage=compute_op_quantization_coverage(int8_model),
    )
