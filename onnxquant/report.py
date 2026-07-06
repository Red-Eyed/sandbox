from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import onnx
from pydantic import BaseModel

from onnxquant.absent import NotComputed
from onnxquant.config import QuantizationConfig
from onnxquant.parity import ParityReport, compare_outputs


class ModelStats(BaseModel):
    op_type_counts: dict[str, int]
    size_bytes: int


def _load_model_proto(model: str | Path | onnx.ModelProto) -> onnx.ModelProto:
    if isinstance(model, onnx.ModelProto):
        return model
    return onnx.load(str(model))


def compute_model_stats(model: str | Path | onnx.ModelProto) -> ModelStats:
    proto = _load_model_proto(model)
    op_type_counts = Counter(node.op_type for node in proto.graph.node)
    return ModelStats(op_type_counts=dict(op_type_counts), size_bytes=proto.ByteSize())


class QuantizationReport(BaseModel):
    config: QuantizationConfig
    fp32_stats: ModelStats
    int8_stats: ModelStats
    parity: ParityReport | NotComputed

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
            "## Op type counts",
            "| op_type | fp32 | int8 |",
            "|---|---|---|",
        ]
        all_op_types = sorted(
            set(self.fp32_stats.op_type_counts) | set(self.int8_stats.op_type_counts)
        )
        for op_type in all_op_types:
            fp32_count = self.fp32_stats.op_type_counts.get(op_type, 0)
            int8_count = self.int8_stats.op_type_counts.get(op_type, 0)
            lines.append(f"| {op_type} | {fp32_count} | {int8_count} |")

        lines += ["", "## Parity"]
        if isinstance(self.parity, NotComputed):
            lines.append(f"Not computed: {self.parity.reason}")
        else:
            lines.append(f"Compared on {self.parity.num_samples} samples.")
            lines += ["", "| output | max_abs_diff | mean_abs_diff |", "|---|---|---|"]
            for output_diff in self.parity.per_output:
                lines.append(
                    f"| {output_diff.output_name} | {output_diff.max_abs_diff:.6f} | "
                    f"{output_diff.mean_abs_diff:.6f} |"
                )

        return "\n".join(lines)


def build_report(
    fp32_model: str | Path,
    int8_model: str | Path,
    config: QuantizationConfig,
    parity_samples: Iterable[dict[str, np.ndarray]] | None = None,
) -> QuantizationReport:
    parity: ParityReport | NotComputed
    if parity_samples is None:
        parity = NotComputed(reason="no parity_samples provided to build_report()")
    else:
        parity = compare_outputs(fp32_model, int8_model, parity_samples)

    return QuantizationReport(
        config=config,
        fp32_stats=compute_model_stats(fp32_model),
        int8_stats=compute_model_stats(int8_model),
        parity=parity,
    )
