from collections.abc import Iterable
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort
from pydantic import BaseModel


class OutputDiff(BaseModel):
    output_name: str
    max_abs_diff: float
    mean_abs_diff: float


class ParityReport(BaseModel):
    num_samples: int
    per_output: list[OutputDiff]


def compare_outputs(
    fp32_model: str | Path,
    int8_model: str | Path,
    samples: Iterable[dict[str, np.ndarray]],
) -> ParityReport:
    """Compare two ONNX models' outputs over caller-supplied samples.

    Deliberately domain-agnostic (raw numeric diff only, no top-1/label logic) since
    this library has no knowledge of what the model's outputs mean — a classifier can
    layer its own agreement check on top of the returned per-output diffs.
    """
    fp32_session = ort.InferenceSession(str(fp32_model), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(int8_model), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in fp32_session.get_outputs()]

    max_abs_diff = dict.fromkeys(output_names, 0.0)
    sum_abs_diff = dict.fromkeys(output_names, 0.0)
    num_samples = 0

    for feed in samples:
        # session.run() is typed to also allow SparseTensor/dict/list outputs (ONNX's
        # sequence/map types); this library only supports models with dense tensor
        # outputs, so the cast documents that boundary rather than working around it.
        fp32_outputs = cast("list[np.ndarray]", fp32_session.run(output_names, feed))
        int8_outputs = cast("list[np.ndarray]", int8_session.run(output_names, feed))
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
