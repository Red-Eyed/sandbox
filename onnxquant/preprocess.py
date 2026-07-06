from pathlib import Path

import onnx
from onnxruntime.quantization import quant_pre_process


def preprocess(
    model: str | Path | onnx.ModelProto,
    output_path: str | Path,
    *,
    skip_optimization: bool = False,
    skip_onnx_shape: bool = False,
    skip_symbolic_shape: bool = False,
    save_as_external_data: bool = False,
) -> None:
    """Shape-infer and clean up a graph before quantizing.

    Dynamo-exported graphs carry far more primitive ops (Expand/Slice/Reshape) around
    what used to be a single Conv/Gemm; skipping this step lets that noise reach the
    QDQ pattern-matcher and can produce missed or misplaced insertion points.
    """
    quant_pre_process(
        model,
        output_path,
        skip_optimization=skip_optimization,
        skip_onnx_shape=skip_onnx_shape,
        skip_symbolic_shape=skip_symbolic_shape,
        save_as_external_data=save_as_external_data,
    )
