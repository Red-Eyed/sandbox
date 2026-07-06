import tempfile
from pathlib import Path

import onnx
from onnxruntime.quantization import get_qdq_config
from onnxruntime.quantization import quantize as ort_quantize

from onnxquant.config import QuantizationConfig
from onnxquant.preprocess import preprocess
from onnxquant.protocols import CalibrationDataReader


def quantize(
    model: str | Path | onnx.ModelProto,
    output_path: str | Path,
    calibration_data_reader: CalibrationDataReader,
    config: QuantizationConfig | None = None,
) -> None:
    """Static W8A8 QDQ PTQ: preprocess, then quantize, writing to output_path."""
    config = config or QuantizationConfig()
    with tempfile.TemporaryDirectory() as tmp_dir:
        preprocessed_path = Path(tmp_dir) / "preprocessed.onnx"
        preprocess(model, preprocessed_path)
        qdq_config = get_qdq_config(
            preprocessed_path,
            calibration_data_reader,
            **config.to_get_qdq_config_kwargs(),
        )
        ort_quantize(preprocessed_path, output_path, qdq_config)
