import tempfile
from pathlib import Path

from examples.synthetic_calibration_reader import SyntheticCalibrationReader
from examples.toy_model import INPUT_SHAPE, export_to_onnx
from onnxquant import QuantizationConfig, build_report, compute_layer_error, quantize


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        fp32_path = tmp / "model_fp32.onnx"
        int8_path = tmp / "model_int8.onnx"

        export_to_onnx(str(fp32_path))

        config = QuantizationConfig()
        calibration_reader = SyntheticCalibrationReader(
            input_name="input", input_shape=INPUT_SHAPE, num_samples=32
        )
        quantize(fp32_path, int8_path, calibration_reader, config)

        validation_reader = SyntheticCalibrationReader(
            input_name="input", input_shape=INPUT_SHAPE, num_samples=8, seed=1
        )
        report = build_report(fp32_path, int8_path, config, parity_samples=validation_reader)
        print(report.to_markdown())

        layer_error_report = compute_layer_error(
            fp32_path,
            int8_path,
            make_samples=lambda: SyntheticCalibrationReader(
                input_name="input", input_shape=INPUT_SHAPE, num_samples=8, seed=2
            ),
        )
        print()
        print(layer_error_report.to_markdown())


if __name__ == "__main__":
    main()
