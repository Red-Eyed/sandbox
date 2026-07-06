# Examples

`basic_usage.py` runs the full pipeline end to end against a tiny toy conv net:

1. Build the model and export it with `torch.onnx.export(..., dynamo=True)`.
2. Quantize it with `onnxquant.quantize()`, using a synthetic `CalibrationDataReader`
   (`synthetic_calibration_reader.py`) standing in for a real dataset.
3. Compare FP32 vs INT8 outputs with `onnxquant.compare_outputs()`.

Run from the repo root:

```bash
uv run python -m examples.basic_usage
```

`synthetic_calibration_reader.py` is the part every real user replaces — swap it for a
reader that yields your own in-distribution samples. Everything else in this example is
the library's real public API.
