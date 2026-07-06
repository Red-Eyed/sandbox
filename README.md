# onnxquant

W8A8 (int8 weights, int8 activations) static post-training quantization for ONNX
models exported via `torch.onnx.export(..., dynamo=True)`.

**This is a demo/experimentation project, not a published or production-ready package.**
It exists to work through the current (2026) state of ONNX W8A8 PTQ against dynamo-decomposed
graphs hands-on — read it, run it, fork it, but don't depend on it as-is.

## Why this exists

PyTorch's modern quantization flow (`torch.export` → `prepare_pt2e`/`convert_pt2e`) quantizes
*before* export, but exporting an already-quantized graph through the dynamo-based ONNX
exporter is currently broken (no ONNX decomposition for `quantized_decomposed.*`/`FakeQuantize`
ops — see [pytorch/pytorch#167063](https://github.com/pytorch/pytorch/issues/167063)). This
library takes the reliable path instead: export a plain FP32 ONNX graph first, then quantize at
the ONNX level with `onnxruntime.quantization`. Dynamo also decomposes graphs far more than the
legacy exporter did (a `Linear` or `LayerNorm` becomes many primitive ops), so preprocessing
(shape inference + graph cleanup) before quantizing isn't optional here the way it might be
for a TorchScript-exported graph.

## Design

- **Protocol, not a data-loading layer.** Calibration and validation data are supplied by the
  caller via `onnxquant.CalibrationDataReader` (re-exported from `onnxruntime.quantization`,
  which already structurally matches any object with `get_next()`). This library never reads
  a dataset off disk itself.
- **Symmetric int8 by default.** `QuantizationConfig` defaults to `QInt8`/`QInt8`, symmetric,
  per-channel weights — true W8A8, overriding onnxruntime's own default of asymmetric uint8
  activations (an x86-VNNI convention this project doesn't target; the generic ARM/XNNPACK
  CPU EP is the assumed deployment target).
- **`get_qdq_config()`, not hand-built `extra_options`.** It auto-scans the model for
  quantizable op types instead of relying on a hardcoded allowlist, which matters once
  decomposition changes which ops are actually present in the graph.

## Install

```bash
uv sync --group dev
```

## Usage

```python
from onnxquant import CalibrationDataReader, QuantizationConfig, build_report, quantize

class MyCalibrationReader(CalibrationDataReader):
    def get_next(self) -> dict[str, np.ndarray] | None:
        ...  # yield real in-distribution samples, or None when exhausted

config = QuantizationConfig()
quantize("model_fp32.onnx", "model_int8.onnx", MyCalibrationReader(), config)

report = build_report("model_fp32.onnx", "model_int8.onnx", config, parity_samples=MyCalibrationReader())
print(report.to_markdown())
```

See [examples/](examples/) for a full runnable pipeline — a hybrid Conv+SDPA+LayerNorm+GELU
toy model exported with dynamo, quantized, and reported on end to end:

```bash
uv run python -m examples.basic_usage
```

## Development

```bash
just check   # format-check + lint + types + tests
just fix     # format + lint-fix, then check
just hooks   # install prek git hooks
```
