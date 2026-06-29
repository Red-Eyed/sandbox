# EP Assignment Guide — Verifying XNNPACK Delegation

## Collecting an ORT profile

```python
import onnxruntime as ort

so = ort.SessionOptions()
so.enable_profiling = True
sess = ort.InferenceSession("model.onnx", so,
    providers=["XNNPACKExecutionProvider", "CPUExecutionProvider"])

sess.run(None, feeds)
profile_path = sess.end_profiling()   # e.g. "onnxruntime_profile_2024-01-01.json"
```

Pass this file to the tool:
```bash
speedup model.onnx --profile onnxruntime_profile_2024-01-01.json
```

## Reading the EP breakdown

The report's "Profile Analysis" section shows `xnnpack_pct` and `cpu_fallback_pct`.
A healthy deployment should be >85% XNNPACK.

## Common MLAS fallback causes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Cast` nodes at runtime | dtype mismatch (f32 input, f16 weight) | eliminate_cast pass or fix model dtypes |
| `Softmax` on MLAS | XNNPACK only delegates Softmax on certain shapes | ensure softmax is over the last dim |
| `Gather` / `Slice` on MLAS | XNNPACK does not support these | unavoidable; minimize their count |
| `Reshape` on MLAS | XNNPACK delegates reshape only for specific patterns | check if onnxslim can eliminate them |
| Any op > opset 17 on MLAS | XNNPACK lags behind opset support | pin export to opset 17 |

## Forcing XNNPACK EP in ORT

On Android, set the provider order explicitly in the Android ORT API:

```kotlin
val options = OrtSession.SessionOptions()
options.addXnnpack(mapOf())   // must be first
session = OrtEnvironment.getEnvironment()
    .createSession(modelBytes, options)
```

Without `addXnnpack()`, ORT defaults to MLAS for everything.
