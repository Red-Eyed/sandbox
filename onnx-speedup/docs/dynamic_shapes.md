# Dynamic Shapes Guide

## Why dynamic shapes hurt on XNNPACK

XNNPACK prepacks weight tensors into a layout optimized for a specific input shape.
When the input shape changes (e.g. a different sequence length), XNNPACK must repack
on every call — often costing more than the computation itself.

## Bucketing strategy

Instead of one model with a fully dynamic `seq_len`, export N models with fixed
`seq_len` values that cover the real distribution, then dispatch at runtime:

```python
BUCKETS = [8, 16, 32, 64, 128]

def pick_bucket(seq_len: int) -> int:
    for b in BUCKETS:
        if seq_len <= b:
            return b
    return BUCKETS[-1]   # clamp to max

def export_buckets(model: torch.nn.Module, path_template: str):
    for seq_len in BUCKETS:
        dummy = torch.zeros(1, seq_len, model.d_model)
        torch.onnx.export(model, dummy, path_template.format(seq_len))
```

Android inference then pads the input to the next bucket length and picks the
correct `.onnx` file.

## Fixing unbounded dynamic dims

To convert an exported model from dynamic to static for a specific bucket:

```python
import onnx
from onnx.tools import update_model_dims

model = onnx.load("model.onnx")
model = update_model_dims.update_inputs_outputs_dims(
    model,
    input_dims={"x": [1, 32, 64]},   # batch=1, seq=32, d_model=64
    output_dims={"out": [1, 32, 64]},
)
onnx.save(model, "model_seq32.onnx")
```

Then run `speedup.py` on each static variant for maximum fusion.
