# torchquant

W8A8 static post-training quantization done on the PyTorch side, then exported to ONNX.
Built as a hands-on way to understand quantization mechanics, not just to run a pipeline.

This is a sibling experiment to the `quantization` branch's `onnxquant`, which quantizes
an already-exported ONNX graph via `onnxruntime.quantization`. That approach kept hitting
bugs in onnxruntime's tooling when fed graphs from `torch.onnx.export(dynamo=True)` — the
decomposed, primitive-op-heavy shape of a dynamo-exported graph doesn't always match what
onnxruntime's QDQ pattern-matcher expects.

`torchquant` sidesteps that entirely by quantizing *before* export: it uses PyTorch 2
Export Quantization (PT2E) — `prepare_pt2e` / `convert_pt2e` — to insert observers,
calibrate, and bake `QuantizeLinear`/`DequantizeLinear`-equivalent ops directly into the
FX graph. Only then does it call `torch.onnx.export`, which translates that already-quantized
graph into ONNX QDQ nodes.

## Usage

```python
import torch
from torch import nn

from torchquant import Pt2eQuantizationConfig, quantize_torch_model

model = MyModel().eval()
example_args = (torch.randn(2, 3, 224, 224),)
# Needed for anything but a fixed batch size — see "Dynamic shapes" below.
dynamic_shapes = ({0: torch.export.Dim("batch")},)

def calibrate(prepared: nn.Module) -> None:
    for batch in my_calibration_loader():
        prepared(batch)

quantize_torch_model(
    model, example_args, calibrate, "model_int8.onnx",
    config=Pt2eQuantizationConfig.static_per_channel(),
    dynamic_shapes=dynamic_shapes,
)
```

`calibrate` is a closure you supply — it owns the calibration dataset and loop, so this
library never needs to know your data's shape or source.

## Presets

`Pt2eQuantizationConfig` has four named constructors, covering the two real axes
XNNPACKQuantizer exposes:

- `static_per_channel()` — default; per-channel weight scales, best accuracy.
- `static_per_tensor()` — one weight scale per tensor, for backends without per-channel kernels.
- `dynamic_per_channel()` / `dynamic_per_tensor()` — weights static int8, activations
  quantized at runtime instead of from calibration data.

## Dynamic shapes

`torch.export.export` guards on the exact shape of `example_args` by default. Any
calibration or inference batch of a different size — including a real `DataLoader`'s
uneven trailing batch — raises a guard-failure error unless you mark the batch dim
dynamic via `dynamic_shapes`. Two gotchas found the hard way:

- `dynamic_shapes` must be passed to `quantize_torch_model` itself, not just implied by
  `example_args` — it's forwarded to *both* the PT2E export step and the final ONNX
  export, since the ONNX graph re-guards independently otherwise.
- Use an example batch size of 2, not 1: torch's 0/1-specialization quirk silently treats
  a dim of size 1 as static even when marked dynamic.

## Reports

`build_report(fp32_model, int8_model, config, parity_samples)` returns a
`QuantizationReport` (`.to_markdown()` / `.to_json_str()`) covering:

- **Size** — fp32 vs int8 byte size and the reduction ratio.
- **Parity** — per-output max/mean absolute diff over real samples. The fp32 and int8
  ONNX graphs come from different export pipelines and don't share tensor names, so
  outputs are paired by position, not name.
- **Op quantization coverage** — which op *types* have their input fed directly by a
  `DequantizeLinear` node. This flags quantization-island entry points, not full
  downstream coverage: a `Relu` immediately after a quantized `Conv` still shows up as
  "unquantized" here, since it consumes `Conv`'s plain float output rather than a
  `DequantizeLinear` output itself. Use it to spot op types that never touch dequantized
  data at all — see the finding below — not to total up "how much of the model runs at
  int8".

Reports are backend-agnostic: they only need two ONNX file paths, regardless of how each
was quantized (or not).

## Finding: attention isn't actually quantized

XNNPACKQuantizer annotates the `Linear` layers around attention (the qkv-in and out
projections) but never annotates `scaled_dot_product_attention`'s internals. Tracing a
quantized model's graph: the `MatMul`s coming from `Linear` layers all take a
`DequantizeLinear` output as input (real int8); the two matmuls *inside* SDPA (Q·Kᵀ and
attn-weights·V) take raw `Mul`/`Softmax`/`Gather` outputs — never a `DequantizeLinear`.
So "W8A8 quantization" of a model with attention really means int8 for conv/linear, fp32
for the attention math itself. This is a real, structural limit of the quantizer, not a
bug — attention quantization is a harder, less mature area than conv/linear across the
whole quantization ecosystem right now.

## Known caveat: the quantizer

The only `Quantizer` that actually imports on macOS is
`torchao.testing.pt2e._xnnpack_quantizer.XNNPACKQuantizer` — used here as the default. It
works and produces correct QDQ output, but:

- It lives under `torchao.testing` (leading underscore — not public API) and is marked
  `deprecated` upstream.
- The maintained replacements, `X86InductorQuantizer` and `ArmInductorQuantizer`, both
  hard-crash on import on this platform (`torch.ops.mkldnn._is_mkldnn_acl_supported()` is
  missing from torch's macOS build — mkldnn/oneDNN isn't shipped for macOS at all).

`Pt2eQuantizationConfig.quantizer` accepts any `Quantizer`, so swap in a maintained one
once you have an environment (e.g. Linux) where it actually imports.

## Example: MNIST end to end

`examples/mnist/` trains a real model (Conv/BatchNorm/residual stem, self-attention +
MLP block, ~39K params — deliberately covering Conv2d, BatchNorm2d, MaxPool2d,
AdaptiveAvgPool2d, residual Add, Dropout, LayerNorm, Linear, SDPA, ReLU, GELU) with
Lightning, checkpoints it via git-lfs, then quantizes it with real calibration data and
prints a full `QuantizationReport`:

```
uv sync --group examples
cd examples/mnist
uv run python train.py           # trains, saves checkpoints/mnist_net.ckpt
uv run python quantize_mnist.py  # quantizes + prints the report
```

Last run: fp32 98.08% accuracy → int8 97.92% (0.16pp drop), 1.24x size reduction (modest
because the model is small enough that QDQ node/scale overhead eats into the savings —
this improves substantially on models with more real weight mass).

## This is a demo, not a package

Like the other branches in this repo, `torchquant` exists to explore a technique
hands-on — it isn't meant to be depended on or used as-is.
