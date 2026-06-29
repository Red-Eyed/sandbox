<div align="center">

# onnxforge

**Graph-level ONNX optimization for `torch.onnx.export(dynamo=True)` models — built for researchers who train from scratch and ship to ARM.**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/onnxruntime-1.16+-005ce6.svg)](https://onnxruntime.ai/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Parity](https://img.shields.io/badge/every%20pass-parity%20checked-brightgreen.svg)](#safety-every-pass-is-verified)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

</div>

---

The new `dynamo=True` exporter decomposes operations the old TorchScript exporter kept fused —
a single `k.transpose(-2, -1)` inside attention becomes an 8-node cluster. `onnxforge` runs
an ordered pipeline of graph passes that **fold those artifacts back, pack parallel GEMMs, and
fuse residual norms** — checking numeric parity after every single pass, so the output is always
a model you can trust.

```console
$ uv run onnxforge work_dir/encoder_ln_dynamic.onnx

onnxforge v0.1.0
────────────────────────────────────────────
Model:   encoder_ln_dynamic.onnx  (158 nodes, 786.2K params)
Dynamic: s29

[1/5] onnxslim ..............................  ✓  158 → 147 nodes (-7%)
[2/5] Fold K^T cluster .....................  ✓  4 K^T cluster(s) folded
[3/5] Pack QKV / gated-FF projections ......  ✓  8 projection group(s) packed
[4/5] ORT graph optimization ...............  ✓  135 → 121 nodes
[5/5] Fuse residual Add+LN → SkipLayerNorm .  ✓  7 pair(s) fused

Parity check:  ✓  max_diff=9.54e-07  (atol=1.00e-04)
────────────────────────────────────────────
Before:  158 nodes
After:   114 nodes  (-44)

Optimized: work_dir/encoder_ln_dynamic_optimized.onnx
Report:    work_dir/encoder_ln_dynamic_onnxforge_report.md
```

## Features

- **Five ordered graph passes** — slim, K^T fold, projection packing, ORT fusion, SkipLayerNorm fusion.
- **Parity-checked by construction** — every pass is reverted if it changes outputs beyond `1e-4`.
- **A diff, not a black box** — Markdown report with per-pass node deltas, op distribution, and an LLM-ready summary block.
- **`dynamo=True` native** — understands the exact artifacts torch.export emits at opset 17+, including `PaddedInput`-style NamedTuple exports.
- **Safe on any model** — CNNs with nothing to fuse simply pass through unchanged.
- **Zero model changes** — all optimization is offline graph/weight surgery; you never touch your PyTorch code.

## Contents

- [Quick start](#quick-start)
- [Results](#results)
- [How it works](#how-it-works)
- [Export your own model](#export-your-own-model)
- [Extending](#extending)
- [Deployment](#deployment)
- [Safety: every pass is verified](#safety-every-pass-is-verified)
- [Why not onnx-graphsurgeon?](#why-not-onnx-graphsurgeon)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [License](#license)

## Quick start

```bash
# install (CPU-only, no GPU required)
uv sync --group dev

# build the demo zoo: stackformers encoders, HF BERT-tiny, timm CNNs → ./work_dir/
uv run python -m onnxforge.build_models

# optimize any model
uv run onnxforge work_dir/encoder_ln_dynamic.onnx

# …with a latency benchmark on your CPU
uv run onnxforge work_dir/encoder_ln_dynamic.onnx --bench=true

# …to a custom path
uv run onnxforge work_dir/encoder_ln_dynamic.onnx --output out/encoder.onnx
```

Two artifacts land next to the input: `<model>_optimized.onnx` (ship this) and
`<model>_onnxforge_report.md` (the diff + an LLM-ready summary you can paste into a chat).

## Results

Measured on the bundled demo models. Every result is **parity-checked** (`max_diff < 1e-4`).
Node count is deterministic and reproducible; latency is intentionally not quoted here (it is
machine-, EP-, and sequence-specific — measure it on your target with `--bench`).

| Model | Before | After | Δ nodes | Parity (max_diff) |
|-------|-------:|------:|--------:|-------------------|
| `bert_tiny`              | 199 | 135 | **−64** | `0.00e+00` |
| `efficientnet_b0`        | 239 | 176 | **−63** | `0.00e+00` |
| `encoder_rms_dynamic`    | 208 | 162 | **−46** | `0.00e+00` |
| `decoder_causal_dynamic` | 172 | 127 | **−45** | `7.15e-07` |
| `encoder_ln_dynamic`     | 158 | 114 | **−44** | `9.54e-07` |
| `encoder_ln_static`      | 128 |  91 | **−37** | `7.15e-07` |
| `mobilenetv3`            | 122 | 120 |  −2 | `0.00e+00` |
| `resnet18`               |  49 |  49 |   0 | `0.00e+00` |

> CNNs barely move — they have no parallel projections or residual LayerNorms to fuse, so the
> transformer passes correctly no-op. That is the point: the pipeline is safe to run on anything.

## How it works

The pipeline is a fixed, ordered sequence. Order is load-bearing — packing runs *before* ORT so
ORT fuses the larger GEMM; SkipLayerNorm runs *after* ORT to catch what ORT's conservative fuser
skips.

| # | Pass | Does |
|---|------|------|
| 1 | `onnxslim` | Dead-node elimination, constant folding, shape inference |
| 2 | `fold_transpose` | Collapses the SDPA K^T cluster into one `Transpose` per attention layer |
| 3 | `pack_projections` | Concatenates QKV / gated-FF weights into one GEMM + `Split` |
| 4 | `ort_offline` | ORT `ORT_ENABLE_ALL` — `FusedMatMul`, `QuickGelu`, terminal `SkipLayerNormalization` |
| 5 | `fuse_skip_layernorm` | Fuses the residual `Add`+`LayerNorm` pairs ORT leaves behind |

The op-count delta for `encoder_ln_dynamic` shows where the 44 nodes go:

```
MatMul     36 → 20  (−16, QKV/FF packed + scale → FusedMatMul)
Mul        16 →  4  (−12, attention scale + SiLU folded)
LayerNorm   9 →  1  ( −8, → SkipLayerNormalization ×8)
Reshape    25 → 17  ( −8, K^T cluster Reshapes removed)
Slice      12 →  4  ( −8, runtime shape plumbing removed)
Add        12 →  4  ( −8, residual Adds absorbed into SkipLayerNorm)
Sigmoid     4 →  0  ( −4, → QuickGelu ×4)
Concat     12 →  9  ( −3)
Transpose  20 → 16  ( −4)
```

<details>
<summary><b>The K^T cluster — why it exists, and why you should not fix it in your model</b></summary>

<br>

`torch.onnx.export(dynamo=True)` decomposes `F.scaled_dot_product_attention` at opset 17.
Inside SDPA, `k.transpose(-2, -1)` on a 4-D tensor `[B, H, S, d]` expands into:

```
Transpose([0,2,1,3])   # head split: [B, S, H, d] → [B, H, S, d]
  → Reshape            # merge B,H → [B*H, S, d]  (needs runtime Shape for dynamic seq_len)
  → Transpose([0,2,1]) # 3-D K^T:  [B*H, d, S]
  → Reshape            # restore 4-D: [B, H, d, S]
```

This equals a single `Transpose(perm=[0,1,3,2])` on the original 4-D tensor. The cluster is a
lowering artifact — dynamic sequence-length dimensions force runtime `Shape` extraction to build
the Reshape targets. It is **not a bug in your attention code.** `fold_transpose` fixes it once,
in the graph.
</details>

<details>
<summary><b>Projection packing — fewer, larger GEMMs</b></summary>

<br>

Per transformer block, several MatMuls read the *same* normalized activation:

```
Q = LN(x) @ Wq    K = LN(x) @ Wk    V = LN(x) @ Wv      (attention)
g = LN(x) @ Wg    u = LN(x) @ Wu                        (gated FF)
```

`pack_projections` concatenates the weight matrices along the output axis, runs one GEMM, and
`Split`s the result. One large GEMM beats three small ones on both MLAS and XNNPACK (better cache
reuse, one kernel launch). Pure offline weight surgery — numerically identical, no retrain.
</details>

<details>
<summary><b>SkipLayerNorm fusion — recovering what ORT's fuser skips</b></summary>

<br>

In a pre-norm transformer the residual stream reuses the sum:

```
h = x + attn(LN(x))     # h feeds BOTH the next LN and the next residual Add
```

ORT's offline fuser bails when the residual `Add` has multiple consumers, so 7 of 8 LayerNorms
never fuse. `SkipLayerNormalization` exposes an optional 4th output (`input_skip_bias_sum`) for
exactly this case — `fuse_skip_layernorm` routes the residual stream through it and deletes the
standalone `Add`.
</details>

## Export your own model

**Flat tensor inputs:**

```python
import torch
from torch.export import Dim

seq = Dim("seq_len", min=1, max=4096)
torch.onnx.export(
    model.eval(), (x,), "my_model.onnx",
    dynamo=True,
    input_names=["x"], output_names=["out"],
    dynamic_shapes=({1: seq},),       # tuple mirrors args: (x,)
)
```

**NamedTuple / dataclass inputs** (e.g. stackformers `PaddedInput`) — `dynamic_shapes` must
mirror the pytree exactly; `dynamic_axes` does **not** work in dynamo mode:

```python
from stackformers.sequence import PaddedInput
from torch.export import Dim

seq = Dim("seq_len", min=1, max=4096)
inp = PaddedInput(torch.zeros(1, 8, 128), torch.ones(1, 8, dtype=torch.bool), None)

torch.onnx.export(
    model.eval(), (inp,), "encoder.onnx",
    dynamo=True,
    input_names=["x", "mask"], output_names=["out"],
    dynamic_shapes=(PaddedInput({1: seq}, {1: seq}, None),),
)
```

Then: `uv run onnxforge encoder.onnx`.

## Extending

<details>
<summary><b>Add a custom pass</b></summary>

<br>

Subclass `Pass`, drop the file in `src/onnxforge/passes/`, and add it to `_PASS_PIPELINE`
in `cli.py`. Parity + ONNX validation run automatically — a failing pass is skipped, not fatal.

```python
import copy
import onnx
from onnxforge.passes.base import Pass, PassResult

class MyPass(Pass):
    name = "my_pass"
    description = "Short description shown in CLI output"

    def apply(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, PassResult]:
        model = copy.deepcopy(model)          # never mutate the input
        before = len(model.graph.node)
        ...                                   # rewrite model.graph.node
        after = len(model.graph.node)
        return model, PassResult(
            applied=after < before,
            description=f"{before} → {after} nodes",
            nodes_before=before, nodes_after=after,
        )
```
</details>

<details>
<summary><b>Add a model family to the demo zoo</b></summary>

<br>

Create `src/onnxforge/build_models/specs/<family>.py` exposing `specs() -> list[ExportSpec]`,
then add it to `_PROVIDERS` in `registry.py`:

```python
from ..protocols import ExportSpec

def specs() -> list[ExportSpec]:
    return [ExportSpec(name="my_model", group="my_family", factory=_factory(...))]
```

Run `uv run python -m onnxforge.build_models` and your model is ready for `uv run onnxforge`.
</details>

## Deployment

The tool produces the optimized graph; runtime wiring lives in your app.

**Android / XNNPACK:**

```java
OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
opts.addXnnpack(Collections.emptyMap());
OrtSession session = env.createSession("encoder_optimized.onnx", opts);
```

XNNPACK accelerates `MatMul`, `Conv`, `Relu`, `Sigmoid`, `Softmax`, and element-wise ops. ORT
custom ops (`FusedMatMul`, `SkipLayerNormalization`) fall back to ORT MLAS — still faster than the
decomposed form on ARM, and the win is *larger* than on desktop x86 because the decomposed
`Shape → Slice → Concat → Reshape` K^T plumbing falls off the accelerated path entirely.

**CoreML (optional, Mac/iOS)** — `CPUExecutionProvider` is the default; opt in explicitly:

```python
ort.InferenceSession("model.onnx",
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
```

## Safety: every pass is verified

After each pass the pipeline runs the ONNX checker and a numeric parity check — random inputs over
3 seeds, comparing original vs. current outputs with `atol=1e-4`. **If a pass changes outputs or
produces an invalid graph, it is reverted and the pipeline continues** with the last good model.
The final model is parity-checked end-to-end against the original. You never ship a silently-broken
graph.

## Why not onnx-graphsurgeon?

[onnx-graphsurgeon](https://github.com/NVIDIA/onnx-graphsurgeon) is a mutable-IR scalpel — perfect
when you already know the exact subgraph to swap. `onnxforge` is the operating procedure: an
*ordered, parity-checked pipeline* that knows the specific artifacts `dynamo=True` emits, applies
the right rewrites in the right order, and proves it didn't break anything. Surgeon is how you'd
*implement* a one-off fix; this is how you'd *ship* a repeatable one.

## Project layout

```
src/onnxforge/
├── cli.py                  # `uv run onnxforge <model.onnx>` entry point
├── passes/
│   ├── base.py             # Pass ABC + PassResult
│   ├── onnxslim_pass.py    # dead-node elimination + constant folding
│   ├── fold_transpose.py   # K^T cluster → single Transpose
│   ├── pack_projections.py # QKV / gated-FF → one GEMM + Split
│   ├── ort_offline.py      # ORT ORT_ENABLE_ALL
│   └── fuse_skip_layernorm.py  # residual Add+LN → SkipLayerNormalization
├── diagnose/               # graph / shape / ORT-profile analyzers
├── verify/                 # parity check + latency benchmark
├── report/                 # Markdown report + LLM-ready summary
└── build_models/           # demo zoo (stackformers, HF, timm) behind a SpecProvider registry
```

## Requirements

- Python 3.11+
- PyTorch 2.12+ — required for `dynamo=True` with NamedTuple `dynamic_shapes`
- onnxruntime 1.16+

Build-zoo extras (`stackformers`, `transformers`, `timm`) install with `uv sync --group dev`.

## License

[MIT](LICENSE) © Vadym Stupakov
