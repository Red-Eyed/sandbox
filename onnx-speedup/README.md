# onnx-speedup

ONNX optimization pipeline for Android/ARM deployment via ORT + XNNPACK.

Takes a raw PyTorch-exported `.onnx`, runs automatic graph passes, verifies
numeric parity, and produces an optimized model ready to deploy.

## Quickstart

```bash
# Install
uv pip install -e .

# Run (minimal — static analysis + all automatic passes)
uv run speedup model.onnx

# With ORT profile for EP assignment analysis
uv run speedup model.onnx --profile profile.json

# Custom output path
uv run speedup model.onnx --output model_opt.onnx
```

Outputs:
- `model_optimized.onnx` — optimized model
- `model_speedup_report.md` — full report, LLM-pasteable

## Pass pipeline

| Order | Pass | What it does |
|-------|------|-------------|
| 1 | `onnxslim` | Dead-node elimination, constant folding |
| 2 | `eliminate_cast` | Remove no-op float Cast nodes |
| 3 | `fuse_gemm` | MatMul + Add(bias) → Gemm |
| 4 | `fuse_gelu` | Decomposed GeLU → `com.microsoft.Gelu` |
| 5 | `fuse_layernorm` | 9-node LN chain → `LayerNormalization` |
| 6 | `ort_offline` | ORT `ORT_ENABLE_ALL` optimization |

Parity is checked after every pass (atol=1e-4). A failing pass is skipped
automatically; the pipeline continues with the remaining passes.

## Project layout

```
src/onnx_speedup/
├── cli.py              ← entry point (speedup command)
├── diagnose/           ← static graph analysis, profile parsing, shape analysis
├── passes/             ← optimization passes (one file per pass)
├── verify/             ← numeric parity check, latency benchmark
└── report/             ← markdown report generator

tests/
├── fixtures/builders.py  ← minimal ONNX graph builders for unit tests
├── test_passes.py         ← unit test per pass
└── test_parity.py         ← integration test (full pipeline)

docs/
├── layernorm_fusion.md    ← when/how to fix unfused LayerNorm
├── dynamic_shapes.md      ← bucketing strategy for XNNPACK
└── ep_assignment.md       ← verifying XNNPACK delegation via ORT profile
```

## Running tests

```bash
uv run pytest
```

## What this tool does NOT do

- Retrain or quantize (assumes QAT done upstream if needed)
- Recompose MHA from decomposed attention primitives
- Benchmark on Android device (v1 — desktop ORT only)
- Handle models with control flow (`If`/`Loop` nodes)
