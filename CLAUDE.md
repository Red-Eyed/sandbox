# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`onnxforge` — a graph-level ONNX optimizer for models exported with
`torch.onnx.export(dynamo=True)`, targeting ARM/XNNPACK (and CPU generally). It runs an ordered
pipeline of graph passes, **verifies numeric parity after every pass**, and emits a Markdown
report. Scope is strictly `dynamo=True` exports — TorchScript-era exports are out of scope and
must not be reintroduced.

## Commands

All Python goes through `uv` — never bare `pip`/`python`.

```bash
uv sync --group dev                              # install (incl. torch, stackformers, timm, transformers)
uv run onnxforge work_dir/<model>.onnx           # optimize a model
uv run onnxforge work_dir/<model>.onnx --bench=true   # + latency (CPUExecutionProvider); note: --bench=true, not bare --bench
uv run python -m onnxforge.build_models          # (re)build the demo ONNX zoo into ./work_dir/

uv run ruff check src/                            # lint
uv run ruff format src/                           # format (rewrites in place — re-read touched files)
```

There is no test suite (this is a sandbox repo); do not add unit tests unless asked. Validate
changes by running the pipeline on the demo models in `work_dir/` and checking the parity line.

## Architecture

Two cooperating halves under `src/onnxforge/`:

**1. The optimizer pipeline** (`cli.py`)
- `_PASS_PIPELINE` is an ordered `list[Pass]`. Current order:
  `OnnxSlimPass → FoldTransposePass → PackProjectionsPass → OrtOfflinePass → FuseSkipLayerNormPass`.
  **Order is load-bearing:** packing runs *before* ORT so ORT fuses the larger GEMM;
  SkipLayerNorm runs *after* ORT to mop up residual `Add`+`LN` pairs ORT's conservative fuser skips.
- Each pass subclasses `Pass` (`passes/base.py`): implements `apply(model) -> (new_model, PassResult)`
  and **must not mutate the input** (`copy.deepcopy` first). Optional `is_applicable(model)` gates it.
- The CLI runner wraps every pass in a guard: ONNX checker + `check_parity` (random inputs, 3 seeds,
  `atol=1e-4`). **A pass that fails validation or parity is reverted, not fatal** — the pipeline
  continues with the last good model. This invariant is the whole point; preserve it when editing.
- To add a pass: new file in `passes/`, then add one line to `_PASS_PIPELINE`. Passes are
  self-contained and isolated — keep one concern per file (e.g. `fold_transpose.py`,
  `pack_projections.py`, `fuse_skip_layernorm.py`).

**2. The demo model builder** (`build_models/`)
- A `SpecProvider` registry: each family file in `build_models/specs/` exposes
  `specs() -> list[ExportSpec]`; `registry.py::_PROVIDERS` lists them (one line to add a family).
- `ExportSpec.factory()` is **pure** — returns `(model, args, export_kwargs)`, no I/O. The engine
  (`engine.py`) calls `torch.onnx.export(..., dynamo=True, **export_kwargs)`.

Supporting modules: `diagnose/` (graph/shape/ORT-profile analysis → `Finding`s),
`verify/` (`parity.py`, `benchmark.py`), `report/` (Markdown + LLM-ready summary block).

## Domain specifics that bite

- **`dynamo=True` decomposes SDPA.** `k.transpose(-2,-1)` becomes a multi-node K^T cluster
  (Reshape→Transpose→Reshape + runtime Shape/Slice plumbing) because dynamic seq-len forces runtime
  shape extraction. This is an export artifact, **not** a model bug — fix it in the graph
  (`fold_transpose.py`), never by changing the source model.
- **`dynamic_shapes` must mirror the input pytree.** For a NamedTuple like stackformers
  `PaddedInput(x, mask, abs_positions)`: `dynamic_shapes=(PaddedInput({1: Dim(...)}, {1: Dim(...)}, None),)`.
  `dynamic_axes` does **not** work with NamedTuple inputs in dynamo mode.
- **Boolean mask inputs** (`TensorProto.BOOL`) must be handled in `verify/parity.py` input
  generation — a float fed as a bool mask silently corrupts parity.
- **ORT's attention fusion (`MultiHeadAttention`) cannot match dynamo-exported attention**, so the
  graph keeps explicit `FusedMatMul → Softmax → MatMul`. This is intended: on CPU/ARM the decomposed
  MLAS GEMM path is as fast or faster than the fused attention kernel, and XNNPACK doesn't have an
  Attention kernel anyway (it only delegates 2D MatMul/Gemm/Conv/Softmax — 4D batched matmuls and
  `com.microsoft` contrib ops fall back to MLAS regardless).

## Conventions

- `torch>=2.12`, Python 3.11+. Strict types over stringly-typed data; SOLID, especially
  Open/Closed — add behavior by adding a pass/spec file, not by threading flags through existing
  functions.
- Generated artifacts (`work_dir/`, `*_optimized.onnx`, `*_onnxforge_report.md`) are gitignored.
- This repo is **public** — never commit secrets or local machine paths.
