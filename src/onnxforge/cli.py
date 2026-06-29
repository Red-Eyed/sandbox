#!/usr/bin/env python3
"""onnxforge — graph-level ONNX optimization for dynamo exports, targeting ARM."""

import sys
from pathlib import Path

import onnx
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp, CliPositionalArg

from onnxforge.diagnose import graph_analyzer, profile_analyzer, shape_analyzer
from onnxforge.optimize import OptimizeConfig, OptimizeResult, optimize
from onnxforge.passes.base import Pass, PassResult
from onnxforge.report.report import PipelineResult
from onnxforge.report.report import generate as generate_report
from onnxforge.verify.benchmark import BenchResult, benchmark

__version__ = "0.1.0"

_SEP = "─" * 44


class SpeedupArgs(BaseSettings):
    model: CliPositionalArg[Path] = Field(description="Input .onnx file")
    output: Path | None = Field(
        default=None,
        description="Output .onnx path (default: <model>_optimized.onnx)",
    )
    target_opset: int | None = Field(
        default=None,
        alias="target-opset",
        description="Convert the model's standard opset to this version before optimizing "
        "(guarded by parity; unlocks opset-gated passes like RMSNorm fusion at ≥23)",
    )
    bench: bool = Field(
        default=False,
        description="Measure inference latency before and after (CPUExecutionProvider)",
    )
    profile: Path | None = Field(
        default=None,
        description="ORT profile JSON for EP assignment analysis",
    )
    exported_graph: Path | None = Field(
        default=None,
        alias="exported-graph",
        description="ExportedProgram graph dump (.txt)",
    )

    def cli_cmd(self) -> None:
        sys.exit(_run(self))


class _ConsoleObserver:
    """Prints pipeline progress to stdout in the onnxforge CLI format.  This is
    the presentation layer kept out of the optimization core (see optimize.py)."""

    def on_opset_converted(self, source: int, target: int, ok: bool) -> None:
        if ok:
            print(f"Opset:   {source} → {target}  ✓")
        else:
            print(f"Opset:   {source} → {target}  ✗ (reverted, kept {source})")

    def on_pass_start(self, step: int, total: int, pass_: Pass) -> None:
        label = f"[{step}/{total}] {pass_.description}..."
        print(f"{label:<48}", end="", flush=True)

    def on_pass_skipped(self, step: int, pass_: Pass, reason: str) -> None:
        suffix = "" if reason == "not applicable" else f" ({reason})"
        print(f"—  skipped{suffix}")

    def on_pass_result(self, step: int, pass_: Pass, result: PassResult) -> None:
        if result.error:
            print(f"✗  {result.error}")
        elif result.applied:
            print(f"✓  {result.description or ''}")
        else:
            print(f"—  {result.description or ''}")


def main() -> None:
    CliApp.run(SpeedupArgs)


def _run(args: SpeedupArgs) -> int:
    if not args.model.exists():
        print(f"Error: {args.model} not found", file=sys.stderr)
        return 1

    output_path = args.output or _default_output(args.model)
    report_path = args.model.with_name(args.model.stem + "_onnxforge_report.md")

    print(f"onnxforge v{__version__}")
    print(_SEP)

    model = onnx.load(str(args.model))
    shape_info = shape_analyzer.analyze(model)
    _print_header(args.model, graph_analyzer.analyze(model), shape_info)

    config = OptimizeConfig(target_opset=args.target_opset)
    result = optimize(model, config, observer=_ConsoleObserver())

    print()
    parity_mark = "✓" if result.parity.passed else "✗"
    print(
        f"Parity check:  {parity_mark}  max_diff={result.parity.max_diff:.2e}  "
        f"(atol={result.parity.atol:.2e})"
    )

    bench_before, bench_after = _maybe_benchmark(args, model, result.model)
    profile = _maybe_profile(args)

    _print_summary(result, bench_before, bench_after)

    onnx.save(result.model, str(output_path))
    print(f"Optimized: {output_path}")

    pipeline_result = PipelineResult(
        model_name=args.model.name,
        pass_results=result.pass_results,
        graph_before=result.graph_before,
        graph_after=result.graph_after,
        parity=result.parity,
        profile=profile,
        bench_before=bench_before,
        bench_after=bench_after,
    )
    generate_report(pipeline_result, report_path)
    print(f"Report:    {report_path}")
    print(_SEP)

    _print_manual_actions(pipeline_result)
    return 0


def _maybe_benchmark(
    args: SpeedupArgs,
    original: onnx.ModelProto,
    optimized: onnx.ModelProto,
) -> tuple[BenchResult | None, BenchResult | None]:
    if not args.bench:
        return None, None
    print()
    print("Benchmarking (CPUExecutionProvider, 10 warmup + 100 runs)...")
    return (
        benchmark(original, n_warmup=10, n_runs=100),
        benchmark(optimized, n_warmup=10, n_runs=100),
    )


def _maybe_profile(args: SpeedupArgs):
    if not args.profile:
        return None
    try:
        return profile_analyzer.analyze(args.profile)
    except Exception as exc:
        print(f"Warning: profile analysis failed: {exc}", file=sys.stderr)
        return None


def _print_summary(
    result: OptimizeResult,
    bench_before: BenchResult | None,
    bench_after: BenchResult | None,
) -> None:
    before = result.graph_before.n_nodes
    after = result.graph_after.n_nodes
    print()
    print(_SEP)
    print(f"Before:  {before} nodes")
    print(f"After:   {after} nodes  (-{before - after})")
    if bench_before and bench_after:
        speedup = bench_before.p50_ms / bench_after.p50_ms
        print(
            f"Latency: {bench_before.p50_ms:.2f}ms → {bench_after.p50_ms:.2f}ms  "
            f"(p50, {speedup:.2f}× speedup)"
        )
    print()


def _default_output(model_path: Path) -> Path:
    return model_path.with_name(model_path.stem + "_optimized.onnx")


def _print_header(
    model_path: Path,
    graph: graph_analyzer.GraphAnalysis,
    shape: shape_analyzer.ShapeAnalysis,
) -> None:
    dynamic_names = list(dict.fromkeys(d.name for d in shape.dynamic_dims))
    dynamic_str = ", ".join(dynamic_names) if dynamic_names else "none"
    params = _human_params(graph.n_params)
    print(f"Model:   {model_path.name}  ({graph.n_nodes} nodes, {params} params)")
    print(f"Dynamic: {dynamic_str}")
    print()


def _human_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _print_manual_actions(result: PipelineResult) -> None:
    from onnxforge.report.report import _collect_all_findings

    findings = _collect_all_findings(result)
    if not findings:
        return

    print(f"\nManual actions required ({len(findings)}):\n")
    for i, f in enumerate(findings, 1):
        print(f"  [{f.severity}] {f.title}")
        print(f"         {f.detail}")
        if f.guide:
            print(f"         Fix:   See {f.guide}")
        print()


if __name__ == "__main__":
    main()
