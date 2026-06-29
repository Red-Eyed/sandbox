#!/usr/bin/env python3
"""onnx-speedup — ONNX optimization pipeline for Android/ARM deployment."""

import sys
from pathlib import Path

import onnx
import onnx.checker
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp, CliPositionalArg

from onnx_speedup.diagnose import graph_analyzer, profile_analyzer, shape_analyzer
from onnx_speedup.passes.base import Pass, PassResult
from onnx_speedup.passes.onnxslim_pass import OnnxSlimPass
from onnx_speedup.passes.ort_offline import OrtOfflinePass
from onnx_speedup.report.report import PipelineResult
from onnx_speedup.report.report import generate as generate_report
from onnx_speedup.verify.parity import check_parity

__version__ = "0.1.0"

_PASS_PIPELINE: list[Pass] = [
    OnnxSlimPass(),
    OrtOfflinePass(),
]

_SEP = "─" * 44


class SpeedupArgs(BaseSettings):
    model: CliPositionalArg[Path] = Field(description="Input .onnx file")
    output: Path | None = Field(
        default=None,
        description="Output .onnx path (default: <model>_optimized.onnx)",
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


def main() -> None:
    CliApp.run(SpeedupArgs)


def _run(args: SpeedupArgs) -> int:
    if not args.model.exists():
        print(f"Error: {args.model} not found", file=sys.stderr)
        return 1

    output_path = args.output or _default_output(args.model)
    report_path = args.model.with_name(args.model.stem + "_speedup_report.md")

    print(f"onnx-speedup v{__version__}")
    print(_SEP)

    model = onnx.load(str(args.model))
    graph_before = graph_analyzer.analyze(model)
    shape_info = shape_analyzer.analyze(model)

    _print_header(args.model, graph_before, shape_info)

    original_model = model
    pass_results: list[tuple[str, PassResult]] = []
    current_model = model

    for step, pass_ in enumerate(_PASS_PIPELINE, 1):
        label = f"[{step}/{len(_PASS_PIPELINE)}] {pass_.description}..."
        print(f"{label:<48}", end="", flush=True)

        if not pass_.is_applicable(current_model):
            result = PassResult(
                applied=False,
                description="not applicable",
                nodes_before=len(current_model.graph.node),
                nodes_after=len(current_model.graph.node),
            )
            print("— skipped")
            pass_results.append((pass_.name, result))
            continue

        try:
            new_model, result = pass_.apply(current_model)
        except Exception as exc:
            result = PassResult(
                applied=False,
                description=str(exc),
                nodes_before=len(current_model.graph.node),
                nodes_after=len(current_model.graph.node),
                error=str(exc),
            )
            print(f"✗  ERROR: {exc}")
            pass_results.append((pass_.name, result))
            continue

        if result.error:
            print(f"✗  {result.error}")
            pass_results.append((pass_.name, result))
            continue

        try:
            onnx.checker.check_model(new_model)
        except Exception as exc:
            result.applied = False
            result.error = f"post-pass validation failed: {exc}"
            print("✗  validation failed, pass skipped")
            pass_results.append((pass_.name, result))
            continue

        if result.applied:
            parity = check_parity(current_model, new_model)
            if not parity.passed:
                result.applied = False
                result.error = (
                    f"parity check failed (max_diff={parity.max_diff:.2e}, "
                    f"atol={parity.atol:.2e}): {parity.error or ''}"
                )
                print("✗  parity FAIL — pass skipped")
                pass_results.append((pass_.name, result))
                continue
            current_model = new_model

        summary = result.description or ""
        print(f"✓  {summary}" if result.applied else f"—  {summary}")
        pass_results.append((pass_.name, result))

    print()
    final_parity = check_parity(original_model, current_model)
    parity_mark = "✓" if final_parity.passed else "✗"
    print(
        f"Parity check:  {parity_mark}  max_diff={final_parity.max_diff:.2e}  "
        f"(atol={final_parity.atol:.2e})"
    )

    graph_after = graph_analyzer.analyze(current_model)

    profile = None
    if args.profile:
        try:
            profile = profile_analyzer.analyze(args.profile)
        except Exception as exc:
            print(f"Warning: profile analysis failed: {exc}", file=sys.stderr)

    print()
    print(_SEP)
    print(f"Before:  {graph_before.n_nodes} nodes")
    print(f"After:   {graph_after.n_nodes} nodes  (-{graph_before.n_nodes - graph_after.n_nodes})")
    print()

    onnx.save(current_model, str(output_path))
    print(f"Optimized: {output_path}")

    pipeline_result = PipelineResult(
        model_name=args.model.name,
        pass_results=pass_results,
        graph_before=graph_before,
        graph_after=graph_after,
        parity=final_parity,
        profile=profile,
    )
    generate_report(pipeline_result, report_path)
    print(f"Report:    {report_path}")
    print(_SEP)

    _print_manual_actions(pipeline_result)
    return 0


def _default_output(model_path: Path) -> Path:
    return model_path.with_name(model_path.stem + "_optimized.onnx")


def _print_header(
    model_path: Path,
    graph: graph_analyzer.GraphAnalysis,
    shape: shape_analyzer.ShapeAnalysis,
) -> None:
    params_str = f"{graph.n_params / 1e6:.1f}M" if graph.n_params >= 1e6 else str(graph.n_params)
    dynamic_str = ", ".join(d.name for d in shape.dynamic_dims) if shape.dynamic_dims else "none"
    print(f"Model:   {model_path.name}  ({graph.n_nodes} nodes, {params_str} params)")
    print(f"Dynamic: {dynamic_str}")
    print()


def _print_manual_actions(result: PipelineResult) -> None:
    from onnx_speedup.report.report import _collect_all_findings

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
