from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from onnx_speedup.diagnose.graph_analyzer import Finding, GraphAnalysis
from onnx_speedup.diagnose.profile_analyzer import ProfileAnalysis
from onnx_speedup.passes.base import PassResult
from onnx_speedup.verify.parity import ParityResult


@dataclass
class PipelineResult:
    model_name: str
    pass_results: list[tuple[str, PassResult]]  # (pass_name, result)
    graph_before: GraphAnalysis
    graph_after: GraphAnalysis
    parity: ParityResult
    profile: ProfileAnalysis | None = None


def generate(result: PipelineResult, output_path: Path) -> None:
    lines = _build_report(result)
    output_path.write_text("\n".join(lines))


def _build_report(r: PipelineResult) -> list[str]:
    lines: list[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    lines += [
        f"# ONNX Speedup Report — {r.model_name}",
        f"Generated: {ts}",
        "",
        "## Summary",
        f"- Before: {r.graph_before.n_nodes} nodes, ~{r.graph_before.n_params / 1e6:.1f}M params",
        f"- After:  {r.graph_after.n_nodes} nodes "
        f"(-{r.graph_before.n_nodes - r.graph_after.n_nodes})",
        f"- Passes applied: {', '.join(n for n, res in r.pass_results if res.applied)}",
        f"- Parity: {'✓' if r.parity.passed else '✗'} max_diff={r.parity.max_diff:.2e}",
        "",
    ]

    lines += _section_passes(r.pass_results)
    lines += _section_manual_actions(r)
    lines += _section_graph(r.graph_before, r.graph_after)

    if r.profile:
        lines += _section_profile(r.profile)

    lines += _section_llm_summary(r)
    return lines


def _section_passes(pass_results: list[tuple[str, PassResult]]) -> list[str]:
    lines = ["## Automatic Optimizations Applied", ""]
    lines += [
        "| Pass | Nodes before | Nodes after | Result |",
        "|------|-------------|------------|--------|",
    ]
    for name, res in pass_results:
        status = "✓" if res.applied else ("✗" if res.error else "—")
        lines.append(
            f"| {name} | {res.nodes_before} | {res.nodes_after} | {status} {res.description} |"
        )
    lines.append("")
    return lines


def _section_manual_actions(r: PipelineResult) -> list[str]:
    findings = _collect_all_findings(r)
    if not findings:
        return []

    lines = ["## Manual Actions Required", ""]
    for i, f in enumerate(findings, 1):
        lines += [
            f"### {i}. {f.title}  [{f.severity}]",
            f.detail,
        ]
        if f.guide:
            lines.append(f"See: [{f.guide}]({f.guide})")
        lines.append("")
    return lines


def _collect_all_findings(r: PipelineResult) -> list[Finding]:
    findings = list(r.graph_before.findings)
    if r.profile:
        findings.extend(r.profile.flags)
    return sorted(findings, key=lambda f: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[f.severity])


def _section_graph(before: GraphAnalysis, after: GraphAnalysis) -> list[str]:
    lines = ["## Graph Analysis", ""]
    lines += ["### Op distribution (before)", ""]
    lines += ["| Op type | Count |", "|---------|-------|"]
    top_ops = sorted(before.op_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for op, count in top_ops:
        lines.append(f"| {op} | {count} |")
    lines.append("")

    if before.dynamic_dims:
        lines += [f"**Dynamic dims:** {', '.join(before.dynamic_dims)}", ""]

    if before.domain_ops:
        non_std = {d: ops for d, ops in before.domain_ops.items()}
        if non_std:
            lines += ["**Non-standard domain ops:**"]
            for domain, ops in non_std.items():
                lines.append(f"- `{domain}`: {', '.join(set(ops))}")
            lines.append("")
    return lines


def _section_profile(p: ProfileAnalysis) -> list[str]:
    lines = ["## Profile Analysis", ""]
    lines += [
        f"- Total inference: {p.total_ms:.2f}ms",
        f"- XNNPACK: {p.xnnpack_pct:.1f}%",
        f"- CPU fallback (MLAS): {p.cpu_fallback_pct:.1f}%",
        "",
        "### Top ops by time",
        "",
        "| Op | Total ms | Calls | EP |",
        "|----|---------|-------|----|",
    ]
    for stat in p.by_op[:10]:
        lines.append(f"| {stat.op_type} | {stat.total_ms:.2f} | {stat.call_count} | {stat.ep} |")
    lines.append("")
    return lines


def _section_llm_summary(r: PipelineResult) -> list[str]:
    lines = [
        "## LLM-Ready Summary",
        "<!-- Paste this section when asking for optimization advice -->",
        "",
        f"Model: {r.model_name}",
        f"Nodes before/after: {r.graph_before.n_nodes} → {r.graph_after.n_nodes}",
        f"Params: ~{r.graph_before.n_params / 1e6:.1f}M",
        f"Dynamic dims: {', '.join(r.graph_before.dynamic_dims) or 'none'}",
        f"Parity: {'PASS' if r.parity.passed else 'FAIL'} (max_diff={r.parity.max_diff:.2e})",
        "",
        "Passes:",
    ]
    for name, res in r.pass_results:
        status = "applied" if res.applied else ("error" if res.error else "skipped")
        lines.append(f"  {name}: {status} — {res.description}")

    findings = _collect_all_findings(r)
    if findings:
        lines += ["", "Issues requiring manual action:"]
        for f in findings:
            lines.append(f"  [{f.severity}] {f.title}: {f.detail}")

    return lines
