import json
from dataclasses import dataclass, field
from pathlib import Path

from onnxforge.diagnose.graph_analyzer import Finding


@dataclass
class OpStat:
    op_type: str
    total_ms: float
    call_count: int
    ep: str

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.call_count if self.call_count else 0.0


@dataclass
class EpStat:
    ep: str
    total_ms: float
    pct: float


@dataclass
class ProfileAnalysis:
    total_ms: float
    by_op: list[OpStat]
    by_ep: list[EpStat]
    xnnpack_pct: float
    cpu_fallback_pct: float
    flags: list[Finding] = field(default_factory=list)


def analyze(profile_path: Path) -> ProfileAnalysis:
    raw = json.loads(profile_path.read_text())
    events = [e for e in raw if e.get("cat") == "Node" and "dur" in e]

    by_op_ms: dict[str, float] = {}
    by_op_count: dict[str, int] = {}
    by_op_ep: dict[str, str] = {}
    by_ep_ms: dict[str, float] = {}

    for e in events:
        args = e.get("args", {})
        op = args.get("op_name", e.get("name", "unknown"))
        dur_ms = e["dur"] / 1000.0
        ep = args.get("provider", "unknown")

        by_op_ms[op] = by_op_ms.get(op, 0.0) + dur_ms
        by_op_count[op] = by_op_count.get(op, 0) + 1
        by_op_ep[op] = ep
        by_ep_ms[ep] = by_ep_ms.get(ep, 0.0) + dur_ms

    total_ms = sum(by_ep_ms.values()) or 1.0

    by_op = sorted(
        [OpStat(op, by_op_ms[op], by_op_count[op], by_op_ep[op]) for op in by_op_ms],
        key=lambda s: s.total_ms,
        reverse=True,
    )

    by_ep = sorted(
        [EpStat(ep, ms, ms / total_ms * 100) for ep, ms in by_ep_ms.items()],
        key=lambda s: s.total_ms,
        reverse=True,
    )

    xnnpack_pct = next((s.pct for s in by_ep if "XNNPACK" in s.ep), 0.0)
    cpu_fallback_pct = next((s.pct for s in by_ep if "CPUExecutionProvider" in s.ep), 0.0)

    flags = _detect_flags(by_op, by_ep, total_ms, cpu_fallback_pct)

    return ProfileAnalysis(
        total_ms=total_ms,
        by_op=by_op,
        by_ep=by_ep,
        xnnpack_pct=xnnpack_pct,
        cpu_fallback_pct=cpu_fallback_pct,
        flags=flags,
    )


def _detect_flags(
    by_op: list[OpStat],
    by_ep: list[EpStat],
    total_ms: float,
    cpu_fallback_pct: float,
) -> list[Finding]:
    flags: list[Finding] = []

    if cpu_fallback_pct > 15:
        flags.append(
            Finding(
                severity="HIGH",
                title="High CPU fallback (MLAS) usage",
                detail=f"{cpu_fallback_pct:.1f}% of time on CPUExecutionProvider — "
                "significant fraction not delegated to XNNPACK.",
                guide="docs/ep_assignment.md",
            )
        )

    for stat in by_op[:5]:
        pct = stat.total_ms / total_ms * 100
        if pct > 30:
            flags.append(
                Finding(
                    severity="MEDIUM",
                    title=f"Hotspot: {stat.op_type} ({pct:.1f}% of total)",
                    detail=f"{stat.op_type} accounts for {pct:.1f}% of inference time "
                    f"({stat.total_ms:.2f}ms, {stat.call_count} calls).",
                    locations=[stat.op_type],
                )
            )

    cast_stats = [s for s in by_op if s.op_type == "Cast"]
    if cast_stats:
        total_cast_ms = sum(s.total_ms for s in cast_stats)
        flags.append(
            Finding(
                severity="LOW",
                title="Cast ops present in profile",
                detail=f"Cast ops consuming {total_cast_ms:.2f}ms — dtype mismatch at runtime.",
            )
        )

    return flags
