"""Programmatic optimization API — the functional core of onnxforge.

This module owns the guarded pass pipeline: it takes an in-memory
``onnx.ModelProto``, runs each pass with the parity/validation guard, and
returns the optimized model plus a structured result.  It performs **no disk
I/O and no printing** — progress is reported through an injected
:class:`PassObserver`, and file loading/saving lives in the thin
:func:`optimize_file` shell (and in ``cli.py``).

Integrate it into an export routine directly::

    import onnx
    from onnxforge import optimize, OptimizeConfig

    model = onnx.load("model.onnx")  # or build in-memory from torch.onnx.export
    result = optimize(model, OptimizeConfig(target_opset=23))
    onnx.save(result.model, "model_optimized.onnx")
"""

from pathlib import Path
from typing import Protocol, Sequence

import onnx
import onnx.checker
import onnx.version_converter
from pydantic import BaseModel, ConfigDict

from onnxforge.diagnose import graph_analyzer
from onnxforge.passes.base import Pass, PassResult
from onnxforge.passes.fold_transpose import FoldTransposePass
from onnxforge.passes.fuse_rms_norm import FuseRMSNormPass
from onnxforge.passes.fuse_skip_layernorm import FuseSkipLayerNormPass
from onnxforge.passes.fuse_skip_rms_norm import FuseSkipRMSNormPass
from onnxforge.passes.onnxslim_pass import OnnxSlimPass
from onnxforge.passes.ort_offline import OrtOfflinePass
from onnxforge.passes.pack_projections import PackProjectionsPass
from onnxforge.verify.parity import ParityResult, check_parity

# Order is load-bearing — see CLAUDE.md.  Packing runs before ORT so ORT fuses
# the larger GEMM; SkipLayerNorm runs after ORT to mop up residual Add+LN pairs.
DEFAULT_PIPELINE: tuple[Pass, ...] = (
    FuseRMSNormPass(),  # before onnxslim: opset ≥ 23 only; gamma not yet absorbed
    FuseSkipRMSNormPass(),  # before onnxslim: any opset; fuses Add+RMSNorm pairs
    OnnxSlimPass(),
    FoldTransposePass(),
    PackProjectionsPass(),  # before ORT: pack QKV/gated-FF so ORT fuses the larger GEMM
    OrtOfflinePass(),
    FuseSkipLayerNormPass(),  # after ORT: mop up residual Add+LN that ORT's fuser skipped
)


class OptimizeConfig(BaseModel):
    """Configuration for :func:`optimize`.

    ``target_opset`` converts the model's standard-domain opset to the given
    version (via ``onnx.version_converter``) as the first step, which also
    unlocks opset-gated passes (e.g. RMSNorm fusion needs ≥ 23).  ``None``
    leaves the source opset untouched.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    target_opset: int | None = None
    passes: Sequence[Pass] = DEFAULT_PIPELINE
    run_parity: bool = True
    parity_samples: int = 3
    parity_atol: float = 1e-4
    parity_rtol: float = 1e-4


class OptimizeResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: onnx.ModelProto
    pass_results: list[tuple[str, PassResult]]
    parity: ParityResult
    graph_before: graph_analyzer.GraphAnalysis
    graph_after: graph_analyzer.GraphAnalysis


class PassObserver(Protocol):
    """Reports pipeline progress.  The core calls these; presentation (printing,
    logging) lives in the implementer so the core stays I/O-free."""

    def on_pass_start(self, step: int, total: int, pass_: Pass) -> None: ...

    def on_pass_skipped(self, step: int, pass_: Pass, reason: str) -> None: ...

    def on_pass_result(self, step: int, pass_: Pass, result: PassResult) -> None: ...

    def on_opset_converted(self, source: int, target: int, ok: bool) -> None: ...


class _NullObserver:
    """No-op observer — the default when a caller does not want progress events."""

    def on_pass_start(self, step: int, total: int, pass_: Pass) -> None: ...

    def on_pass_skipped(self, step: int, pass_: Pass, reason: str) -> None: ...

    def on_pass_result(self, step: int, pass_: Pass, result: PassResult) -> None: ...

    def on_opset_converted(self, source: int, target: int, ok: bool) -> None: ...


def optimize(
    model: onnx.ModelProto,
    config: OptimizeConfig = OptimizeConfig(),
    observer: PassObserver | None = None,
) -> OptimizeResult:
    """Run the guarded optimization pipeline on an in-memory model.

    Does not mutate ``model``.  Each pass is reverted (not fatal) if it fails
    validation or the parity check, so the returned model is always at least as
    correct as the input.
    """
    obs = observer or _NullObserver()
    original = model
    graph_before = graph_analyzer.analyze(original)

    current = _apply_target_opset(original, config, obs)

    pass_results: list[tuple[str, PassResult]] = []
    total = len(config.passes)
    for step, pass_ in enumerate(config.passes, 1):
        obs.on_pass_start(step, total, pass_)
        current, result = _run_pass(current, pass_, config, obs, step)
        pass_results.append((pass_.name, result))

    parity = check_parity(
        original,
        current,
        n_samples=config.parity_samples,
        atol=config.parity_atol,
        rtol=config.parity_rtol,
    )
    return OptimizeResult(
        model=current,
        pass_results=pass_results,
        parity=parity,
        graph_before=graph_before,
        graph_after=graph_analyzer.analyze(current),
    )


def optimize_file(
    model_path: Path,
    output_path: Path | None = None,
    config: OptimizeConfig = OptimizeConfig(),
    observer: PassObserver | None = None,
) -> OptimizeResult:
    """Thin shell over :func:`optimize` that loads from and saves to disk."""
    model = onnx.load(str(model_path))
    result = optimize(model, config, observer)
    out = output_path or model_path.with_name(model_path.stem + "_optimized.onnx")
    onnx.save(result.model, str(out))
    return result


def _std_opset(model: onnx.ModelProto) -> int:
    return next(
        (i.version for i in model.opset_import if (i.domain or "") == ""),
        0,
    )


def _apply_target_opset(
    model: onnx.ModelProto,
    config: OptimizeConfig,
    obs: PassObserver,
) -> onnx.ModelProto:
    """Convert the standard-domain opset to ``config.target_opset``, guarded by a
    parity check.  Returns the original model unchanged if conversion is a no-op,
    raises, or breaks parity."""
    target = config.target_opset
    source = _std_opset(model)
    if target is None or target == source:
        return model

    try:
        converted = onnx.version_converter.convert_version(model, target)
        onnx.checker.check_model(converted)
    except Exception:
        obs.on_opset_converted(source, target, ok=False)
        return model

    if config.run_parity:
        parity = check_parity(
            model,
            converted,
            n_samples=config.parity_samples,
            atol=config.parity_atol,
            rtol=config.parity_rtol,
        )
        if not parity.passed:
            obs.on_opset_converted(source, target, ok=False)
            return model

    obs.on_opset_converted(source, target, ok=True)
    return converted


def _run_pass(
    current: onnx.ModelProto,
    pass_: Pass,
    config: OptimizeConfig,
    obs: PassObserver,
    step: int,
) -> tuple[onnx.ModelProto, PassResult]:
    """Apply one pass under the validation/parity guard.  Returns the model to
    carry forward (unchanged on any failure) and the pass result."""
    n_nodes = len(current.graph.node)
    std_opset = _std_opset(current)

    if pass_.min_std_opset > 1 and std_opset < pass_.min_std_opset:
        reason = f"requires opset ≥ {pass_.min_std_opset}, model has {std_opset}"
        result = PassResult(False, reason, n_nodes, n_nodes)
        obs.on_pass_skipped(step, pass_, reason)
        return current, result

    if not pass_.is_applicable(current):
        result = PassResult(False, "not applicable", n_nodes, n_nodes)
        obs.on_pass_skipped(step, pass_, "not applicable")
        return current, result

    try:
        new_model, result = pass_.apply(current)
    except Exception as exc:
        result = PassResult(False, str(exc), n_nodes, n_nodes, error=str(exc))
        obs.on_pass_result(step, pass_, result)
        return current, result

    if result.error:
        obs.on_pass_result(step, pass_, result)
        return current, result

    try:
        onnx.checker.check_model(new_model)
    except Exception as exc:
        result.applied = False
        result.error = f"post-pass validation failed: {exc}"
        obs.on_pass_result(step, pass_, result)
        return current, result

    if result.applied and config.run_parity:
        parity = check_parity(
            current,
            new_model,
            n_samples=config.parity_samples,
            atol=config.parity_atol,
            rtol=config.parity_rtol,
        )
        if not parity.passed:
            result.applied = False
            result.error = (
                f"parity check failed (max_diff={parity.max_diff:.2e}, "
                f"atol={parity.atol:.2e}): {parity.error or ''}"
            )
            obs.on_pass_result(step, pass_, result)
            return current, result

    obs.on_pass_result(step, pass_, result)
    return (new_model if result.applied else current), result
