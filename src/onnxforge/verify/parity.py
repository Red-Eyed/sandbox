from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto


@dataclass
class ParityResult:
    passed: bool
    max_diff: float
    atol: float
    rtol: float
    n_samples: int
    error: str | None = None


def check_parity(
    original: onnx.ModelProto,
    optimized: onnx.ModelProto,
    n_samples: int = 3,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    debug_dir: Path | None = None,
) -> ParityResult:
    try:
        orig_sess = _make_session(original)
        opt_sess = _make_session(optimized)
        input_specs = _get_input_specs(original)

        max_diff = 0.0
        for seed in range(n_samples):
            feeds = _generate_inputs(input_specs, seed)
            orig_outs = orig_sess.run(None, feeds)
            opt_outs = opt_sess.run(None, feeds)

            for orig_out, opt_out in zip(orig_outs, opt_outs):
                diff = np.max(np.abs(orig_out.astype(np.float32) - opt_out.astype(np.float32)))
                max_diff = max(max_diff, float(diff))

            if max_diff > atol:
                if debug_dir is not None:
                    _save_debug(feeds, orig_outs, opt_outs, debug_dir, seed)
                return ParityResult(
                    passed=False,
                    max_diff=max_diff,
                    atol=atol,
                    rtol=rtol,
                    n_samples=seed + 1,
                )

        return ParityResult(
            passed=True, max_diff=max_diff, atol=atol, rtol=rtol, n_samples=n_samples
        )

    except Exception as exc:
        return ParityResult(
            passed=False,
            max_diff=float("inf"),
            atol=atol,
            rtol=rtol,
            n_samples=0,
            error=str(exc),
        )


def _make_session(model: onnx.ModelProto) -> ort.InferenceSession:
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    try:
        onnx.save(model, path)
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _get_input_specs(model: onnx.ModelProto) -> list[dict]:
    specs = []
    for inp in model.graph.input:
        tt = inp.type.tensor_type
        dtype = tt.elem_type if tt.HasField("elem_type") else TensorProto.FLOAT
        shape = []
        if tt.HasField("shape"):
            for dim in tt.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(8)  # concrete fallback for dynamic dims
        else:
            shape = [1, 8]
        specs.append({"name": inp.name, "dtype": dtype, "shape": shape})
    return specs


def _generate_inputs(specs: list[dict], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feeds: dict[str, np.ndarray] = {}
    for spec in specs:
        dtype = spec["dtype"]
        shape = spec["shape"]
        if dtype in (TensorProto.FLOAT, TensorProto.DOUBLE):
            arr = rng.standard_normal(shape).astype(np.float32)
        elif dtype == TensorProto.FLOAT16:
            arr = rng.standard_normal(shape).astype(np.float16)
        elif dtype in (TensorProto.INT8, TensorProto.INT32, TensorProto.INT64):
            arr = rng.integers(0, 128, size=shape, dtype=np.int64)
        elif dtype == TensorProto.BOOL:
            arr = rng.integers(0, 2, size=shape, dtype=np.bool_)
        else:
            arr = rng.standard_normal(shape).astype(np.float32)
        feeds[spec["name"]] = arr
    return feeds


def _save_debug(
    feeds: dict[str, np.ndarray],
    orig_outs: list[np.ndarray],
    opt_outs: list[np.ndarray],
    debug_dir: Path,
    seed: int,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in feeds.items():
        np.save(debug_dir / f"seed{seed}_input_{name}.npy", arr)
    for i, (o, p) in enumerate(zip(orig_outs, opt_outs)):
        np.save(debug_dir / f"seed{seed}_orig_out{i}.npy", o)
        np.save(debug_dir / f"seed{seed}_opt_out{i}.npy", p)
