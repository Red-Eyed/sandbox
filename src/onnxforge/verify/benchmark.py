"""Latency benchmarking stub — desktop ORT only in v1."""

from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime as ort

from onnxforge.verify.parity import _generate_inputs, _get_input_specs


@dataclass
class BenchResult:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    n_runs: int


def benchmark(
    model: onnx.ModelProto,
    n_warmup: int = 5,
    n_runs: int = 50,
) -> BenchResult:
    import os
    import tempfile
    import time

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    try:
        onnx.save(model, path)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    specs = _get_input_specs(model)
    feeds = _generate_inputs(specs, seed=0)

    for _ in range(n_warmup):
        sess.run(None, feeds)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return BenchResult(
        mean_ms=float(arr.mean()),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        n_runs=n_runs,
    )
