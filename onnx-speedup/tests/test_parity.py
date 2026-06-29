"""Integration test — full pipeline on a minimal transformer block."""

import onnx
import onnx.checker

from onnx_speedup.passes.eliminate_cast import EliminateCastPass
from onnx_speedup.passes.fuse_gemm import FuseGemmPass
from onnx_speedup.passes.fuse_layernorm import FuseLayerNormPass
from onnx_speedup.verify.parity import check_parity
from tests.fixtures.builders import make_transformer_block


def test_pipeline_reduces_nodes():
    model = make_transformer_block()
    before = len(model.graph.node)

    model, _ = FuseGemmPass().apply(model)
    model, _ = FuseLayerNormPass().apply(model)
    model, _ = EliminateCastPass().apply(model)

    after = len(model.graph.node)
    assert after < before, f"Expected fewer nodes after passes: {before} → {after}"


def test_pipeline_parity():
    original = make_transformer_block()

    optimized = original
    for pass_ in (FuseGemmPass(), FuseLayerNormPass(), EliminateCastPass()):
        optimized, _ = pass_.apply(optimized)

    result = check_parity(original, optimized)
    assert result.passed, f"Parity FAILED after full pipeline: max_diff={result.max_diff:.2e}"


def test_optimized_model_is_valid():
    model = make_transformer_block()
    for pass_ in (FuseGemmPass(), FuseLayerNormPass()):
        model, _ = pass_.apply(model)
    onnx.checker.check_model(model)
