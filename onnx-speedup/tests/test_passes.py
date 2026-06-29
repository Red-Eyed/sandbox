"""Unit tests — one per pass, using minimal ONNX fixtures."""

import onnx

from onnx_speedup.passes.eliminate_cast import EliminateCastPass
from onnx_speedup.passes.fuse_gelu import FuseGeluPass
from onnx_speedup.passes.fuse_gemm import FuseGemmPass
from onnx_speedup.passes.fuse_layernorm import FuseLayerNormPass
from onnx_speedup.verify.parity import check_parity
from tests.fixtures.builders import (
    make_gelu_graph,
    make_layernorm_graph,
    make_matmul_add_graph,
    make_redundant_cast_graph,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _op_types(model: onnx.ModelProto) -> list[str]:
    return [n.op_type for n in model.graph.node]


def _assert_parity(original: onnx.ModelProto, optimized: onnx.ModelProto) -> None:
    result = check_parity(original, optimized)
    assert result.passed, f"Parity FAILED: max_diff={result.max_diff:.2e}, error={result.error}"


# ── FuseGemmPass ──────────────────────────────────────────────────────────────


def test_fuse_gemm_applies():
    model = make_matmul_add_graph()
    pass_ = FuseGemmPass()
    new_model, result = pass_.apply(model)
    assert result.applied
    assert "Gemm" in _op_types(new_model)
    assert "MatMul" not in _op_types(new_model)
    assert "Add" not in _op_types(new_model)


def test_fuse_gemm_parity():
    model = make_matmul_add_graph()
    new_model, _ = FuseGemmPass().apply(model)
    _assert_parity(model, new_model)


def test_fuse_gemm_no_double_fuse():
    """Applying the pass twice should yield no additional fusions."""
    model = make_matmul_add_graph()
    model1, r1 = FuseGemmPass().apply(model)
    model2, r2 = FuseGemmPass().apply(model1)
    assert r1.applied
    assert not r2.applied


# ── FuseGeluPass ──────────────────────────────────────────────────────────────


def test_fuse_gelu_applies():
    model = make_gelu_graph()
    new_model, result = FuseGeluPass().apply(model)
    assert result.applied
    ops = _op_types(new_model)
    assert "Gelu" in ops
    assert "Erf" not in ops
    assert "Div" not in ops


def test_fuse_gelu_parity():
    model = make_gelu_graph()
    new_model, _ = FuseGeluPass().apply(model)
    # com.microsoft.Gelu requires ORT — run parity via ORT session
    _assert_parity(model, new_model)


# ── FuseLayerNormPass ─────────────────────────────────────────────────────────


def test_fuse_layernorm_applies():
    model = make_layernorm_graph()
    new_model, result = FuseLayerNormPass().apply(model)
    assert result.applied, f"Pass did not apply: {result.description}"
    ops = _op_types(new_model)
    assert "LayerNormalization" in ops
    assert "ReduceMean" not in ops
    assert "Sqrt" not in ops


def test_fuse_layernorm_parity():
    model = make_layernorm_graph()
    new_model, result = FuseLayerNormPass().apply(model)
    assert result.applied
    _assert_parity(model, new_model)


def test_fuse_layernorm_no_match_on_plain_graph():
    """A plain MatMul+Add graph should not trigger LN fusion."""
    model = make_matmul_add_graph()
    _, result = FuseLayerNormPass().apply(model)
    assert not result.applied


# ── EliminateCastPass ─────────────────────────────────────────────────────────


def test_eliminate_cast_applies():
    model = make_redundant_cast_graph()
    new_model, result = EliminateCastPass().apply(model)
    assert result.applied
    assert "Cast" not in _op_types(new_model)
    assert "Relu" in _op_types(new_model)


def test_eliminate_cast_parity():
    model = make_redundant_cast_graph()
    new_model, _ = EliminateCastPass().apply(model)
    _assert_parity(model, new_model)


def test_eliminate_cast_no_false_positive():
    """Cast float32→int64 must NOT be eliminated."""
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
    out = helper.make_tensor_value_info("out", TensorProto.INT64, [4])
    node = helper.make_node("Cast", ["x"], ["out"], to=TensorProto.INT64)
    graph = helper.make_graph([node], "cast_test", [x], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    _, result = EliminateCastPass().apply(model)
    assert not result.applied
