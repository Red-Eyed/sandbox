"""Minimal ONNX graph builders for unit tests."""

import math

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_matmul_add_graph(
    x_shape: list[int] = [2, 4],
    w_shape: list[int] = [4, 8],
    bias_shape: list[int] = [8],
) -> onnx.ModelProto:
    """MatMul(x, W) → Add(result, bias)"""
    W = numpy_helper.from_array(np.random.randn(*w_shape).astype(np.float32), name="W")
    bias = numpy_helper.from_array(np.random.randn(*bias_shape).astype(np.float32), name="bias")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [x_shape[0], w_shape[1]])

    matmul = helper.make_node("MatMul", ["x", "W"], ["mm_out"])
    add = helper.make_node("Add", ["mm_out", "bias"], ["out"])

    graph = helper.make_graph([matmul, add], "test_gemm", [x], [out], [W, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def make_gelu_graph(x_shape: list[int] = [2, 16]) -> onnx.ModelProto:
    """Torch-decomposed GeLU: x → Div(sqrt2) → Erf → Add(1) → Mul(x) → Mul(0.5)"""
    sqrt2 = numpy_helper.from_array(np.array(math.sqrt(2.0), dtype=np.float32), name="sqrt2")
    one = numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="one")
    half = numpy_helper.from_array(np.array(0.5, dtype=np.float32), name="half")

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, x_shape)

    nodes = [
        helper.make_node("Div", ["x", "sqrt2"], ["div_out"]),
        helper.make_node("Erf", ["div_out"], ["erf_out"]),
        helper.make_node("Add", ["erf_out", "one"], ["add_out"]),
        helper.make_node("Mul", ["x", "add_out"], ["mul1_out"]),
        helper.make_node("Mul", ["mul1_out", "half"], ["out"]),
    ]
    graph = helper.make_graph(nodes, "test_gelu", [x], [out], [sqrt2, one, half])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def make_layernorm_graph(
    x_shape: list[int] = [2, 8, 16],
    norm_shape: list[int] = [16],
) -> onnx.ModelProto:
    """Torch-decomposed LayerNorm over last dim."""
    weight = numpy_helper.from_array(np.ones(norm_shape, dtype=np.float32), name="ln_weight")
    bias = numpy_helper.from_array(np.zeros(norm_shape, dtype=np.float32), name="ln_bias")
    eps = numpy_helper.from_array(np.array(1e-5, dtype=np.float32), name="eps")
    two = numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="two")

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, x_shape)

    nodes = [
        helper.make_node("ReduceMean", ["x"], ["mean"], axes=[-1], keepdims=1),
        helper.make_node("Sub", ["x", "mean"], ["sub"]),
        helper.make_node("Pow", ["sub", "two"], ["pow"]),
        helper.make_node("ReduceMean", ["pow"], ["var"], axes=[-1], keepdims=1),
        helper.make_node("Add", ["var", "eps"], ["var_eps"]),
        helper.make_node("Sqrt", ["var_eps"], ["std"]),
        helper.make_node("Div", ["sub", "std"], ["norm"]),
        helper.make_node("Mul", ["norm", "ln_weight"], ["scaled"]),
        helper.make_node("Add", ["scaled", "ln_bias"], ["out"]),
    ]
    graph = helper.make_graph(nodes, "test_layernorm", [x], [out], [weight, bias, eps, two])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def make_redundant_cast_graph(x_shape: list[int] = [2, 8]) -> onnx.ModelProto:
    """x → Cast(float32→float32) → Relu"""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, x_shape)

    nodes = [
        helper.make_node("Cast", ["x"], ["cast_out"], to=TensorProto.FLOAT),
        helper.make_node("Relu", ["cast_out"], ["out"]),
    ]
    # Add value_info for cast output so EliminateCastPass knows its dtype
    cast_vi = helper.make_tensor_value_info("cast_out", TensorProto.FLOAT, x_shape)
    graph = helper.make_graph(nodes, "test_cast", [x], [out])
    graph.value_info.append(cast_vi)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def make_transformer_block(
    seq_len: int = 8,
    d_model: int = 32,
    n_heads: int = 4,
) -> onnx.ModelProto:
    """
    Minimal transformer-like block for integration tests:
      x → MatMul+Add (Q,K,V projections) → LayerNorm chain
    Not a faithful attention implementation — just enough nodes to exercise passes.
    """
    batch = 1

    inits = []
    nodes = []

    def add_init(name: str, arr: np.ndarray) -> None:
        inits.append(numpy_helper.from_array(arr.astype(np.float32), name=name))

    # Q, K, V projection weights + biases
    for proj in ("q", "k", "v"):
        add_init(f"W_{proj}", np.random.randn(d_model, d_model))
        add_init(f"b_{proj}", np.random.randn(d_model))
        nodes.append(helper.make_node("MatMul", ["x", f"W_{proj}"], [f"mm_{proj}"]))
        nodes.append(helper.make_node("Add", [f"mm_{proj}", f"b_{proj}"], [f"{proj}"]))

    # LayerNorm chain after Q
    add_init("ln_weight", np.ones(d_model))
    add_init("ln_bias", np.zeros(d_model))
    add_init("eps", np.array(1e-5))
    add_init("two", np.array(2.0))

    nodes += [
        helper.make_node("ReduceMean", ["q"], ["mean"], axes=[-1], keepdims=1),
        helper.make_node("Sub", ["q", "mean"], ["sub"]),
        helper.make_node("Pow", ["sub", "two"], ["pow"]),
        helper.make_node("ReduceMean", ["pow"], ["var"], axes=[-1], keepdims=1),
        helper.make_node("Add", ["var", "eps"], ["var_eps"]),
        helper.make_node("Sqrt", ["var_eps"], ["std"]),
        helper.make_node("Div", ["sub", "std"], ["norm"]),
        helper.make_node("Mul", ["norm", "ln_weight"], ["scaled"]),
        helper.make_node("Add", ["scaled", "ln_bias"], ["out"]),
    ]

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, seq_len, d_model])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [batch, seq_len, d_model])

    graph = helper.make_graph(nodes, "transformer_block", [x], [out], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model
