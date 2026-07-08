from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from torchquant import quantize_torch_model


class ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def test_quantize_torch_model_produces_qdq_onnx(tmp_path: Path) -> None:
    model = ToyMLP()
    example_args = (torch.randn(4, 16),)

    def calibrate(prepared: nn.Module) -> None:
        for _ in range(4):
            prepared(*example_args)

    output_path = tmp_path / "toy_mlp_int8.onnx"
    quantize_torch_model(model, example_args, calibrate, output_path)

    proto = onnx.load(str(output_path))
    op_counts = Counter(node.op_type for node in proto.graph.node)
    assert op_counts["QuantizeLinear"] > 0
    assert op_counts["DequantizeLinear"] > 0

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    # session.run() is typed to also allow SparseTensor/dict/list outputs (ONNX's
    # sequence/map types); this test's model only has dense tensor outputs, so the cast
    # documents that boundary rather than working around it.
    outputs = cast("list[np.ndarray]", session.run(None, {input_name: example_args[0].numpy()}))
    assert outputs[0].shape == (4, 8)
