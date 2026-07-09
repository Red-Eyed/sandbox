from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from torchquant import QuantizationPipeline


class ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class ToyMLPPipeline(QuantizationPipeline):
    # No __init__ on the base, so the subclass takes whatever it needs — here the model
    # itself — and get_model() just surfaces it, mirroring real usage.
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def get_output_basename(self) -> str:
        return "toy_mlp"

    def get_num_calibration_batches(self) -> int:
        return 4

    def get_model(self) -> nn.Module:
        return self.model

    def get_calibration_batches(self) -> Iterable[tuple[torch.Tensor, ...]]:
        return [(torch.randn(4, 16),) for _ in range(4)]


def test_quantization_pipeline_produces_qdq_onnx(tmp_path: Path) -> None:
    result = ToyMLPPipeline(ToyMLP()).run(tmp_path)

    proto = onnx.load(str(result.int8_onnx_path))
    op_counts = Counter(node.op_type for node in proto.graph.node)
    assert op_counts["QuantizeLinear"] > 0
    assert op_counts["DequantizeLinear"] > 0

    session = ort.InferenceSession(str(result.int8_onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    # session.run() is typed to also allow SparseTensor/dict/list outputs (ONNX's
    # sequence/map types); this test's model only has dense tensor outputs, so the cast
    # documents that boundary rather than working around it.
    sample = np.random.randn(4, 16).astype(np.float32)
    outputs = cast("list[np.ndarray]", session.run(None, {input_name: sample}))
    assert outputs[0].shape == (4, 8)
