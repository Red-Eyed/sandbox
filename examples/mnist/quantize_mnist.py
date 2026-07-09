from collections.abc import Iterable
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort
import torch
import torchmetrics
from lit_model import LitMNISTNet
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from torchquant import DynamicShapes, QuantizationPipeline

HERE = Path(__file__).parent
NUM_CLASSES = 10


def load_test_loader(batch_size: int = 64) -> DataLoader:
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_set = MNIST(str(HERE / "data"), train=False, transform=tfm)
    return DataLoader(test_set, batch_size=batch_size)


def onnx_accuracy(onnx_path: Path, loader: DataLoader) -> float:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES)
    for images, labels in loader:
        # session.run() is typed to also allow SparseTensor/dict/list outputs (ONNX's
        # sequence/map types); this model only has a dense tensor output, so the cast
        # documents that boundary rather than working around it.
        outputs = cast("list[np.ndarray]", session.run(None, {input_name: images.numpy()}))
        metric(torch.from_numpy(outputs[0]), labels)
    return metric.compute().item()


class MnistQuantizationPipeline(QuantizationPipeline):
    def __init__(self, checkpoint_path: Path, loader: DataLoader) -> None:
        self.checkpoint_path = checkpoint_path
        self.loader = loader

    def get_output_basename(self) -> str:
        return "mnist_net"

    def get_model(self) -> nn.Module:
        lit_model = LitMNISTNet.load_from_checkpoint(str(self.checkpoint_path), map_location="cpu")
        return lit_model.net.eval()

    def get_dynamic_shapes(self) -> DynamicShapes:
        return ({0: torch.export.Dim("batch")},)

    def get_calibration_batches(self) -> Iterable[tuple[torch.Tensor, ...]]:
        return ((images,) for images, _ in self.loader)

    def evaluate(self, onnx_path: Path) -> float:
        return onnx_accuracy(onnx_path, self.loader)


def main() -> None:
    loader = load_test_loader()
    pipeline = MnistQuantizationPipeline(HERE / "checkpoints" / "mnist_net.ckpt", loader)
    result = pipeline.run(HERE / "checkpoints")

    print(result.report.to_markdown())
    print()
    if result.fp32_score is not None:
        print(f"fp32 accuracy: {result.fp32_score:.4f}")
    if result.int8_score is not None:
        print(f"int8 accuracy: {result.int8_score:.4f}")


if __name__ == "__main__":
    main()
