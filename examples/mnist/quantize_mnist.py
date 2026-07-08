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

from torchquant import DynamicShapes, Pt2eQuantizationConfig, build_report, quantize_torch_model

HERE = Path(__file__).parent
NUM_CALIBRATION_BATCHES = 20


def load_test_loader(batch_size: int = 64) -> DataLoader:
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_set = MNIST(str(HERE / "data"), train=False, transform=tfm)
    return DataLoader(test_set, batch_size=batch_size)


def onnx_accuracy(onnx_path: Path, loader: DataLoader, num_classes: int = 10) -> float:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    for images, labels in loader:
        # session.run() is typed to also allow SparseTensor/dict/list outputs (ONNX's
        # sequence/map types); this model only has a dense tensor output, so the cast
        # documents that boundary rather than working around it.
        outputs = cast("list[np.ndarray]", session.run(None, {input_name: images.numpy()}))
        metric(torch.from_numpy(outputs[0]), labels)
    return metric.compute().item()


def load_trained_model(checkpoint_path: Path) -> nn.Module:
    lit_model = LitMNISTNet.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
    return lit_model.net.eval()


def export_fp32_baseline(
    model: nn.Module,
    example_args: tuple[torch.Tensor, ...],
    dynamic_shapes: DynamicShapes,
    output_path: Path,
) -> None:
    torch.onnx.export(
        model,
        example_args,
        str(output_path),
        dynamo=True,
        external_data=False,
        dynamic_shapes=dynamic_shapes,
    )


def make_parity_samples(
    loader: DataLoader, input_name: str, num_batches: int = 10
) -> list[dict[str, np.ndarray]]:
    samples = []
    for i, (images, _) in enumerate(loader):
        if i >= num_batches:
            break
        samples.append({input_name: images.numpy()})
    return samples


def main() -> None:
    model = load_trained_model(HERE / "checkpoints" / "mnist_net.ckpt")
    test_loader = load_test_loader()

    # Batch size 2, not 1: torch.export's 0/1-specialization quirk silently treats a
    # dim of size 1 in the example as static even when marked dynamic. A real DataLoader's
    # uneven trailing batch means the batch dim must be dynamic regardless.
    example_args = (torch.randn(2, 1, 28, 28),)
    dynamic_shapes = ({0: torch.export.Dim("batch")},)

    def calibrate(prepared: nn.Module) -> None:
        for i, (images, _) in enumerate(test_loader):
            if i >= NUM_CALIBRATION_BATCHES:
                break
            prepared(images)

    config = Pt2eQuantizationConfig.static_per_channel()
    int8_path = HERE / "checkpoints" / "mnist_net_int8.onnx"
    quantize_torch_model(
        model, example_args, calibrate, int8_path, config=config, dynamic_shapes=dynamic_shapes
    )

    fp32_path = HERE / "checkpoints" / "mnist_net_fp32.onnx"
    export_fp32_baseline(model, example_args, dynamic_shapes, fp32_path)

    int8_session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    input_name = int8_session.get_inputs()[0].name
    parity_samples = make_parity_samples(test_loader, input_name)
    report = build_report(fp32_path, int8_path, config, parity_samples)

    print(report.to_markdown())
    print()
    print(f"fp32 accuracy: {onnx_accuracy(fp32_path, test_loader):.4f}")
    print(f"int8 accuracy: {onnx_accuracy(int8_path, test_loader):.4f}")


if __name__ == "__main__":
    main()
