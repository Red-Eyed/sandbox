"""ExportSpecs for timm CNN models."""

from __future__ import annotations

from typing import Callable

import torch

from ..protocols import ExportSpec


def _cnn_factory(model_id: str) -> Callable:
    def factory():
        import timm

        m = timm.create_model(model_id, pretrained=False).eval()
        dummy = torch.zeros(1, 3, 224, 224)
        return m, (dummy,), {"input_names": ["image"], "output_names": ["logits"]}

    return factory


def specs() -> list[ExportSpec]:
    return [
        ExportSpec(
            name="resnet18",
            group="timm",
            factory=_cnn_factory("resnet18.a1_in1k"),
        ),
        ExportSpec(
            name="mobilenetv3",
            group="timm",
            factory=_cnn_factory("mobilenetv3_small_100.lamb_in1k"),
        ),
        ExportSpec(
            name="efficientnet_b0",
            group="timm",
            factory=_cnn_factory("efficientnet_b0.ra_in1k"),
        ),
    ]
