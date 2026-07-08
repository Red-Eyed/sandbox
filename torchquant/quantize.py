from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchao.quantization.pt2e import move_exported_model_to_eval
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from torchquant.config import Pt2eQuantizationConfig
from torchquant.protocols import CalibrationFn

DynamicShapes = dict[str, Any] | tuple[Any, ...] | list[Any]


def quantize_torch_model(
    model: nn.Module,
    example_args: tuple[torch.Tensor, ...],
    calibrate: CalibrationFn,
    output_path: str | Path,
    config: Pt2eQuantizationConfig | None = None,
    embed_weights: bool = True,
    dynamic_shapes: DynamicShapes | None = None,
) -> None:
    """Static W8A8 PT2E PTQ: export, annotate, calibrate, convert, then export to ONNX."""
    config = config or Pt2eQuantizationConfig()
    # dynamic_shapes must be forwarded to both export calls, or the ONNX graph re-guards
    # on the exact example_args shape even though the exported program doesn't.
    exported = torch.export.export(model.eval(), example_args, dynamic_shapes=dynamic_shapes)

    prepared = prepare_pt2e(exported.module(), config.build_quantizer())
    calibrate(prepared)
    converted = convert_pt2e(prepared)
    move_exported_model_to_eval(converted)

    torch.onnx.export(
        converted,
        example_args,
        str(output_path),
        dynamo=True,
        external_data=not embed_weights,  # False embeds weights in the .onnx file itself
        dynamic_shapes=dynamic_shapes,
    )
