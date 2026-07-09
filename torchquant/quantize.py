from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel
from torch import nn
from torchao.quantization.pt2e import move_exported_model_to_eval
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from torchquant.config import Pt2eQuantizationConfig
from torchquant.protocols import CalibrationFn

DynamicShapes = dict[str, Any] | tuple[Any, ...] | list[Any]


class OnnxExportOptions(BaseModel):
    """Knobs forwarded to torch.onnx.export, shared by the fp32 and int8 exports.

    dynamo stays on internally and isn't exposed as a toggle: this pipeline's decomposed,
    PT2E-quantized graph only round-trips to ONNX through the dynamo exporter, so flipping
    it off would break the int8 path rather than offer a real choice.
    """

    opset_version: int | None = None  # None lets torch pick its default opset
    embed_weights: bool = True  # False writes weights to a sidecar file instead of the .onnx


def _onnx_export(
    module: nn.Module,
    example_args: tuple[torch.Tensor, ...],
    output_path: str | Path,
    *,
    options: OnnxExportOptions,
    dynamic_shapes: DynamicShapes | None,
) -> None:
    # dynamic_shapes must be forwarded on every export call, or the ONNX graph re-guards on
    # the exact example_args shape even when the exported program itself doesn't.
    torch.onnx.export(
        module,
        example_args,
        str(output_path),
        dynamo=True,
        opset_version=options.opset_version,
        external_data=not options.embed_weights,
        dynamic_shapes=dynamic_shapes,
    )


def export_fp32_onnx(
    model: nn.Module,
    example_args: tuple[torch.Tensor, ...],
    output_path: str | Path,
    *,
    options: OnnxExportOptions | None = None,
    dynamic_shapes: DynamicShapes | None = None,
) -> None:
    """Export the unquantized model to ONNX — the fp32 baseline for parity/size comparison."""
    _onnx_export(
        model.eval(),
        example_args,
        output_path,
        options=options or OnnxExportOptions(),
        dynamic_shapes=dynamic_shapes,
    )


def export_quantized_onnx(
    model: nn.Module,
    example_args: tuple[torch.Tensor, ...],
    calibrate: CalibrationFn,
    output_path: str | Path,
    *,
    config: Pt2eQuantizationConfig | None = None,
    options: OnnxExportOptions | None = None,
    dynamic_shapes: DynamicShapes | None = None,
) -> None:
    """Static W8A8 PT2E PTQ: export, annotate, calibrate, convert, then export to ONNX."""
    config = config or Pt2eQuantizationConfig()
    exported = torch.export.export(model.eval(), example_args, dynamic_shapes=dynamic_shapes)

    prepared = prepare_pt2e(exported.module(), config.build_quantizer())
    calibrate(prepared)
    converted = convert_pt2e(prepared)
    move_exported_model_to_eval(converted)

    _onnx_export(
        converted,
        example_args,
        output_path,
        options=options or OnnxExportOptions(),
        dynamic_shapes=dynamic_shapes,
    )
