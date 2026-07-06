from collections.abc import Callable
from typing import Any

from onnxruntime.quantization import CalibrationMethod, QuantType
from pydantic import BaseModel


class QuantizationConfig(BaseModel):
    """W8A8 static QDQ knobs, mapped 1:1 onto onnxruntime's get_qdq_config() kwargs.

    Defaults to symmetric signed int8 for both weights and activations, overriding
    onnxruntime's own default of asymmetric uint8 activations — that default is an
    x86-VNNI-era convention this project doesn't target.
    """

    calibrate_method: CalibrationMethod = CalibrationMethod.MinMax
    calibrate_args: dict[str, Any] | None = None
    activation_type: QuantType = QuantType.QInt8
    weight_type: QuantType = QuantType.QInt8
    activation_symmetric: bool = True
    weight_symmetric: bool | None = True
    per_channel: bool = True
    reduce_range: bool = False
    keep_removable_activations: bool = False
    min_real_range: float | None = None
    tensor_quant_overrides: dict[str, list[dict[str, Any]]] | None = None
    calibration_providers: list[str] | None = None
    op_types_to_quantize: list[str] | None = None
    nodes_to_exclude: list[str] | Callable[..., bool] | None = None
    extra_options: dict[str, Any] | None = None

    def to_get_qdq_config_kwargs(self) -> dict[str, Any]:
        return self.model_dump()
