from pydantic import BaseModel, ConfigDict
from torchao.quantization.pt2e.quantizer import Quantizer
from torchao.testing.pt2e._xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)


class Pt2eQuantizationConfig(BaseModel):
    """PT2E quantization knobs. See README for why XNNPACKQuantizer is the default."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    is_per_channel: bool = True
    is_dynamic: bool = False
    quantizer: Quantizer | None = None

    @classmethod
    def static_per_channel(cls) -> "Pt2eQuantizationConfig":
        """Static W8A8, per-channel weight scales — best accuracy, the default."""
        return cls(is_per_channel=True, is_dynamic=False)

    @classmethod
    def static_per_tensor(cls) -> "Pt2eQuantizationConfig":
        """Static W8A8, one weight scale per tensor — for backends without per-channel kernels."""
        return cls(is_per_channel=False, is_dynamic=False)

    @classmethod
    def dynamic_per_channel(cls) -> "Pt2eQuantizationConfig":
        """Static int8 weights (per-channel); activations quantized dynamically, no calibration."""
        return cls(is_per_channel=True, is_dynamic=True)

    @classmethod
    def dynamic_per_tensor(cls) -> "Pt2eQuantizationConfig":
        """Static int8 weights (per-tensor); activations quantized dynamically, no calibration."""
        return cls(is_per_channel=False, is_dynamic=True)

    def build_quantizer(self) -> Quantizer:
        if self.quantizer is not None:
            return self.quantizer
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(
            get_symmetric_quantization_config(
                is_per_channel=self.is_per_channel, is_dynamic=self.is_dynamic
            )
        )
        return quantizer
