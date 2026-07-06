from onnxruntime.quantization import CalibrationMethod, QuantType

from onnxquant import QuantizationConfig


def test_defaults_are_symmetric_int8_w8a8() -> None:
    config = QuantizationConfig()

    assert config.activation_type == QuantType.QInt8
    assert config.weight_type == QuantType.QInt8
    assert config.activation_symmetric is True
    assert config.weight_symmetric is True
    assert config.per_channel is True


def test_to_get_qdq_config_kwargs_round_trips_every_field() -> None:
    config = QuantizationConfig(
        calibrate_method=CalibrationMethod.Percentile,
        per_channel=False,
        op_types_to_quantize=["Conv", "MatMul"],
    )

    kwargs = config.to_get_qdq_config_kwargs()

    assert kwargs["calibrate_method"] == CalibrationMethod.Percentile
    assert kwargs["per_channel"] is False
    assert kwargs["op_types_to_quantize"] == ["Conv", "MatMul"]
    assert set(kwargs) == set(QuantizationConfig.model_fields)
