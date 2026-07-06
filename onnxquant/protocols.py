from onnxruntime.quantization import CalibrationDataReader

# Re-exported rather than redefined: onnxruntime's ABC already uses a __subclasshook__
# that structurally matches any object exposing get_next(), so it *is* the minimal
# protocol callers need to implement — no separate abstraction needed on top of it.
__all__ = ["CalibrationDataReader"]
