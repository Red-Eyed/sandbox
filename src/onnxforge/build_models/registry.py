"""Model registry: lists all SpecProviders in export order.

To add a new model family:
  1. Create src/onnxforge/build_models/specs/<family>.py with specs() -> list[ExportSpec].
  2. Import it here and add it to _PROVIDERS.
"""

from __future__ import annotations

from .protocols import ExportSpec, SpecProvider
from .specs import huggingface, stackformers, timm_cnns

_PROVIDERS: list[SpecProvider] = [
    stackformers.specs,
    huggingface.specs,
    timm_cnns.specs,
]


def all_specs() -> list[ExportSpec]:
    return [spec for provider in _PROVIDERS for spec in provider()]
