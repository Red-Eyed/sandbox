"""Core types for the model-builder pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch.nn as nn


@dataclass
class ExportSpec:
    """Declarative description of one model export.

    factory() returns (model, args, export_kwargs).
    The engine calls torch.onnx.export(model, args, path, dynamo=True, **export_kwargs).
    Factory is pure: no I/O, no printing, no side effects.
    """

    name: str
    group: str
    factory: Callable[[], tuple[nn.Module, tuple, dict[str, Any]]]


class SpecProvider(Protocol):
    """Any callable that returns a list of ExportSpec for one model family."""

    def __call__(self) -> list[ExportSpec]: ...
