"""Export engine: runs one ExportSpec and returns the node count."""

from __future__ import annotations

from pathlib import Path

import torch.onnx

from .protocols import ExportSpec


def export_one(spec: ExportSpec, out_dir: Path) -> int:
    """Export one model to out_dir/<name>.onnx; return its node count."""
    import onnx

    model, args, kwargs = spec.factory()
    path = out_dir / f"{spec.name}.onnx"
    torch.onnx.export(model, args, str(path), dynamo=True, **kwargs)
    return len(onnx.load(str(path)).graph.node)
