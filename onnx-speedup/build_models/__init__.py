"""build_models — declarative ONNX model builder for onnx-speedup demos.

Usage:
    uv run python -m build_models          # write to work_dir/
    uv run python -m build_models --out /path/to/dir

Extending:
    Add a new specs/<family>.py with specs() -> list[ExportSpec].
    Register it in registry.py — one line, no other changes needed.
"""
