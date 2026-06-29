"""Entry point: uv run python -m build_models [--out work_dir]"""

from __future__ import annotations

import argparse
from pathlib import Path

from .engine import export_one
from .registry import all_specs


def main(out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    current_group = ""
    for spec in all_specs():
        if spec.group != current_group:
            current_group = spec.group
            print(f"\n{current_group}:")
        n = export_one(spec, out_dir)
        print(f"  {spec.name}.onnx  ({n} nodes)")

    print(f"\nAll models written to {out_dir}/")
    print("Run the optimizer on any of them:")
    print(f"  uv run speedup {out_dir}/encoder_ln_dynamic.onnx")
    print(f"  uv run speedup {out_dir}/bert_tiny.onnx")
    print(f"  uv run speedup {out_dir}/resnet18.onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent.parent / "work_dir",
        help="Directory to write .onnx files (default: work_dir/)",
    )
    args = parser.parse_args()
    main(args.out)
