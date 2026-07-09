from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
import torch
from pydantic import BaseModel
from torch import nn

from torchquant.config import Pt2eQuantizationConfig
from torchquant.quantize import (
    DynamicShapes,
    OnnxExportOptions,
    export_fp32_onnx,
    export_quantized_onnx,
)
from torchquant.report import QuantizationReport
from torchquant.report import build_report as build_onnx_report

# One set of model inputs — a tuple so multi-input models work the same as single-input ones.
Batch = tuple[torch.Tensor, ...]


class ExportPaths(BaseModel):
    """Where a run writes its two ONNX files."""

    fp32: Path
    int8: Path


class QuantizationRunResult(BaseModel):
    """Everything a run() produces: the report, the two ONNX files, and optional scores."""

    report: QuantizationReport
    fp32_onnx_path: Path
    int8_onnx_path: Path
    fp32_score: float | None = None
    int8_score: float | None = None


class QuantizationPipeline(ABC):
    """Subclass this to quantize a model — the pipeline becomes local to you.

    Implement the two required hooks and call run(output_dir):

        class MyPipeline(QuantizationPipeline):
            def get_model(self): return load_my_model()
            def get_calibration_batches(self):
                return ((images,) for images, _ in my_loader)

        result = MyPipeline().run("out/")
        print(result.report.to_markdown())

    Almost everything is a hook (Lightning-style): run() is a thin orchestrator that only
    calls self.<step>(), and every method below has a sane default. To change anything you
    override one small method — never run() itself. Override an *input* hook (get_*) to feed
    it different data or knobs; a *step* hook (export_fp32, build_report, ...) to change the
    mechanics; a *lifecycle* hook (on_*) to observe a run.
    """

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the trained module to quantize."""

    @abstractmethod
    def get_calibration_batches(self) -> Iterable[Batch]:
        """Yield input tuples; run() feeds them through the prepared model to set observers.

        Return a *fresh* iterable each call — run() re-iterates it (to derive example_args,
        to calibrate, and, by default, for parity). A generator expression over a
        re-iterable source like a DataLoader satisfies this; a one-shot iterator does not.
        """

    def get_example_args(self) -> Batch:
        """One representative input tuple used to trace the model for export.

        Defaults to the first calibration batch — a calibration input is exactly a valid
        model input, so there's normally nothing extra to specify. Override only when you
        need a different trace shape, e.g. force batch size 2 to dodge torch.export's
        0/1-specialization when your calibration batches happen to be size 1.
        """
        first_batch = next(iter(self.get_calibration_batches()), None)
        if first_batch is None:
            raise ValueError(
                "get_calibration_batches() yielded no batches; cannot derive example_args"
            )
        return first_batch

    def get_config(self) -> Pt2eQuantizationConfig:
        """Quantization knobs. Default: static W8A8, per-channel weights (best accuracy)."""
        return Pt2eQuantizationConfig.static_per_channel()

    def get_dynamic_shapes(self) -> DynamicShapes | None:
        """Which dims are dynamic, e.g. ({0: torch.export.Dim("batch")},). Default: none."""
        return None

    def get_export_options(self) -> OnnxExportOptions:
        """Export knobs (opset_version, embed_weights) shared by both exports.

        Override export_fp32/export_quantized to change the export target itself; this hook
        just carries the settings those steps read.
        """
        return OnnxExportOptions()

    def get_parity_batches(self) -> Iterable[Batch]:
        """Inputs for numeric fp32-vs-int8 parity. Defaults to reusing calibration data."""
        return self.get_calibration_batches()

    def get_output_basename(self) -> str:
        """Stem for the exported files: {basename}_fp32.onnx and {basename}_int8.onnx."""
        return "model"

    def get_num_calibration_batches(self) -> int:
        """How many batches from get_calibration_batches() to run through the observers."""
        return 20

    def get_num_parity_batches(self) -> int:
        """How many batches from get_parity_batches() to compare fp32 against int8."""
        return 10

    def evaluate(self, onnx_path: Path) -> float | None:
        """Optional domain metric (e.g. accuracy) for an exported ONNX model.

        Default: no evaluation. Override to score fp32 and int8 with your labels + metric.
        """
        return None

    def run(self, output_dir: str | Path) -> QuantizationRunResult:
        """Drive the whole pipeline end to end and return the result.

        A thin orchestrator: resolve the subclass hooks, export the fp32 baseline, quantize
        and export the int8 model, then build the report and result. Each line is a step
        hook you can override in isolation — this method itself never needs overriding.
        """
        self.on_run_start()

        paths = self.resolve_output_paths(output_dir)
        model = self.get_model().eval()
        example_args = self.get_example_args()
        dynamic_shapes = self.get_dynamic_shapes()
        config = self.get_config()
        options = self.get_export_options()

        # fp32 baseline — the untouched reference the report diffs int8 against. Quantization
        # runs on an exported copy of the graph, so reusing `model` for both exports is safe:
        # the original stays fp32.
        self.export_fp32(model, example_args, paths.fp32, dynamic_shapes, options)
        self.on_model_exported("fp32", paths.fp32)
        self.export_quantized(model, example_args, paths.int8, config, dynamic_shapes, options)
        self.on_model_exported("int8", paths.int8)

        report = self.build_report(paths.fp32, paths.int8, config)
        result = self.build_result(report, paths)

        self.on_run_end(result)
        return result

    def resolve_output_paths(self, output_dir: str | Path) -> ExportPaths:
        """Create output_dir and derive the fp32/int8 file paths. Override to change layout."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        basename = self.get_output_basename()
        return ExportPaths(
            fp32=output_dir / f"{basename}_fp32.onnx",
            int8=output_dir / f"{basename}_int8.onnx",
        )

    def export_fp32(
        self,
        model: nn.Module,
        example_args: Batch,
        output_path: Path,
        dynamic_shapes: DynamicShapes | None,
        options: OnnxExportOptions,
    ) -> None:
        """Export the unquantized reference model. Override to replace the export entirely."""
        export_fp32_onnx(
            model, example_args, output_path, options=options, dynamic_shapes=dynamic_shapes
        )

    def export_quantized(
        self,
        model: nn.Module,
        example_args: Batch,
        output_path: Path,
        config: Pt2eQuantizationConfig,
        dynamic_shapes: DynamicShapes | None,
        options: OnnxExportOptions,
    ) -> None:
        """PT2E-quantize and export the int8 model. Override for a custom quantize/export tail.

        Delegates calibration to self.calibrate, so overriding just the loop rarely needs
        touching this method.
        """
        export_quantized_onnx(
            model,
            example_args,
            self.calibrate,
            output_path,
            config=config,
            options=options,
            dynamic_shapes=dynamic_shapes,
        )

    def calibrate(self, prepared: nn.Module) -> None:
        """Populate the prepared model's observers by feeding calibration batches through it.

        Handed to export_quantized as its calibration callback. islice caps the count so a
        large or infinite calibration source doesn't force a full pass. Override to change
        how batches are fed (e.g. move them to a device first).
        """
        for batch in islice(self.get_calibration_batches(), self.get_num_calibration_batches()):
            prepared(*batch)

    def build_report(
        self,
        fp32_path: Path,
        int8_path: Path,
        config: Pt2eQuantizationConfig,
    ) -> QuantizationReport:
        """Compare the two ONNX files into a report. Override to customize what's reported.

        Reads input names from the exported int8 graph (they're assigned during export, not
        before), builds parity feeds keyed by them, then diffs size/parity/op-coverage.
        """
        input_names = self.read_input_names(int8_path)
        parity_samples = self.build_parity_samples(input_names)
        return build_onnx_report(fp32_path, int8_path, config, parity_samples)

    def read_input_names(self, onnx_path: Path) -> list[str]:
        """Read a model's ONNX input tensor names. Override if you key feeds differently."""
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        return [model_input.name for model_input in session.get_inputs()]

    def build_parity_samples(self, input_names: list[str]) -> list[dict[str, np.ndarray]]:
        """Turn parity batches into ONNX feed dicts keyed by input name.

        Each tensor in a batch is paired positionally with the model's ONNX input names, so
        multi-input models feed correctly; strict=True turns an arity mismatch into an error
        rather than silently dropping inputs. Override to reshape or preprocess feeds.
        """
        batches = islice(self.get_parity_batches(), self.get_num_parity_batches())
        return [
            {name: tensor.numpy() for name, tensor in zip(input_names, batch, strict=True)}
            for batch in batches
        ]

    def build_result(self, report: QuantizationReport, paths: ExportPaths) -> QuantizationRunResult:
        """Assemble the result, scoring each ONNX file via evaluate(). Override to add fields."""
        return QuantizationRunResult(
            report=report,
            fp32_onnx_path=paths.fp32,
            int8_onnx_path=paths.int8,
            fp32_score=self.evaluate(paths.fp32),
            int8_score=self.evaluate(paths.int8),
        )

    def on_run_start(self) -> None:
        """Called once before any work begins. Override for setup or logging."""

    def on_model_exported(self, kind: Literal["fp32", "int8"], path: Path) -> None:
        """Called after each export with its kind and file path. Override for progress logging."""

    def on_run_end(self, result: QuantizationRunResult) -> None:
        """Called once after the result is built. Override for teardown or logging."""
