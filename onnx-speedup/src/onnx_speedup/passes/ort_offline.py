import os
import tempfile

import onnx
import onnxruntime as ort

from onnx_speedup.passes.base import Pass, PassResult


class OrtOfflinePass(Pass):
    name = "ort_offline"
    description = "ORT graph optimization (ORT_ENABLE_ALL) — run last"

    def apply(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, PassResult]:
        before = len(model.graph.node)
        input_path = output_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                input_path = f.name
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                output_path = f.name

            onnx.save(model, input_path)

            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.optimized_model_filepath = output_path
            ort.InferenceSession(input_path, so, providers=["CPUExecutionProvider"])

            optimized = onnx.load(output_path)
            after = len(optimized.graph.node)
            removed = before - after
            return optimized, PassResult(
                applied=removed > 0,
                description=f"ORT optimization: {before} → {after} nodes"
                if removed > 0
                else "ORT optimization: no additional changes",
                nodes_before=before,
                nodes_after=after,
            )
        except Exception as exc:
            return model, PassResult(
                applied=False,
                description="ORT offline optimization failed",
                nodes_before=before,
                nodes_after=before,
                error=str(exc),
            )
        finally:
            for path in (input_path, output_path):
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
