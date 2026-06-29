import onnx
import onnxslim

from onnx_speedup.passes.base import Pass, PassResult


class OnnxSlimPass(Pass):
    name = "onnxslim"
    description = "onnxslim — dead-node elimination, constant folding, shape inference"

    def apply(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, PassResult]:
        before = len(model.graph.node)
        try:
            slimmed = onnxslim.slim(model)
            after = len(slimmed.graph.node)
            removed = before - after
            pct = removed / before * 100 if before else 0
            return slimmed, PassResult(
                applied=removed > 0,
                description=f"{before} → {after} nodes (-{pct:.0f}%)"
                if removed > 0
                else "no nodes removed",
                nodes_before=before,
                nodes_after=after,
            )
        except Exception as exc:
            return model, PassResult(
                applied=False,
                description="onnxslim failed",
                nodes_before=before,
                nodes_after=before,
                error=str(exc),
            )
