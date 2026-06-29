# LayerNorm Fusion Guide

## What the tool does automatically

`fuse_layernorm` walks the graph for the 9-node chain that `torch.export` emits for
`nn.LayerNorm` and replaces it with a single `LayerNormalization` node.

Pattern matched:
```
ReduceMean(x, axis) → Sub(x, mean) → Pow(_, 2) → ReduceMean(_, axis)
  → Add(_, eps) → Sqrt → Div(sub, _) → Mul(_, weight) → Add(_, bias)
```

## When fusion fails

The pass bails out (without a partial rewrite) if any step in the chain does not
match exactly. Common causes:

1. **Extra consumer on an intermediate output** — if `sub` or `norm` is used by a
   second node (e.g. a residual skip), the use-count check fails.
   Fix: restructure the PyTorch model so the intermediate is not reused.

2. **Non-scalar eps** — if the eps initializer is a full-rank tensor instead of a
   scalar, extraction fails.
   Fix: ensure `nn.LayerNorm(eps=1e-5)` — never set eps as a buffer.

3. **Axis mismatch between the two ReduceMean nodes** — rare, but occurs when
   LN is over multiple axes and the torch exporter inserts an extra reshape.
   Fix: use `nn.LayerNorm(normalized_shape=[last_dim_only])`.

## Manual fix: replace in PyTorch before export

If the pass cannot fuse automatically, add a custom export annotation:

```python
import torch
from torch.onnx import symbolic_helper

@torch.onnx.symbolic_opset(17)
def layer_norm(g, input, normalized_shape, weight, bias, eps, cudnn_enable):
    return g.op("LayerNormalization", input, weight, bias,
                axis_i=-len(normalized_shape), epsilon_f=eps)
```

This forces a single-node LayerNorm at export time, bypassing decomposition.
