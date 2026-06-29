"""ExportSpecs for stackformers transformer models.

stackformers uses PaddedInput(x, mask, abs_positions) for padded inference.
torch.export is explicitly supported — all ops are export-compatible per the
stackformers attention README (scatter/gather helpers use repeat_interleave,
not Python loops over data-dependent values).

dynamic_shapes must mirror the PaddedInput pytree so torch.export's tree
matching succeeds: PaddedInput({dim_idx: Dim(...)}, {dim_idx: Dim(...)}, None).
"""

from __future__ import annotations

from typing import Any

import torch

from ..protocols import ExportSpec


def _encoder_factory(
    norm_cfg: Any,
    ff_cfg: Any,
    *,
    dim: int,
    heads: int,
    layers: int,
    seq: int,
    dynamic: bool,
    causal: bool = False,
):
    def factory():
        import warnings

        from stackformers.attention.config import SelfAttentionConfig
        from stackformers.positional.config import NoPosEncodingConfig
        from stackformers.presets.encoder import TransformerEncoder, TransformerEncoderConfig
        from stackformers.sequence import PaddedInput
        from torch.export import Dim

        cfg = TransformerEncoderConfig(
            attn=SelfAttentionConfig(dim=dim, heads=heads, dim_head=dim // heads, causal=causal),
            ff=ff_cfg,
            norm=norm_cfg,
            pos_encoding=NoPosEncodingConfig(),
            num_layers=layers,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = TransformerEncoder(cfg).eval()

        x = torch.zeros(1, seq, dim)
        mask = torch.ones(1, seq, dtype=torch.bool)
        inp = PaddedInput(x, mask, None)

        kwargs: dict[str, Any] = {
            "input_names": ["x", "mask"],
            "output_names": ["out"],
        }
        if dynamic:
            seq_dim = Dim("seq_len", min=1, max=4096)
            kwargs["dynamic_shapes"] = (PaddedInput({1: seq_dim}, {1: seq_dim}, None),)

        return model, (inp,), kwargs

    return factory


def specs() -> list[ExportSpec]:
    from stackformers.feedforward.config import GEGLUConfig, SwiGLUConfig
    from stackformers.norm.config import LayerNormConfig, RMSNormConfig

    DIM, HEADS, LAYERS = 128, 4, 4

    return [
        ExportSpec(
            name="encoder_ln_dynamic",
            group="stackformers",
            factory=_encoder_factory(
                LayerNormConfig(dim=DIM),
                SwiGLUConfig(dim=DIM, hidden_dim=DIM * 4),
                dim=DIM,
                heads=HEADS,
                layers=LAYERS,
                seq=8,
                dynamic=True,
            ),
        ),
        ExportSpec(
            name="encoder_ln_static",
            group="stackformers",
            factory=_encoder_factory(
                LayerNormConfig(dim=DIM),
                SwiGLUConfig(dim=DIM, hidden_dim=DIM * 4),
                dim=DIM,
                heads=HEADS,
                layers=LAYERS,
                seq=32,
                dynamic=False,
            ),
        ),
        ExportSpec(
            name="encoder_rms_dynamic",
            group="stackformers",
            factory=_encoder_factory(
                RMSNormConfig(dim=DIM),
                GEGLUConfig(dim=DIM, hidden_dim=DIM * 4),
                dim=DIM,
                heads=HEADS,
                layers=LAYERS,
                seq=8,
                dynamic=True,
            ),
        ),
        ExportSpec(
            name="decoder_causal_dynamic",
            group="stackformers",
            factory=_encoder_factory(
                LayerNormConfig(dim=DIM),
                SwiGLUConfig(dim=DIM, hidden_dim=DIM * 4),
                dim=DIM,
                heads=HEADS,
                layers=LAYERS,
                seq=8,
                dynamic=True,
                causal=True,
            ),
        ),
    ]
