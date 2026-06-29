"""
Build a variety of test ONNX models for onnx-speedup analysis and demonstration.

All exports use dynamo=True (torch.export-based exporter).

Models written to work_dir/:

  Transformer variants (stackformers) — PaddedInput-based, export-compatible
  ─────────────────────────────────────────────────────────────────────────────
  encoder_ln_dynamic.onnx    — TransformerEncoder, LayerNorm+SwiGLU, dynamic seq_len
  encoder_ln_static.onnx     — TransformerEncoder, LayerNorm+SwiGLU, fixed seq_len=32
  encoder_rms_dynamic.onnx   — TransformerEncoder, RMSNorm+GEGLU, dynamic seq_len
  decoder_causal_dynamic.onnx — TransformerEncoder(causal=True), GPT-style, dynamic seq_len

  HuggingFace models
  ─────────────────────────────────────────────────────────────────────────────
  bert_tiny.onnx             — BERT-tiny (4L, 256d), dynamic seq_len

  timm CNN models
  ─────────────────────────────────────────────────────────────────────────────
  resnet18.onnx              — ResNet-18, static 224×224
  mobilenetv3.onnx           — MobileNetV3-Small, static 224×224
  efficientnet_b0.onnx       — EfficientNet-B0, static 224×224

Run:
  uv run python scripts/build_models.py
  uv run speedup work_dir/encoder_ln_dynamic.onnx
"""

from pathlib import Path

import torch
import torch.onnx
from torch.export import Dim

OUT = Path(__file__).parent.parent / "work_dir"
OUT.mkdir(exist_ok=True)


def _node_count(path: Path) -> int:
    import onnx

    return len(onnx.load(str(path)).graph.node)


# ── stackformers ──────────────────────────────────────────────────────────────
# stackformers uses PaddedInput(x, mask, abs_positions) for padded inference.
# torch.export is fully supported — all ops are export-compatible per stackformers docs.
# dynamic_shapes must mirror the PaddedInput pytree: PaddedInput({dim: Dim}, {dim: Dim}, None).


def build_stackformers_models() -> None:
    import warnings

    from stackformers.attention.config import SelfAttentionConfig
    from stackformers.feedforward.config import GEGLUConfig, SwiGLUConfig
    from stackformers.norm.config import LayerNormConfig, RMSNormConfig
    from stackformers.positional.config import NoPosEncodingConfig
    from stackformers.presets.encoder import TransformerEncoder, TransformerEncoderConfig
    from stackformers.sequence import PaddedInput

    DIM, HEADS, LAYERS = 128, 4, 4
    SEQ_STATIC = 32

    def _make_encoder(norm_cfg, ff_cfg, causal: bool = False) -> TransformerEncoder:
        cfg = TransformerEncoderConfig(
            attn=SelfAttentionConfig(dim=DIM, heads=HEADS, dim_head=DIM // HEADS, causal=causal),
            ff=ff_cfg,
            norm=norm_cfg,
            pos_encoding=NoPosEncodingConfig(),
            num_layers=LAYERS,
        )
        return TransformerEncoder(cfg).eval()

    def _export_dynamic(model: TransformerEncoder, name: str) -> None:
        seq = 8
        x = torch.zeros(1, seq, DIM)
        mask = torch.ones(1, seq, dtype=torch.bool)
        inp = PaddedInput(x, mask, None)

        seq_dim = Dim("seq_len", min=1, max=4096)
        # dynamic_shapes must be the same NamedTuple type as the input
        dyn = (PaddedInput({1: seq_dim}, {1: seq_dim}, None),)

        path = OUT / f"{name}.onnx"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            torch.onnx.export(
                model,
                (inp,),
                str(path),
                dynamo=True,
                input_names=["x", "mask"],
                output_names=["out"],
                dynamic_shapes=dyn,
            )
        print(f"  {name}.onnx  ({_node_count(path)} nodes)")

    def _export_static(model: TransformerEncoder, name: str) -> None:
        x = torch.zeros(1, SEQ_STATIC, DIM)
        mask = torch.ones(1, SEQ_STATIC, dtype=torch.bool)
        inp = PaddedInput(x, mask, None)

        path = OUT / f"{name}.onnx"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            torch.onnx.export(
                model,
                (inp,),
                str(path),
                dynamo=True,
                input_names=["x", "mask"],
                output_names=["out"],
            )
        print(f"  {name}.onnx  ({_node_count(path)} nodes)")

    print("stackformers:")
    _export_dynamic(
        _make_encoder(LayerNormConfig(dim=DIM), SwiGLUConfig(dim=DIM, hidden_dim=DIM * 4)),
        "encoder_ln_dynamic",
    )
    _export_static(
        _make_encoder(LayerNormConfig(dim=DIM), SwiGLUConfig(dim=DIM, hidden_dim=DIM * 4)),
        "encoder_ln_static",
    )
    _export_dynamic(
        _make_encoder(RMSNormConfig(dim=DIM), GEGLUConfig(dim=DIM, hidden_dim=DIM * 4)),
        "encoder_rms_dynamic",
    )
    # GPT-style causal decoder via TransformerEncoder(causal=True)
    _export_dynamic(
        _make_encoder(
            LayerNormConfig(dim=DIM),
            SwiGLUConfig(dim=DIM, hidden_dim=DIM * 4),
            causal=True,
        ),
        "decoder_causal_dynamic",
    )


# ── HuggingFace BERT ──────────────────────────────────────────────────────────


def build_bert_tiny() -> None:
    from transformers import BertConfig, BertModel

    cfg = BertConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=64,
        vocab_size=1000,
    )
    model = BertModel(cfg).eval()

    seq_len = 16
    input_ids = torch.zeros(1, seq_len, dtype=torch.long)
    attn_mask = torch.ones(1, seq_len, dtype=torch.long)

    seq_dim = Dim("seq_len", min=1, max=64)
    dynamic_shapes = ({1: seq_dim}, {1: seq_dim})

    path = OUT / "bert_tiny.onnx"
    torch.onnx.export(
        model,
        (input_ids, attn_mask),
        str(path),
        dynamo=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_shapes=dynamic_shapes,
    )
    print("\nHuggingFace BERT:")
    print(f"  bert_tiny.onnx  ({_node_count(path)} nodes)")


# ── timm CNNs ─────────────────────────────────────────────────────────────────


def build_timm_models() -> None:
    import timm

    models = [
        ("resnet18", "resnet18.a1_in1k"),
        ("mobilenetv3", "mobilenetv3_small_100.lamb_in1k"),
        ("efficientnet_b0", "efficientnet_b0.ra_in1k"),
    ]
    print("\ntimm CNNs:")
    for name, model_id in models:
        m = timm.create_model(model_id, pretrained=False).eval()
        dummy = torch.zeros(1, 3, 224, 224)
        path = OUT / f"{name}.onnx"
        torch.onnx.export(
            m,
            dummy,
            str(path),
            dynamo=True,
            input_names=["image"],
            output_names=["logits"],
        )
        print(f"  {name}.onnx  ({_node_count(path)} nodes)")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_stackformers_models()
    build_bert_tiny()
    build_timm_models()

    print(f"\nAll models written to {OUT}/")
    print("Run the optimizer on any of them:")
    print("  uv run speedup work_dir/encoder_ln_dynamic.onnx")
    print("  uv run speedup work_dir/bert_tiny.onnx")
    print("  uv run speedup work_dir/resnet18.onnx")
