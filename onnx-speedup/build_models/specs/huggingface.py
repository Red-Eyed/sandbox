"""ExportSpecs for HuggingFace models."""

from __future__ import annotations

import torch

from ..protocols import ExportSpec


def _bert_tiny_factory():
    from torch.export import Dim
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
    return (
        model,
        (input_ids, attn_mask),
        {
            "input_names": ["input_ids", "attention_mask"],
            "output_names": ["last_hidden_state", "pooler_output"],
            "dynamic_shapes": ({1: seq_dim}, {1: seq_dim}),
        },
    )


def specs() -> list[ExportSpec]:
    return [
        ExportSpec(name="bert_tiny", group="HuggingFace", factory=_bert_tiny_factory),
    ]
