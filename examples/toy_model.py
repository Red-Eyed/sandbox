import torch
import torch.nn.functional as F
from torch import nn

INPUT_SHAPE = (1, 3, 32, 32)


class ToyConvAttnNet(nn.Module):
    """Conv stem + one pre-norm transformer block.

    Deliberately mixes op families that dynamo decomposes heavily (LayerNorm,
    SDPA, GELU all expand into many primitive ops) with the plain Conv/Linear
    that don't, so the quantization pipeline is exercised against exactly the
    kind of decomposed graph this library targets.
    """

    def __init__(self, embed_dim: int = 8, num_heads: int = 2) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.conv = nn.Conv2d(3, embed_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp_fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.mlp_fc2 = nn.Linear(embed_dim * 4, embed_dim)

        self.head = nn.Linear(embed_dim, 10)

    def _attention(self, tokens: torch.Tensor) -> torch.Tensor:
        b, seq_len, embed_dim = tokens.shape
        qkv = self.qkv(tokens).reshape(b, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(b, seq_len, embed_dim)
        return self.out_proj(attn_out)

    def _mlp(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.mlp_fc1(tokens))
        return self.mlp_fc2(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        tokens = x.flatten(2).transpose(1, 2)  # (batch, h*w, embed_dim)

        tokens = tokens + self._attention(self.norm1(tokens))
        tokens = tokens + self._mlp(self.norm2(tokens))

        return self.head(tokens.mean(dim=1))


def export_to_onnx(output_path: str) -> None:
    model = ToyConvAttnNet().eval()
    example_input = torch.randn(*INPUT_SHAPE)
    torch.onnx.export(
        model,
        (example_input,),
        output_path,
        dynamo=True,
        input_names=["input"],
        output_names=["logits"],
        opset_version=20,
    )
