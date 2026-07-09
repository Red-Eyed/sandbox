import torch
import torch.nn.functional as F
from torch import nn


class MNISTNet(nn.Module):
    """Conv/BatchNorm/residual stem feeding a self-attention + MLP block.

    Deliberately covers a broad op mix (Conv2d, BatchNorm2d, MaxPool2d,
    AdaptiveAvgPool2d, residual Add, Dropout, LayerNorm, Linear, SDPA, ReLU, GELU) so the
    quantization pipeline gets exercised against more than a toy Linear-only model.
    """

    def __init__(self, num_classes: int = 10, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, embed_dim, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(embed_dim)

        self.res_conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.res_bn1 = nn.BatchNorm2d(embed_dim)
        self.res_conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.res_bn2 = nn.BatchNorm2d(embed_dim)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.1)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_out = nn.Linear(embed_dim, embed_dim)

        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp_fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.mlp_fc2 = nn.Linear(embed_dim * 4, embed_dim)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def _conv_stem(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)))

    def _residual_block(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.res_bn1(self.res_conv1(x)))
        y = self.res_bn2(self.res_conv2(y))
        return F.relu(y + x)

    def _attention_block(self, tokens: torch.Tensor) -> torch.Tensor:
        b, seq_len, c = tokens.shape
        qkv = self.qkv(self.attn_norm(tokens)).reshape(b, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, seq_len, c)
        return tokens + self.attn_out(attn)

    def _mlp_block(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.mlp_fc1(self.mlp_norm(tokens)))
        return tokens + self.mlp_fc2(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_stem(x)
        x = self._residual_block(x)
        x = self.dropout(self.pool(x))

        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self._attention_block(tokens)
        tokens = self._mlp_block(tokens)
        spatial = tokens.transpose(1, 2).reshape(b, c, h, w)

        pooled = self.global_pool(spatial).flatten(1)
        x = F.relu(self.fc1(pooled))
        return self.fc2(x)
