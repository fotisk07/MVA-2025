from typing import Optional

import torch
from torch import nn


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            QuickGELU(),
            nn.Linear(hidden, width),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.ln_1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x
