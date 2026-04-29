import torch
import torch.nn.functional as F
from torch import nn

from clip_reproduction.models.transformer import ResidualAttentionBlock


class ByteTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    vocab_size = 259

    def __init__(self, context_length: int = 77) -> None:
        self.context_length = context_length

    def encode(self, text: str) -> torch.Tensor:
        byte_ids = [b + 3 for b in text.encode("utf-8")]
        token_ids = [self.bos_id] + byte_ids[: self.context_length - 2] + [self.eos_id]
        if len(token_ids) < self.context_length:
            token_ids += [self.pad_id] * (self.context_length - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)

    def batch_encode(self, texts: list[str]) -> torch.Tensor:
        return torch.stack([self.encode(t) for t in texts], dim=0)


class TextTransformerCLIP(nn.Module):
    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 259,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        embed_dim: int = 512,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        normalize: bool = True,
        eos_id: int = 2,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.width = width
        self.normalize = normalize
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))

        self.transformer = nn.ModuleList(
            [
                ResidualAttentionBlock(width, heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(layers)
            ]
        )
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, embed_dim))

        self.register_buffer("causal_mask", self._build_causal_mask(context_length), persistent=False)
        self._init_parameters()

    @staticmethod
    def _build_causal_mask(context_length: int) -> torch.Tensor:
        mask = torch.full((context_length, context_length), float("-inf"))
        return torch.triu(mask, diagonal=1)

    def _init_parameters(self) -> None:
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=self.width**-0.5)

    def _end_of_text_positions(self, token_ids: torch.Tensor) -> torch.Tensor:
        eos_mask = token_ids.eq(self.eos_id)
        has_eos = eos_mask.any(dim=1)
        eos_pos = eos_mask.float().argmax(dim=1)
        fallback = token_ids.ne(self.pad_id).sum(dim=1).clamp(min=1) - 1
        return torch.where(has_eos, eos_pos, fallback)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError(f"Expected token_ids shape (B, T), got {tuple(token_ids.shape)}")
        if token_ids.size(1) != self.context_length:
            raise ValueError(
                f"Expected context length {self.context_length}, got {token_ids.size(1)}"
            )

        x = self.token_embedding(token_ids)
        x = x + self.positional_embedding.unsqueeze(0)

        attn_mask = self.causal_mask.to(dtype=x.dtype, device=x.device)
        for block in self.transformer:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_final(x)
        eot_positions = self._end_of_text_positions(token_ids)
        x = x[torch.arange(x.shape[0], device=x.device), eot_positions]
        x = x @ self.text_projection

        if self.normalize:
            x = F.normalize(x, dim=-1)

        return x
