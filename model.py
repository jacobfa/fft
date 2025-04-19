import math
from typing import Iterable, Optional

import torch
import torch.nn as nn

from spectre import SpectreMix  # assumes spectre.py is in PYTHONPATH or same dir


class MLP(nn.Module):
    """Feed‑forward network used inside the Transformer block."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpectreBlock(nn.Module):
    """Transformer block with SPECTRE mixing instead of MHSA."""

    def __init__(
        self,
        dim: int,
        heads: int,
        max_seq_len: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        wrm_skip_ratio: float = 0.9,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mix = SpectreMix(
            dim=dim,
            heads=heads,
            max_seq_len=max_seq_len,
            enable_wrm=True,
            wrm_skip_ratio=wrm_skip_ratio,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,H,W)
        x = self.proj(x)  # (B,embed_dim,H',W')
        x = x.flatten(2).transpose(1, 2)  # (B,N,embed_dim)
        return x


class SpectreViT(nn.Module):
    """Vision Transformer with SPECTRE token mixing optimised for CIFAR‑10."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        wrm_skip_ratio: float = 0.9,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.seq_len = num_patches + 1  # + cls token

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList(
            [
                SpectreBlock(
                    dim=embed_dim,
                    heads=heads,
                    max_seq_len=self.seq_len,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    wrm_skip_ratio=wrm_skip_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,H,W)
        B = x.shape[0]
        x = self.patch_embed(x)  # (B,N,embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B,seq_len,embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


class SpectreLM(nn.Module):
    """Causal Language Model using SPECTRE token mixing."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 1024,
        embed_dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        wrm_skip_ratio: float = 0.9,
        tie_weights: bool = True,
    ) -> None:
        """Create a SpectreLM model.

        Args:
            vocab_size: Size of the input/output vocabulary.
            max_seq_len: Maximum sequence length the model will handle.
            embed_dim: Embedding dimension (also the hidden size).
            depth: Number of Spectre blocks.
            heads: Number of attention heads in SpectreMix.
            mlp_ratio: Expansion ratio for the feed‑forward network.
            drop_rate: Dropout rate applied after embeddings and within MLPs.
            wrm_skip_ratio: Ratio controlling windowed receptive masking in SpectreMix.
            tie_weights: If True, use the same weights for the input token embeddings and the output LM head.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList(
            [
                SpectreBlock(
                    dim=embed_dim,
                    heads=heads,
                    max_seq_len=max_seq_len,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    wrm_skip_ratio=wrm_skip_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        if tie_weights:
            # Tie input embedding and output projection weights
            self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        if self.lm_head.weight is not self.token_embed.weight:
            nn.init.trunc_normal_(self.lm_head.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Tensor of shape (B, L) containing token indices.

        Returns:
            Logits of shape (B, L, vocab_size).
        """
        B, L = input_ids.shape
        if L > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {L} exceeds model's maximum {self.max_seq_len}."
            )

        positions = torch.arange(0, L, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0)  # (1, L)

        x = self.token_embed(input_ids) + self.pos_embed[:, :L]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
