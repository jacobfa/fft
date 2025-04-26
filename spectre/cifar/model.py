# models.py
# ---------------------------------------------------------------------
# Two illustrative Transformer-style models that use SPECTRE
#  • SpectreTransformerLM  – a GPT-like language model
#  • SpectreViT_CIFAR10    – a ViT-style vision model tuned for CIFAR-10
# ---------------------------------------------------------------------
# Save this file next to spectre.py and import the classes you need.
# No training / main script is provided.
# ---------------------------------------------------------------------

from __future__ import annotations
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from spectre import SpectreLayer               # ← your previous file


# ---------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------

class MLP(nn.Module):
    """Simple feed-forward network used inside encoder blocks."""
    def __init__(self, dim: int, hidden_mult: int = 4, p: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(dim, hidden_mult * dim)
        self.lin2 = nn.Linear(hidden_mult * dim, dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.lin2(self.act(self.lin1(x))))


class SpectreEncoderLayer(nn.Module):
    """LN-SPECTRE-drop + LN-MLP-drop (pre-norm)."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        low_rank_r: int = 0,
        mlp_ratio: int = 4,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mixer = SpectreLayer(
            embed_dim, num_heads, max_seq_len, low_rank_r
        )
        self.drop1 = nn.Dropout(p_drop)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, p_drop)
        self.drop2 = nn.Dropout(p_drop)

    def forward(
        self,
        x: torch.Tensor,             # (B, N, E)
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.drop1(self.mixer(self.ln1(x), positions=positions))
        x = x + self.drop2(self.mlp(self.ln2(x)))
        return x


# ---------------------------------------------------------------------
# 1) Language model
# ---------------------------------------------------------------------

class SpectreTransformerLM(nn.Module):
    """
    GPT-style left-to-right LM using SPECTRE.
    Defaults are relatively small; scale as needed.
    """
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 1024,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        low_rank_r: int = 0,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, embed_dim)

        self.blocks = nn.ModuleList(
            SpectreEncoderLayer(
                embed_dim,
                num_heads,
                max_seq_len,
                low_rank_r,
                mlp_ratio=4,
                p_drop=p_drop,
            )
            for _ in range(depth)
        )
        self.ln_out = nn.LayerNorm(embed_dim)
        # Weight-tied language modelling head
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        self.drop = nn.Dropout(p_drop)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx : (B, N) int tokens
        Returns logits (B, N, vocab_size)
        """
        B, N = idx.shape
        assert (
            N <= self.max_seq_len
        ), "sequence exceeds model's max_seq_len"

        pos = torch.arange(N, device=idx.device)

        h = self.token_emb(idx)                  # (B, N, E)
        h = self.drop(h)

        for blk in self.blocks:
            h = blk(h, positions=pos)

        h = self.ln_out(h)
        logits = self.head(h)                    # (B, N, V)
        return logits


# ---------------------------------------------------------------------
# 2) Vision Transformer for CIFAR-10
# ---------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    img (B,3,H,W) → (B, N_patches, embed_dim)
    """
    def __init__(self, img_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            3, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, E, H/P, W/P) → (B, N, E)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SpectreViT(nn.Module):
    """
    ViT-style classifier with SPECTRE token mixer.
    Defaults chosen for CIFAR-10 (32×32).
    """
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        num_classes: int = 10,
        low_rank_r: int = 0,
        p_drop: float = 0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        n_patches = self.patch_embed.num_patches
        self.seq_len = n_patches + 1                     # +1 for CLS

        # CLS token
        self.cls_param = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Learnable 1-D positional embedding (will be converted to phase)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))

        self.blocks = nn.ModuleList(
            SpectreEncoderLayer(
                embed_dim,
                num_heads,
                max_seq_len=self.seq_len,
                low_rank_r=low_rank_r,
                mlp_ratio=4,
                p_drop=p_drop,
            )
            for _ in range(depth)
        )

        self.ln_out = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Init
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_param, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, 32, 32)
        returns logits (B, 10)
        """
        B = x.shape[0]
        patches = self.patch_embed(x)                     # (B, Np, E)

        cls = self.cls_param.expand(B, -1, -1)           # (B,1,E)
        h = torch.cat([cls, patches], dim=1) + self.pos_emb

        positions = torch.arange(self.seq_len, device=x.device)

        for blk in self.blocks:
            h = blk(h, positions=positions)

        h = self.ln_out(h)
        cls_final = h[:, 0]                               # CLS token
        return self.head(cls_final)
