# model.py  ───────────────────────────────────────────────────────────────────
"""
Vision Transformer with SPECTRE token mixer, sized for CIFAR‑10 (32 × 32 RGB),
augmented with **stochastic depth** (a.k.a. DropPath).

Place `spectre.py` in the same directory.  No training / main code included.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from spectre import Spectre

# ───────────────────────────── DropPath (stochastic depth) ───────────────────────────── #
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample — identity during inference."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)    # broadcast over dims
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask.div(keep)


# ────────────────────────────────── utility modules ─────────────────────────────────── #
class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_chans: int = 3, embed_dim: int = 256):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                 # (b, d, h, w)
        return x.flatten(2).transpose(1, 2)   # (b, n, d)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class SpectreBlock(nn.Module):
    """Encoder block: LayerNorm → SPECTRE → DropPath → FFN → DropPath."""

    def __init__(self, dim: int, seq_len: int, n_heads: int,
                 mlp_ratio: float, drop: float, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = Spectre(d_model=dim, n_heads=n_heads,
                             seq_len=seq_len, wavelet=False)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.mixer(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# ───────────────────────────────────── ViT ───────────────────────────────────── #
class SpectreViT(nn.Module):
    """
    Vision Transformer using SPECTRE + stochastic depth.

    CIFAR‑10 default hyper‑params:
        img_size=32, patch_size=4, embed_dim=256, depth=8, n_heads=8
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 8,
        n_heads: int = 8,
        mlp_ratio: float = 2.0,
        drop: float = 0.1,
        drop_path_rate: float = 0.1,        # ← stochastic‑depth rate (0 = off)
    ):
        super().__init__()

        # ─ patches & positional encoding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.num_patches
        self.seq_len = n_patches + 1                          # +CLS
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
        self.pos_drop = nn.Dropout(drop)

        # ─ stochastic‑depth schedule (linear)
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()

        self.blocks = nn.ModuleList(
            [
                SpectreBlock(
                    dim=embed_dim,
                    seq_len=self.seq_len,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes, bias=False)
        self._init_weights()

    # ───────────────────────────────── weight init ───────────────────────────────── #
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)

    # ─────────────────────────────────── forward ─────────────────────────────────── #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = self.patch_embed(x)                                # (b, n, d)
        x = torch.cat([self.cls_token.expand(b, -1, -1), x], 1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])
