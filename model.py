from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from spectre import SPECTREBlock, PrefixFFTCache

# -----------------------------------------------------------------------------
# Patch‑Embedding helper for ViT
# -----------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Convolutional patch embedding (stride = patch_size)."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W)
        x = self.proj(x)  # (B, D, H/P, W/P)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, D)
        return x


# -----------------------------------------------------------------------------
# Spectre Vision Transformer (CIFAR‑10)
# -----------------------------------------------------------------------------


class SpectreViT(nn.Module):
    """Vision Transformer using SPECTRE token mixer (tailored to 32×32 CIFAR‑10)."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 384,
        depth: int = 12,
        n_heads: int = 6,
        mlp_ratio: int = 4,
        use_wavelet: bool = False,
    ) -> None:
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.0)

        self.blocks = nn.ModuleList(
            [
                SPECTREBlock(
                    d_model=embed_dim,
                    n_heads=n_heads,
                    ffn_hidden=mlp_ratio,
                    max_seq_len=num_patches + 1,
                    use_wavelet=use_wavelet,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    # -----------------------------------------------------------------
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W)
        B = x.size(0)
        x = self.patch_embed(x)  # (B, N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x, _ = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


# -----------------------------------------------------------------------------
# Spectre Language Model (GPT‑like)
# -----------------------------------------------------------------------------


class SpectreLM(nn.Module):
    """Causal language model that stacks SPECTRE blocks."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        mlp_ratio: int = 4,
        pad_id: Optional[int] = None,
        share_gates: bool = True,
        use_wavelet: bool = False,
    ) -> None:
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList(
            [
                SPECTREBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_hidden=mlp_ratio,
                    max_seq_len=max_seq_len,
                    share_gates=share_gates,
                    use_wavelet=use_wavelet,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embed.weight

        nn.init.trunc_normal_(self.pos_embed, std=0.01)
        self.apply(self._init_weights)

    # -----------------------------------------------------------------
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialization that respects module type."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    # -----------------------------------------------------------------
    def init_caches(self, device: torch.device) -> List[PrefixFFTCache]:
        """Create fresh Prefix‑FFT caches—one per transformer block."""
        return [blk.spectre.init_cache(device) for blk in self.blocks]

    # -----------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,  # (B, L)
        caches: Optional[List[PrefixFFTCache]] = None,
        incremental: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[PrefixFFTCache]]]:
        B, L = idx.shape
        device = idx.device

        x = self.embed(idx) + self.pos_embed[:, :L]
        x = self.drop(x)

        if incremental:
            if caches is None:
                caches = self.init_caches(device)
            new_caches: List[PrefixFFTCache] = []
            for blk, cache in zip(self.blocks, caches):
                x, cache = blk(x, cache=cache, incremental_state=True)
                new_caches.append(cache)
        else:
            new_caches = None
            for blk in self.blocks:
                x, _ = blk(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_caches
