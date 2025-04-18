from __future__ import annotations
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------- #
#                    differentiable DWT/iDWT from the lib               #
# --------------------------------------------------------------------- #
try:
    from pytorch_wavelets import DWT1DForward, DWT1DInverse  # ≥ 1.0.0
except ImportError as e:
    raise ImportError(
        "pytorch_wavelets not found – `pip install pytorch_wavelets`"
    ) from e  # library gives 1‑D DWT wrappers ✱&#8203;:contentReference[oaicite:0]{index=0}

# --------------------------------------------------------------------- #
#                          SPECTRE Attention Block                      #
# --------------------------------------------------------------------- #
class SPECTREAttention(nn.Module):
    """
    Frequency‑domain token mixer (§ Method).  WRM uses pytorch_wavelets.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        wavelet: bool = False,
        wavelet_levels: int = 1,
    ):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads

        # projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # spectral gates
        self.ln_qbar = nn.LayerNorm(self.dh)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.dh, self.dh),
            nn.GELU(),
            nn.Linear(self.dh, 2),
        )

        # wavelet refinement
        self.wavelet = wavelet
        if self.wavelet:
            self.w_levels = wavelet_levels
            self.wmlp = nn.Sequential(
                nn.Linear(self.dh, self.dh),
                nn.GELU(),
                nn.Linear(self.dh, self.w_levels),
            )
            self.dwt = DWT1DForward(J=self.w_levels, mode="zero", wave="haar")
            self.idwt = DWT1DInverse(mode="zero", wave="haar")

    # ------------------------------------------------------------------ #
    def _complex_gate(self, qbar: torch.Tensor, N: int) -> torch.Tensor:
        """2‑layer MLP → complex gate g ∈ ℂⁿ."""
        g_logits = self.gate_mlp(qbar).unsqueeze(1).repeat_interleave(N, 1)
        return torch.view_as_complex(g_logits)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,N,dim)
        B, N, _ = x.shape
        H, dh = self.heads, self.dh

        q = self.q_proj(x)
        v = self.v_proj(x)

        q = q.view(B, N, H, dh).transpose(1, 2).reshape(B * H, N, dh)
        v = v.view(B, N, H, dh).transpose(1, 2).reshape(B * H, N, dh)

        qbar = self.ln_qbar(q.mean(1))

        v_hat = torch.fft.fft(v, dim=1)
        g = self._complex_gate(qbar, N).unsqueeze(-1)
        v_hat = v_hat * g
        v_tilde = torch.fft.ifft(v_hat, dim=1).real

        if self.wavelet:
            v_tilde = self._wavelet_refinement(v_tilde, qbar)

        v_tilde = (
            v_tilde.reshape(B, H, N, dh).transpose(1, 2).reshape(B, N, self.dim)
        )
        return self.out_proj(v_tilde)

    # ------------------------------------------------------------------ #
        # ------------------------------------------------------------------ #
    #                   Wavelet Refinement Module (WRM)                  #
    # ------------------------------------------------------------------ #
    def _wavelet_refinement(self, x: torch.Tensor, qbar: torch.Tensor):
        """
        Apply WRM to *patch tokens only* (skip CLS) so the sequence length
        is even (64 for CIFAR‑10).  Uses 1‑D Haar DWT from pytorch_wavelets.
        """
        B_H, N, dh = x.shape
        cls, tokens = x[:, :1, :], x[:, 1:, :]          # 1 + 64

        # pytorch_wavelets expects (B,C,L)
        tokens_t = tokens.transpose(1, 2)               # (B*H, dh, 64)
        yl, yh = self.dwt(tokens_t)                     # yl, list[yh_l]

        # Gates: one scalar per level  (0 ≤ σ ≤ 1)
        gates = torch.sigmoid(self.wmlp(qbar))          # (B*H, levels)

        # Modulate each high‑pass band
        gated_yh = [
            band * gates[:, i : i + 1].unsqueeze(-1)    # (B*H,1,1) broadcast
            for i, band in enumerate(yh)
        ]

        # Reconstruct
        recon = self.idwt((yl, gated_yh))               # (B*H, dh, 64)
        recon = recon.transpose(1, 2)                   # (B*H, 64, dh)

        # Residual add & restore CLS token
        return torch.cat([cls, tokens + recon], dim=1)


# --------------------------------------------------------------------- #
#                 Standard Transformer Encoder Block                    #
# --------------------------------------------------------------------- #
class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        wavelet: bool = False,
        wavelet_levels: int = 1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mix = SPECTREAttention(
            dim, heads, wavelet=wavelet, wavelet_levels=wavelet_levels
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------- #
#                      Vision Transformer backbone                      #
# --------------------------------------------------------------------- #
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        wavelet: bool = True,
        wavelet_levels: int = 2
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_embed = nn.Conv2d(
            in_chans, dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (img_size // patch_size) ** 2  # 64 for CIFAR‑32

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim,
                    heads,
                    mlp_ratio,
                    dropout,
                    wavelet=wavelet,
                    wavelet_levels=wavelet_levels,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B,64,D)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), 1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])
