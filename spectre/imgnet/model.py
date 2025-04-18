from __future__ import annotations
from typing import List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------- #
#                          utility: DropPath                            #
# --------------------------------------------------------------------- #
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False):
    """Per‑sample stochastic depth (identical to timm implementation)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)          # broadcast
    rand = torch.rand(shape, dtype=x.dtype, device=x.device)
    mask = rand < keep_prob
    return x * mask / keep_prob


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# --------------------------------------------------------------------- #
#                    differentiable DWT/iDWT from the lib               #
# --------------------------------------------------------------------- #
try:
    from pytorch_wavelets import DWT1DForward, DWT1DInverse  # ≥ 1.0.0
except ImportError as e:
    raise ImportError("pytorch_wavelets not found – `pip install pytorch_wavelets`") from e

# --------------------------------------------------------------------- #
#                          SPECTRE Attention Block                      #
# --------------------------------------------------------------------- #
class SPECTREAttention(nn.Module):
    """
    Frequency‑domain mixer (§ Method).  WRM uses pytorch_wavelets.
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
        g_logits = self.gate_mlp(qbar).unsqueeze(1).repeat_interleave(N, 1)
        return torch.view_as_complex(g_logits)  # (B*H, N)

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
    #                   Wavelet Refinement Module (WRM)                  #
    # ------------------------------------------------------------------ #
    def _wavelet_refinement(self, x: torch.Tensor, qbar: torch.Tensor):
        B_H, N, dh = x.shape
        cls, tokens = x[:, :1, :], x[:, 1:, :]          # CLS + 64

        tokens_t = tokens.transpose(1, 2)               # (B*H, dh, 64)
        yl, yh = self.dwt(tokens_t)

        gates = torch.sigmoid(self.wmlp(qbar))          # (B*H, levels)
        gated_yh = [
            band * gates[:, i : i + 1].unsqueeze(-1) for i, band in enumerate(yh)
        ]

        recon = self.idwt((yl, gated_yh)).transpose(1, 2)  # (B*H, 64, dh)
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
        drop_path: float = 0.0,
        wavelet: bool = False,
        wavelet_levels: int = 1,
        init_values: float | None = None,   # LayerScale γ initial value
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

        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

        # LayerScale parameters (if requested)
        if init_values is not None and init_values > 0:
            self.gamma1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma1 = self.gamma2 = None

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma1 is None:
            x = x + self.drop_path1(self.mix(self.norm1(x)))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma1 * self.mix(self.norm1(x)))
            x = x + self.drop_path2(self.gamma2 * self.mlp(self.norm2(x)))
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
        drop_path_rate: float = 0.0,
        wavelet: bool = True,
        wavelet_levels: int = 2,
        init_values: float | None = None,  # LayerScale
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_embed = nn.Conv2d(
            in_chans, dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        # stochastic‑depth decay rule (0 → drop_path_rate)
        dpr = [drop_path_rate * i / (depth - 1) for i in range(depth)]

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim,
                    heads,
                    mlp_ratio,
                    dropout,
                    drop_path=dpr[i],
                    wavelet=wavelet,
                    wavelet_levels=wavelet_levels,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    # ------------------------------------------------------------------ #
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B,N,D)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), 1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])
