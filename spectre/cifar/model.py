# fast_spectre.py
# ---------------------------------------------------------
# Speed‑optimised SPECTRE‑ViT (PyTorch ≥1.13, best with 2.1+)
# ---------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft, irfft
from typing import List, Optional

# ---------------------------------------------------------------------
# Global perf switches
# ---------------------------------------------------------------------
torch.backends.cudnn.benchmark = True              # autotune convs
torch.set_float32_matmul_precision("high")         # enable TF32 on A100/RTX30+
_HAS_COMPILE = hasattr(torch, "compile")           # PyTorch 2 flag

# ---------------------------------------------------------------------
# Optional fast DropPath from timm
# ---------------------------------------------------------------------
try:
    from timm.layers import DropPath                # fused CUDA kernel
except Exception:                                   # pragma: no cover
    class DropPath(nn.Module):
        """Fallback pure‑Python DropPath (stochastic depth)"""
        def __init__(self, drop_prob: float = 0.) -> None:
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1. - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(
                shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor

# ---------------------------------------------------------------------
# Wavelets (optional)
# ---------------------------------------------------------------------
try:
    from pytorch_wavelets import DWT1DForward, DWT1DInverse  # type: ignore
except ImportError:                                          # pragma: no cover
    DWT1DForward = DWT1DInverse = None

# ---------------------------------------------------------------------
# SPECTRE Mixer
# ---------------------------------------------------------------------
class SpectreMix(nn.Module):
    r"""SPECTRE mixing layer – drop‑in replacement for MHSA."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 12,
        *,
        wavelets: bool = False,
        levels: int = 3,
        wavelet: str = "db4",
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim, self.n_heads, self.dh = dim, n_heads, dim // n_heads
        self.levels = levels

        # linear projections
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

        # 2‑layer complex gate MLP
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, n_heads * ((self.dh // 2) + 1) * 2),
        )

        # wavelet refinement (optional)
        self.use_wavelets = wavelets and DWT1DForward is not None
        if self.use_wavelets:
            self.dwt = DWT1DForward(J=levels, wave=wavelet, mode="zero")
            self.idwt = DWT1DInverse(wave=wavelet, mode="zero")
            self.w_gate = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim // 2),
                nn.GELU(approximate="tanh"),
                nn.Linear(dim // 2, levels),
            )

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, D)
        B, N, _ = x.shape

        # ---- projections ---------------------------------------------------
        q = self.Wq(x)                                    # (B, N, D)
        v = self.Wv(x).view(B, N, self.n_heads, self.dh)  # (B, N, h, dh)
        v = v.permute(0, 2, 1, 3).contiguous()            # (B, h, N, dh)

        # ---- FFT in spectral domain ----------------------------------------
        Vf = rfft(v, dim=2, norm="ortho")                 # (B, h, Nf, dh)
        Nf = Vf.size(2)

        # spectral gates from pooled queries
        q_pool = q.mean(1)                                # (B, D)
        g = self.gate(q_pool).view(B, self.n_heads, Nf, 2)
        g = torch.view_as_complex(g)                      # (B, h, Nf)

        Vf.mul_(g.unsqueeze(-1))
        v_t = irfft(Vf, n=N, dim=2, norm="ortho")         # (B, h, N, dh)

        # ---- optional wavelet refinement -----------------------------------
        if self.use_wavelets:
            yl, yh = self.dwt(v_t)                        # low‑pass, highs list
            gate = torch.sigmoid(self.w_gate(q_pool)).view(
                B, 1, self.levels, 1, 1)                  # (B,1,L,1,1)
            yh = [yh_l * gate[..., i] for i, yh_l in enumerate(yh)]
            v_t = self.idwt((yl, yh))

        # ---- merge heads & project -----------------------------------------
        v_t = v_t.permute(0, 2, 1, 3).reshape(B, N, self.dim)  # (B, N, D)
        return self.proj(v_t)

# ---------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224,
                 patch: int = 16, dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        n_patches = (img_size // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,H,W)
        B = x.size(0)
        x = x.to(memory_format=torch.channels_last)      # NHWC for speed
        x = self.proj(x).flatten(2).transpose(1, 2)      # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), 1) + self.pos
        return x

# ---------------------------------------------------------------------
# Spectre Encoder block
# ---------------------------------------------------------------------
class SpectreBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        drop_path: float = 0.,
        mlp_ratio: int = 4,
        wavelets: bool = False,
        levels: int = 3,
        layer_scale_init: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mix = SpectreMix(dim, heads, wavelets=wavelets, levels=levels)
        self.ls1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop1 = DropPath(drop_path) if drop_path else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.ls2 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop2 = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.ls1 * self.mix(self.norm1(x)))
        x = x + self.drop2(self.ls2 * self.mlp(self.norm2(x)))
        return x

# ---------------------------------------------------------------------
# Vision Transformer w/ SPECTRE
# ---------------------------------------------------------------------
class SpectreViT(nn.Module):

    def __init__(
        self,
        *,
        img_size: int = 32,
        patch: int = 4,
        dim: int = 256,
        depth: int = 6,
        heads: int = 4,
        num_classes: int = 10,
        wavelets: bool = True,
        levels: int = 3,
        drop_path_rate: float = 0.1,
        mlp_ratio: int = 4,
        layer_scale_init: float = 1e-5,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch, dim)

        # stochastic‑depth decay
        dp_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            SpectreBlock(
                dim, heads, dp_rates[i], mlp_ratio,
                wavelets, levels, layer_scale_init
            ) for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes) if num_classes else nn.Identity()

        # weight init
        self.apply(self._init_weights)

    # --------------------------------------------------------------
    @staticmethod
    def _init_weights(m) -> None:  # noqa: N805
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,H,W)
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])                        # CLS token
