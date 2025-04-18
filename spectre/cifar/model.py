import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft, irfft
from typing import Optional, Tuple, List

try:
    from pytorch_wavelets import DWT1DForward, DWT1DInverse  # type: ignore
except ImportError:  # graceful degradation when wavelet lib is missing
    DWT1DForward = DWT1DInverse = None  # pragma: no cover

# -----------------------------------------------------------------------------
# Regularisation helpers
# -----------------------------------------------------------------------------
class DropPath(nn.Module):
    """Stochastic depth per sample (a.k.a. DropPath) as in Huang et al. 2016.
    Copied from timm with minimal deps to avoid external requirement.
    """
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast over all dims
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarise
        return x.div(keep_prob) * random_tensor

# -----------------------------------------------------------------------------
# SPECTRE Mixer
# -----------------------------------------------------------------------------
class SpectreMix(nn.Module):
    """SPECTRE mixing layer – drop‑in replacement for multi‑head self‑attention.

    Args:
        dim: embedding dimension
        n_heads: number of heads
        wavelets: enable wavelet refinement
        levels: wavelet decomposition levels (ignored if wavelets=False)
        wavelet: wavelet family string, e.g. "db4", "haar"
    """

    def __init__(self, dim: int, n_heads: int = 12, *, wavelets: bool = False,
                 levels: int = 3, wavelet: str = "db4") -> None:
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
            nn.GELU(),
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
                nn.GELU(),
                nn.Linear(dim // 2, levels),
            )

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, D)
        B, N, _ = x.shape
        q = self.Wq(x)                                   # (B, N, D)
        v = self.Wv(x)

        # reshape to (B, h, N, dh)
        v = v.reshape(B, N, self.n_heads, self.dh).transpose(1, 2)

        # FFT along sequence dimension
        Vf = rfft(v, dim=-2, norm="ortho")              # (B, h, Nf, dh)
        Nf = Vf.size(-2)

        # spectral gates from pooled queries
        q_pool = q.mean(dim=1)                           # (B, D)
        g = self.gate(q_pool).view(B, self.n_heads, Nf, 2)
        g = torch.view_as_complex(g)                     # (B, h, Nf)

        Vf = Vf * g.unsqueeze(-1)
        v_t = irfft(Vf, n=N, dim=-2, norm="ortho")      # (B, h, N, dh)

        # Wavelet refinement
        if self.use_wavelets:
            yl, yh = self.dwt(v_t)                       # low‑pass, highs list
            gate = torch.sigmoid(self.w_gate(q_pool)).view(B, 1, self.levels, 1, 1)
            yh = [yh_l * gate[..., i] for i, yh_l in enumerate(yh)]
            v_t = self.idwt((yl, yh))

        # merge heads and project out
        v_t = v_t.transpose(1, 2).reshape(B, N, self.dim)
        return self.proj(v_t)

# -----------------------------------------------------------------------------
# Patch embedding
# -----------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch: int = 16, dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        n_patches = (img_size // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 3, H, W)
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)      # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos
        return x

# -----------------------------------------------------------------------------
# Transformer block with SPECTRE + MLP + LayerScale + DropPath
# -----------------------------------------------------------------------------
class SpectreBlock(nn.Module):
    def __init__(self, dim: int, heads: int, drop_path: float = 0.0,
                 mlp_ratio: int = 4, wavelets: bool = False,
                 levels: int = 3, layer_scale_init: float = 1e-5) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mix = SpectreMix(dim, heads, wavelets=wavelets, levels=levels)
        self.ls1 = nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True)
        self.drop1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.ls2 = nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True)
        self.drop2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.ls1 * self.mix(self.norm1(x)))
        x = x + self.drop2(self.ls2 * self.mlp(self.norm2(x)))
        return x

# -----------------------------------------------------------------------------
# Vision Transformer with SPECTRE
# -----------------------------------------------------------------------------
class SpectreViT(nn.Module):
    """Vision Transformer backbone with SPECTRE mixers.

    Args:
        img_size: input resolution (square assumed)
        patch: patch size
        dim: embedding dimension
        depth: number of encoder blocks
        heads: number of heads per block
        num_classes: classification classes
        wavelets: enable wavelet module
        levels: wavelet levels
        drop_path_rate: maximum stochastic depth rate (linearly scaled)
        mlp_ratio: hidden expansion in feed‑forward
        layer_scale_init: LayerScale initial value (0 disables LayerScale)
    """

    def __init__(self, *, img_size: int = 32, patch: int = 4, dim: int = 256,
                 depth: int = 6, heads: int = 4, num_classes: int = 10,
                 wavelets: bool = True, levels: int = 3,
                 drop_path_rate: float = 0.1, mlp_ratio: int = 4,
                 layer_scale_init: float = 1e-5) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch, dim)

        # stochastic depth decay rule
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            SpectreBlock(dim, heads, dp_rates[i], mlp_ratio, wavelets, levels, layer_scale_init)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

        # weight init (trunc normal for linear layers)
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(m) -> None:  # noqa: N805
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,H,W)
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls = x[:, 0]  # CLS token
        return self.head(cls)
