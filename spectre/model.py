# spectre_fast_eager.py  – speed‑up, no Torch‑Compile
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch import amp
from contextlib import nullcontext

# ─── global speed knobs ──────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True      # tensor‑core matmuls for fp32
    torch.backends.cudnn.allow_tf32  = True
    torch.backends.cudnn.benchmark   = True          # pick fastest conv algo once
torch.set_float32_matmul_precision("medium")          # avoid fp32 slowdowns

# ─── external Spectre mixer  ─────────────────────────────────────────────────
from spectre import Spectre, DropPath

# ─── sinusoidal positional embedding ─────────────────────────────────────────
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, dim: int, drop: float):
        super().__init__()
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) *
                        -(math.log(10_000.0) / dim))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, L, D)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, L, D)
        return self.drop(x + self.pe[:, : x.size(1), :].to(x.dtype))

# ─── fused GELU MLP ──────────────────────────────────────────────────────────
class MLP(nn.Sequential):
    def __init__(self, dim: int, hidden: int, drop: float):
        super().__init__(
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Dropout(drop),
            nn.Linear(hidden, dim, bias=False),
            nn.Dropout(drop),
        )

# ─── Spectre transformer block ───────────────────────────────────────────────
class SpectreBlock(nn.Module):
    def __init__(
        self, dim: int, seq_len: int, n_heads: int, mlp_ratio: float,
        drop: float, dp_rate: float, *, chunk_size: int | None = None,
        checkpoint: bool = False
    ):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, 1e-6), nn.LayerNorm(dim, 1e-6)
        self.mixer = Spectre(dim, n_heads, seq_len,
                             chunk_size=chunk_size)
        self.mlp   = MLP(dim, int(dim * mlp_ratio), drop)
        self.dp1, self.dp2 = DropPath(dp_rate), DropPath(dp_rate)
        self.checkpoint = checkpoint

    def _inner(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.mixer(self.norm1(x)))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpoint and self.training:
            return nn.utils.checkpoint.checkpoint(self._inner, x, use_reentrant=False)
        return self._inner(x)

# ─── Spectre Vision Transformer ──────────────────────────────────────────────
class SpectreViT(nn.Module):
    def __init__(
        self, *, img_size: int = 32, patch_size: int = 4, in_chans: int = 3,
        num_classes: int = 10, embed_dim: int = 256, depth: int = 8,
        n_heads: int = 8, mlp_ratio: float = 2.0, drop: float = 0.1,
        dp_rate: float = 0.1, chunk_size: int | None = None,
        checkpoint: bool = False,
        memory_format: str | None = "channels_last",    # opt‑in NHWC
    ):
        super().__init__()
        self.patch = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size, bias=False)
        seq_len    = (img_size // patch_size) ** 2 + 1

        self.cls = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.pos_emb = PositionalEmbedding(seq_len, embed_dim, drop)

        dpr = torch.linspace(0, dp_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            SpectreBlock(embed_dim, seq_len, n_heads, mlp_ratio, drop, dpr[i],
                         chunk_size=chunk_size, checkpoint=checkpoint,
                         )
            for i in range(depth)
        ])

        self.norm, self.head = nn.LayerNorm(embed_dim, 1e-6), nn.Linear(embed_dim, num_classes, False)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)

        self.mformat = torch.channels_last if memory_format == "channels_last" else torch.contiguous_format

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(memory_format=self.mformat)
        ctx = amp.autocast(device_type="cuda", dtype=torch.float16) if x.is_cuda else nullcontext()
        with ctx:
            b = x.size(0)
            x = self.patch(x).flatten(2).transpose(1, 2)           # (B, N, D)
            x = torch.cat((self.cls.expand(b, -1, -1), x), 1)
            x = self.pos_emb(x)
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.norm(x)[:, 0])

# ─── Spectre Language Model ──────────────────────────────────────────────────
class SpectreLM(nn.Module):
    def __init__(
        self, vocab_size: int, *, seq_len: int = 1024, embed_dim: int = 768,
        depth: int = 12, n_heads: int = 12, mlp_ratio: float = 4.0,
        drop: float = 0.1, dp_rate: float = 0.1, chunk_size: int | None = None,
        checkpoint: bool = False, tie_weights: bool = True,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

        self.pos_emb = PositionalEmbedding(seq_len, embed_dim, drop)

        dpr = torch.linspace(0, dp_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            SpectreBlock(embed_dim, seq_len, n_heads, mlp_ratio, drop, dpr[i],
                         chunk_size=chunk_size, checkpoint=checkpoint,
                         )
            for i in range(depth)
        ])

        self.norm, self.head = nn.LayerNorm(embed_dim, 1e-6), nn.Linear(embed_dim, vocab_size, False)
        if tie_weights:
            self.head.weight = self.embed.weight

    def forward(self, tok: torch.Tensor) -> torch.Tensor:          # (B, L)
        if tok.size(1) > self.pos_emb.pe.size(1):
            raise RuntimeError(f"max seq_len {self.pos_emb.pe.size(1)}, got {tok.size(1)}")
        ctx = amp.autocast(device_type="cuda", dtype=torch.float16) if tok.is_cuda else nullcontext()
        with ctx:
            x = self.embed(tok)
            x = self.pos_emb(x)
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.norm(x))

# ─── factory helpers (original names) ────────────────────────────────────────
def create_spectre_vit(**kw): return SpectreViT(**kw)
def create_spectre_lm(**kw):  return SpectreLM(**kw)
