from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
 
import warnings, logging, os

# 1) Hide torch.* UserWarnings / FutureWarnings
warnings.filterwarnings("ignore", category=UserWarning,  module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 2) Mute the torch logger (e.g. CUDA info messages)
logging.getLogger("torch").setLevel(logging.ERROR)
 
# ---------------------------------------------------------------------
# Utility: complex helpers
# ---------------------------------------------------------------------

def view_as_complex(x: torch.Tensor) -> torch.Tensor:
    """Real-tensor view → complex-tensor view (expects last dim == 2)."""
    return torch.view_as_complex(x)           # <— no dim shuffle

def view_as_real(x: torch.Tensor) -> torch.Tensor:
    """Complex-tensor view → real view with last dim size 2."""
    return torch.view_as_real(x)


# ---------------------------------------------------------------------
# Prefix-FFT cache (KV-cache analogue)
# ---------------------------------------------------------------------

class SpectreCache:
    """
    Running real-FFT of a growing sequence.
    * prefix_fft : (Nmax//2+1, d)   complex64
    * sum_q      : (d,)             float32      (for running mean of Q)
    """

    def __init__(self, Nmax: int, d: int, device: torch.device | None = None):
        self.Nmax = Nmax
        self.d = d
        self.device = device or torch.device("cpu")

        # Pre-tabulated twiddle factors e^(−j 2π k / Nmax)
        k = torch.arange(Nmax // 2 + 1, device=self.device)
        self.base_twiddle = torch.exp(-2j * torch.pi * k / Nmax)  # (freq,)

        self.prefix_fft = torch.zeros(
            Nmax // 2 + 1, d, dtype=torch.complex64, device=self.device
        )
        self.sum_q = torch.zeros(d, device=self.device)
        self.t = 0  # current sequence length

    @torch.no_grad()
    def step(self, v_t: torch.Tensor, q_t: torch.Tensor) -> None:
        """
        Ingest one token.
        Args
        ----
        v_t : (d,) value projection of the new token (real)
        q_t : (d,) query projection of the new token (real)
        """
        phase = self.base_twiddle ** self.t                # (freq,)
        self.prefix_fft += phase[:, None] * v_t            # broadcast over d
        self.sum_q += q_t
        self.t += 1

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_fft(self) -> torch.Tensor:
        return self.prefix_fft

    def get_bar_q(self) -> torch.Tensor:
        return self.sum_q / max(self.t, 1)


# ---------------------------------------------------------------------
# SPECTRE mixing layer
# ---------------------------------------------------------------------

class SpectreLayer(nn.Module):
    """
    Frequency-domain alternative to multi-head attention.
    Parameters
    ----------
    embed_dim   : model embedding size (must be divisible by num_heads)
    num_heads   : number of heads
    max_seq_len : Nmax (fixes the size of frequency-domain parameters)
    low_rank_r  : rank of optional low-rank spectral update (0 = off)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        low_rank_r: int = 0,
    ):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len
        self.n_fft = max_seq_len // 2 + 1
        self.low_rank_r = low_rank_r

        # Projections (per head but realised as joint Linear)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Gating MLP: LN(d) → hidden → 2 * n_fft  (real + imag)
        hidden = max(32, self.head_dim)
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(self.head_dim),
            nn.Linear(self.head_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.n_fft),
        )

        if low_rank_r > 0:
            # Two lightweight MLPs that each produce (n_fft × r × 2) real numbers
            self.U_mlp = nn.Sequential(
                nn.LayerNorm(self.head_dim),
                nn.Linear(self.head_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2 * self.n_fft * low_rank_r),
            )
            self.V_mlp = nn.Sequential(
                nn.LayerNorm(self.head_dim),
                nn.Linear(self.head_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2 * self.n_fft * low_rank_r),
            )

        # Register positional phase (frequency indices)
        k = torch.arange(self.n_fft).float()                # (n_fft,)
        self.register_buffer("freq_idx", k, persistent=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,              # (B, N, embed_dim)
        cache: Optional[SpectreCache] = None,
        *,                            # keyword-only
        positions: Optional[torch.Tensor] = None,  # (N,) absolute indices
    ) -> torch.Tensor:
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim

        # ---------------------------------------------------------------
        # 1) Token projections
        # ---------------------------------------------------------------
        q = self.q_proj(x)            # (B, N, E)
        v = self.v_proj(x)

        # head split
        q = (
            q.view(B, N, H, D)
            .transpose(1, 2)          # (B, H, N, D)
            .contiguous()
        )
        v = (
            v.view(B, N, H, D)
            .transpose(1, 2)
            .contiguous()
        )

        # ---------------------------------------------------------------
        # 2) Spectral transform  (RFFT along sequence)
        # ---------------------------------------------------------------
        v_fft = torch.fft.rfft(v, dim=2)       # (B, H, n_fft, D), complex

        # ---------------------------------------------------------------
        # 3) Content-adaptive gating
        # ---------------------------------------------------------------
        # Global descriptor: mean over sequence
        bar_q = q.mean(dim=2)                  # (B, H, D)
        # Apply gate MLP to each head independently
        gate = self.gate_mlp(bar_q)            # (B, H, 2*n_fft)
        gate = gate.view(B, H, self.n_fft, 2)
        gate = view_as_complex(gate)           # complex  (B, H, n_fft)

        if self.low_rank_r > 0:
            # Optional low-rank outer product  U V^T
            U = self.U_mlp(bar_q).view(
                B, H, self.n_fft, self.low_rank_r, 2
            )
            V = self.V_mlp(bar_q).view(
                B, H, self.n_fft, self.low_rank_r, 2
            )
            U = view_as_complex(U)             # (B,H,n_fft,r)
            V = view_as_complex(V)
            gate = gate + torch.einsum("...fr,...gr->...fg", U, V)

        # Positional phase (absolute)
        if positions is None:
            # assume contiguous range [0, N-1]
            positions = torch.arange(
                N, device=x.device, dtype=torch.float32
            )

        # We need gate per freq; broadcast to D
        gate = gate[..., None]                # (B,H,n_fft,1)
        gate = gate.repeat(1, 1, 1, D)        # (B,H,n_fft,D)

        # ---------------------------------------------------------------
        # 4) Apply gate & inverse FFT
        # ---------------------------------------------------------------
        v_fft = v_fft * gate                  # complex element-wise
        v_mixed = torch.fft.irfft(
            v_fft, n=N, dim=2
        )                                     # (B,H,N,D), real

        # ---------------------------------------------------------------
        # 5) Re-assemble heads and project out
        # ---------------------------------------------------------------
        v_mixed = (
            v_mixed.transpose(1, 2)           # (B,N,H,D)
            .reshape(B, N, self.embed_dim)
        )
        out = self.out_proj(v_mixed)          # (B, N, embed_dim)
        return out
