# spectre.py ───────────────────────────────────────────────────────────
# Memory-efficient SPECTRE token mixer + prefix-FFT cache
# (full-sequence & autoregressive modes)
# ----------------------------------------------------------------------

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────
# Helpers for real⇄complex views
# ──────────────────────────────────────────────────────────────────────
def view_as_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_complex(x)          # last-dim must be 2

def view_as_real(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(x)             # append last-dim = 2


# ──────────────────────────────────────────────────────────────────────
# Prefix-FFT cache  (KV-cache analogue)
# ──────────────────────────────────────────────────────────────────────
class SpectreCache:
    """
    Stores a running RFFT and mean(q) for one autoregressive sequence.
    prefix_fft : (H, n_fft, D)  complex64
    sum_q      : (H, D)         float32
    """
    def __init__(
        self,
        Nmax: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | None = None,
    ):
        self.Nmax = Nmax
        self.H, self.D = num_heads, head_dim
        self.n_fft = Nmax // 2 + 1
        self.device = device or torch.device("cpu")

        k = torch.arange(self.n_fft, device=self.device)
        self.base_twiddle = torch.exp(-2j * torch.pi * k / Nmax)  # (freq,)

        self.prefix_fft = torch.zeros(
            self.H, self.n_fft, self.D, dtype=torch.complex64, device=self.device
        )
        self.sum_q = torch.zeros(self.H, self.D, device=self.device)
        self.t = 0                                   # tokens processed

    @torch.no_grad()
    def step(self, v_t: torch.Tensor, q_t: torch.Tensor) -> None:
        """
        Append one token (v_t,q_t shapes: (H,D))
        """
        phase = self.base_twiddle ** self.t              # (freq,)
        self.prefix_fft += phase[None, :, None] * v_t[:, None, :]
        self.sum_q += q_t
        self.t += 1

    # Accessors --------------------------------------------------------
    def get_fft(self) -> torch.Tensor:
        return self.prefix_fft

    def get_bar_q(self) -> torch.Tensor:
        return self.sum_q / max(self.t, 1)


# ──────────────────────────────────────────────────────────────────────
# SPECTRE mixing layer
# ──────────────────────────────────────────────────────────────────────
class SpectreLayer(nn.Module):
    """
    Drop-in replacement for multi-head attention.
    • Full-sequence mode  (training / teacher forcing)
    • Autoregressive mode (provide a SpectreCache)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        low_rank_r: int = 0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.E      = embed_dim
        self.H      = num_heads
        self.D      = embed_dim // num_heads
        self.Nmax   = max_seq_len
        self.n_fft  = max_seq_len // 2 + 1
        self.rank_r = low_rank_r

        # Linear projections
        self.q_proj  = nn.Linear(self.E, self.E, bias=False)
        self.v_proj  = nn.Linear(self.E, self.E, bias=False)
        self.out_proj = nn.Linear(self.E, self.E, bias=False)

        hidden = max(32, self.D)
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(self.D),
            nn.Linear(self.D, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.n_fft),
        )
        if self.rank_r > 0:
            self.U_mlp = nn.Sequential(
                nn.LayerNorm(self.D),
                nn.Linear(self.D, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2 * self.n_fft * self.rank_r),
            )
            self.V_mlp = nn.Sequential(
                nn.LayerNorm(self.D),
                nn.Linear(self.D, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2 * self.n_fft * self.rank_r),
            )

        k = torch.arange(self.n_fft).float()
        self.register_buffer("freq_idx", k, persistent=False)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                        # (B,N,E)  or  (1,1,E) in AR
        *,
        positions: Optional[torch.Tensor] = None,
        spectre_cache: Optional[SpectreCache] = None,
    ) -> torch.Tensor:
        if spectre_cache is None:
            return self._full_sequence(x, positions)
        else:
            return self._autoregressive(x, spectre_cache)

    # ..................................................................
    #  Full-sequence path (memory-efficient)
    # ..................................................................
    def _full_sequence(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Memory footprint:
          • v_fft          : (B,H,freq,D)  complex
          • seq_time_dom   : (B,H,Nmax,D)  real   ← dominant tensor
        No (freq × N) tensor is ever materialised.
        """
        B, N, _ = x.shape
        H, D = self.H, self.D

        # 1) Q / V projections
        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)  # (B,H,N,D)
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)

        # 2) FFT of values
        v_fft = torch.fft.rfft(v, dim=2)                     # (B,H,freq,D)

        # 3) Build complex gate  g_k  (B,H,freq)
        bar_q = q.mean(dim=2)                                # (B,H,D)
        gate  = self._build_gate(bar_q)                      # (B,H,freq)
        spec  = v_fft * gate[..., None]                      # apply gating

        # 4) Inverse FFT → time domain   (B,H,Nmax,D)
        seq_full = torch.fft.irfft(spec, n=self.Nmax, dim=2)

        # 5) Gather the slice corresponding to each token position:
        if positions is None:
            positions = torch.arange(N, device=x.device)

        # seq_full dims: (B,H,T,D)   we gather along T
        gather_idx = (
            positions.view(1, 1, -1, 1)
            .expand(B, H, -1, D)
        )
        v_selected = torch.gather(seq_full, dim=2, index=gather_idx)  # (B,H,N,D)

        # 6) Re-assemble heads
        v_out = (
            v_selected.permute(0, 2, 1, 3)      # (B,N,H,D)
            .reshape(B, N, self.E)
        )
        return self.out_proj(v_out)

    # ..................................................................
    #  Autoregressive path (cache)
    # ..................................................................
    def _autoregressive(
        self,
        x: torch.Tensor,               # (1,1,E)
        cache: SpectreCache,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        assert B == 1 and N == 1, "AR mode expects (1,1,E)"
        H, D = self.H, self.D

        q = self.q_proj(x).view(1, H, D)
        v = self.v_proj(x).view(1, H, D)
        cache.step(v[0], q[0])                      # update cache

        bar_q = cache.get_bar_q().unsqueeze(0)      # (1,H,D)
        gate  = self._build_gate(bar_q)             # (1,H,freq)

        spec  = cache.get_fft().unsqueeze(0) * gate[..., None]  # (1,H,freq,D)

        # Phase for current position  t = cache.t-1
        t = cache.t - 1
        phase = torch.exp(
            2j * torch.pi * self.freq_idx * float(t) / self.Nmax
        )                                            # (freq,)
        spec = spec * phase[None, None, :, None]

        seq = torch.fft.irfft(spec, n=self.Nmax, dim=2)        # (1,H,T,D)
        v_last = seq[..., t, :]                                # (1,H,D)
        return self.out_proj(v_last.view(1, 1, self.E))

    # ------------------------------------------------------------------
    def _build_gate(self, bar_q: torch.Tensor) -> torch.Tensor:
        """
        bar_q : (B,H,D) → complex gate  (B,H,freq)
        """
        B = bar_q.size(0)
        gate = self.gate_mlp(bar_q).view(B, self.H, self.n_fft, 2)
        gate = view_as_complex(gate)                           # (B,H,freq)

        if self.rank_r > 0:
            U = self.U_mlp(bar_q).view(B, self.H, self.n_fft, self.rank_r, 2)
            V = self.V_mlp(bar_q).view(B, self.H, self.n_fft, self.rank_r, 2)
            U = view_as_complex(U)
            V = view_as_complex(V)
            gate = gate + torch.einsum("...fr,...gr->...fg", U, V)

        return gate
