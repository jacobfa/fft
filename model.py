from __future__ import annotations

from math import sqrt, pi
import torch
import torch.nn as nn

inv_sqrt2 = 1 / sqrt(2.0)


# ---------------------------------------------------------------------
# Wavelet Refinement Module (handles odd sequence lengths gracefully)
# ---------------------------------------------------------------------

class WaveletRefinement(nn.Module):
    """One‑level orthogonal Haar Wavelet Refinement Module (WRM).

    * Skips execution when sequence length is odd (cannot decimate by 2).
    * Uses channel‑wise gates from a small MLP.
    """

    def __init__(
        self,
        d_h: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float32,
        skip_ratio: float = 0.9,
    ) -> None:
        super().__init__()
        self.skip_ratio = skip_ratio
        hidden = 4 * d_h
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_h, hidden, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden, max_seq_len, dtype=dtype),
            nn.Sigmoid(),
        )

    # Haar helpers ------------------------------------------------------
    @staticmethod
    def _dwt_haar(x: torch.Tensor) -> torch.Tensor:
        low = (x[:, :, 0::2, :] + x[:, :, 1::2, :]) * inv_sqrt2
        high = (x[:, :, 0::2, :] - x[:, :, 1::2, :]) * inv_sqrt2
        return torch.cat([low, high], dim=2)

    @staticmethod
    def _idwt_haar(coeffs: torch.Tensor) -> torch.Tensor:
        N_half = coeffs.shape[2] // 2
        low, high = torch.split(coeffs, N_half, dim=2)
        out = torch.empty_like(coeffs)
        out[:, :, 0::2, :] = (low + high) * inv_sqrt2
        out[:, :, 1::2, :] = (low - high) * inv_sqrt2
        return out

    # Forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor, bar_q: torch.Tensor) -> torch.Tensor:
        B, H, N, d = x.shape

        # Skip if odd sequence length or stochastic skip controller
        if (N % 2 == 1) or (self.skip_ratio > 0 and torch.rand(1).item() < self.skip_ratio):
            return x

        w = self._dwt_haar(x)
        gate = self.gate_mlp(bar_q.mean(dim=1))[:, :N]  # (B,N)
        gate = gate.view(B, 1, N, 1)
        w = w * gate
        v_ref = self._idwt_haar(w)
        return x + v_ref


# ---------------------------------------------------------------------
# SPECTRE Token Mixing Layer
# ---------------------------------------------------------------------

class SpectreMix(nn.Module):
    """Spectral Projection & Content‑adaptive Transformer Engine (SPECTRE)."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        max_seq_len: int = 2048,
        gate_hidden: int | None = None,
        use_phase: bool = True,
        enable_wrm: bool = False,
        wrm_skip_ratio: float = 0.9,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert dim % heads == 0, "`dim` must be divisible by `heads`."

        self.dim = dim
        self.heads = heads
        self.d_h = dim // heads
        self.max_seq_len = max_seq_len
        self.n_freq_max = max_seq_len // 2 + 1
        self.use_phase = use_phase

        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(dim, dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(dim, dim, bias=False, dtype=dtype)

        # Gate MLP
        gate_hidden = gate_hidden or 4 * self.d_h
        self.norm = nn.LayerNorm(self.d_h, dtype=dtype)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.d_h, gate_hidden, dtype=dtype),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2 * self.n_freq_max, dtype=dtype),
        )

        if self.use_phase:
            k = torch.arange(self.n_freq_max).float()
            self.register_buffer("_freq_idx", k, persistent=False)

        self.wrm = (
            WaveletRefinement(self.d_h, max_seq_len, dtype=dtype, skip_ratio=wrm_skip_ratio)
            if enable_wrm
            else None
        )

    # Helper reshapes ---------------------------------------------------
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.heads, self.d_h).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * d)

    # Forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,N,D)
        B, N, _ = x.shape
        assert N <= self.max_seq_len, "Sequence length exceeds `max_seq_len`."
        n_freq = N // 2 + 1

        # 1) Project
        q = self._split_heads(self.q_proj(x))
        v = self._split_heads(self.v_proj(x))

        # 2) FFT (upgrade to fp32 if needed for cuFFT)
        needs_fp32_fft = v.dtype in (torch.float16, torch.bfloat16) and (N & (N - 1) != 0)
        v_fft_in = v.to(torch.float32) if needs_fp32_fft else v
        v_f = torch.fft.rfft(v_fft_in, n=N, dim=2)

        # 3) Spectral gate
        bar_q = self.norm(q.mean(dim=2))
        gate_coeff = self.gate_mlp(bar_q)
        gate_real = gate_coeff[..., :n_freq]
        gate_imag = gate_coeff[..., self.n_freq_max : self.n_freq_max + n_freq]
        gate_dtype = torch.float32 if needs_fp32_fft else v.dtype  # align with FFT dtype
        g = torch.complex(gate_real.to(gate_dtype), gate_imag.to(gate_dtype)).unsqueeze(-1)

        if self.use_phase:
            pos = torch.arange(N, device=x.device)
            phase = torch.exp(
                1j
                * 2
                * pi
                * self._freq_idx[:n_freq].unsqueeze(-1)
                * pos.unsqueeze(0)
                / N
            )
            g = g * phase.mean(dim=-1)[None, None, :, None]

        v_f = v_f * g

        # 4) Inverse FFT
        v_t = torch.fft.irfft(v_f, n=N, dim=2)
        if needs_fp32_fft:
            v_t = v_t.to(v.dtype)

        # 5) Wavelet refinement
        if self.wrm is not None:
            v_t = self.wrm(v_t, bar_q)

        # 6) Output projection
        y = self._merge_heads(v_t)
        return self.out_proj(y)
