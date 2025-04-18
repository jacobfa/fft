import math
import torch
import torch.nn as nn
import torch.nn.functional as F


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


class Spectre(nn.Module):
    """
    Spectral Projection and Content‑adaptive Transformer Engine (SPECTRE).

    Drop‑in replacement for multi‑head self‑attention with
    O(n·d·log n) runtime and memory.

    Args
    ----
    d_model : int
        Embedding dimension.
    n_heads : int
        Number of heads.
    seq_len : int
        Maximum (and compile‑time) sequence length n.
    gate_hidden : int, optional
        Hidden size of the gate MLP (defaults to 4·d_head).
    wavelet : bool, optional
        Enable the lightweight Wavelet Refinement Module (default: False).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int,
        gate_hidden: int | None = None,
        wavelet: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.seq_len = seq_len
        self.wavelet = wavelet

        # ———————————————————————————————————— token projections ———————————————————————————————————— #
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # —————————————————————————————— content‑adaptive spectral gate —————————————————————————————— #
        gate_hidden = gate_hidden or 4 * self.d_head
        self.ln_q = nn.LayerNorm(self.d_head)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.d_head, gate_hidden, bias=False),
            nn.GELU(),
            nn.Linear(gate_hidden, 2 * seq_len, bias=False),  # (real, imag)
        )

        # ———————————————————————————— optional Wavelet Refinement Module ———————————————————————————— #
        if wavelet:
            self.wrm_mlp = nn.Sequential(
                nn.Linear(self.d_head, gate_hidden, bias=False),
                nn.GELU(),
                nn.Linear(gate_hidden, seq_len, bias=False),  # real gates
            )

    # ———————————————————————————————— basic Haar DWT/iDWT ———————————————————————————————— #
    @staticmethod
    def _dwt_haar(x: torch.Tensor) -> torch.Tensor:
        """
        One‑level orthonormal Haar DWT along the sequence axis.

        x : (b, h, n, d_h) real
        returns : same shape (coefficients concatenated low||high)
        """
        even, odd = x[..., 0::2, :], x[..., 1::2, :]
        low = (even + odd) * _INV_SQRT2
        high = (even - odd) * _INV_SQRT2
        return torch.cat([low, high], dim=-2)

    @staticmethod
    def _idwt_haar(y: torch.Tensor) -> torch.Tensor:
        """
        Inverse of the one‑level Haar DWT used above.

        y : (b, h, n, d_h) real, where n is even
        """
        n = y.shape[-2]
        half = n // 2
        low, high = y[..., :half, :], y[..., half:, :]
        even = (low + high) * _INV_SQRT2
        odd = (low - high) * _INV_SQRT2
        out = torch.empty_like(y)
        out[..., 0::2, :] = even
        out[..., 1::2, :] = odd
        return out

    # —————————————————————————— positional phase‑modulation stub —————————————————————————— #
    @staticmethod
    def _inject_position(g: torch.Tensor) -> torch.Tensor:
        # Full equivariant phase‑modulation is omitted for efficiency.
        return g

    # —————————————————————————————————————— forward —————————————————————————————————————— #
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (b, n, d_model)
        b, n, _ = x.shape
        assert (
            n == self.seq_len
        ), f"SPECTRE compiled for seq_len={self.seq_len}, got {n}"

        # 1) token projections
        q = self.w_q(x)  # (b, n, d)
        v = self.w_v(x)

        # reshape to (b, h, n, d_h)
        q = q.view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, n, self.n_heads, self.d_head).transpose(1, 2)

        # 2) global descriptor & spectral gate
        bar_q = self.ln_q(q.mean(dim=2))  # (b, h, d_h)
        g_raw = self.gate_mlp(bar_q)      # (b, h, 2n)
        g_real, g_imag = g_raw.view(b, self.n_heads, 2, n).unbind(dim=2)
        g = torch.complex(g_real, g_imag)  # (b, h, n) complex
        g = self._inject_position(g)       # optional positional phases

        # 3) FFT → gating → iFFT
        v_hat = torch.fft.fft(v, dim=2)                    # (b, h, n, d_h) complex
        v_hat *= g.unsqueeze(-1)                           # diagonal gating
        v_tilde = torch.fft.ifft(v_hat, dim=2).real        # (b, h, n, d_h)

        # 4) optional Wavelet Refinement Module
        if self.wavelet:
            assert (
                n % 2 == 0
            ), "Wavelet module requires even sequence length."
            w_hat = self._dwt_haar(v_tilde)                # (b, h, n, d_h)
            s = self.wrm_mlp(bar_q).unsqueeze(-1)          # (b, h, n, 1)
            w_hat = w_hat * s
            v_ref = self._idwt_haar(w_hat)                 # (b, h, n, d_h)
            v_tilde = v_tilde + v_ref

        # 5) heads → output projection
        y = (
            v_tilde.transpose(1, 2)
            .contiguous()
            .view(b, n, self.d_model)
        )  # (b, n, d_model)
        return self.w_o(y)
