# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"ComplexHalf support.*experimental.*",
    category=UserWarning,
)
# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Element-wise multiplication of two complex tensors that may be stored as
    separate real/imag channels (…, 2) or native complex dtypes.
    """
    if a.is_complex() and b.is_complex():
        return a * b
    if a.is_complex():
        a_real, a_imag = a.real, a.imag
    else:
        a_real, a_imag = a.unbind(-1)
    if b.is_complex():
        b_real, b_imag = b.real, b.imag
    else:
        b_real, b_imag = b.unbind(-1)
    real = a_real * b_real - a_imag * b_imag
    imag = a_real * b_imag + a_imag * b_real
    if a.is_complex() or b.is_complex():
        return torch.complex(real, imag)
    return torch.stack([real, imag], dim=-1)


def _to_complex(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is complex dtype (PyTorch ≥1.10)."""
    return x if x.is_complex() else torch.view_as_complex(x)


def _from_complex(z: torch.Tensor, as_complex: bool) -> torch.Tensor:
    """
    Convert complex tensor back to (…, 2) real/imag channels if the original
    representation was real-stacked.
    """
    if as_complex:
        return z
    return torch.view_as_real(z)


# ---------------------------------------------------------------------
# Prefix–FFT KV-like cache
# ---------------------------------------------------------------------


class PrefixFFTCache:
    """
    Stores running real-FFT coefficients and running mean of the query
    projection for each (batch, head).  The cache is meant to be carried in the
    model-generated `past_key_values` (akin to a KV cache).
    """

    def __init__(self, max_seq_len: int, head_dim: int, device: torch.device):
        n_freq = max_seq_len // 2 + 1
        self.prefix_fft = torch.zeros(
            0, 0, n_freq, head_dim, dtype=torch.complex64, device=device
        )  # to be materialised at first use
        self.mean_q = torch.zeros(0, 0, head_dim, device=device)
        self.t = 0  # current length
        self.max_seq_len = max_seq_len
        self._twiddle_cache: Dict[int, torch.Tensor] = {}

    # -----------------------------------------------------------------

    def _twiddle(self, t: int) -> torch.Tensor:
        """
        Return e^(−j 2π k t / N_max) for all k in [0, N_max/2].
        Shape: (n_freq,)
        """
        if t in self._twiddle_cache:
            return self._twiddle_cache[t]
        k = torch.arange(self.max_seq_len // 2 + 1, device=self.prefix_fft.device)
        phase = -2 * math.pi * k.float() * t / self.max_seq_len
        twiddle = torch.exp(1j * phase)  # complex64
        self._twiddle_cache[t] = twiddle
        return twiddle

    # -----------------------------------------------------------------

    def maybe_expand_batch(self, batch: int, n_heads: int, head_dim: int) -> None:
        if self.prefix_fft.numel() == 0:
            n_freq = self.max_seq_len // 2 + 1
            self.prefix_fft = torch.zeros(
                batch, n_heads, n_freq, head_dim, dtype=torch.complex64, device=self.mean_q.device
            )
            self.mean_q = torch.zeros(batch, n_heads, head_dim, device=self.mean_q.device)
            return
        if self.prefix_fft.shape[0] < batch:
            pad_b = batch - self.prefix_fft.shape[0]
            self.prefix_fft = F.pad(self.prefix_fft, (0, 0, 0, 0, 0, 0, 0, pad_b))
            self.mean_q = F.pad(self.mean_q, (0, 0, 0, 0, 0, pad_b))

    # -----------------------------------------------------------------

    def update(
        self, v_t: torch.Tensor, q_t: torch.Tensor
    ) -> None:  # (B, H, D_head)
        batch, n_heads, head_dim = v_t.shape
        self.maybe_expand_batch(batch, n_heads, head_dim)

        twiddle = self._twiddle(self.t)  # (n_freq,)
        # Broadcast to (B, H, n_freq, D_head)
        self.prefix_fft[:, :, :, :] += (
            v_t.unsqueeze(-2) * twiddle.view(1, 1, -1, 1).to(v_t.dtype)
        )

        # Running mean for q
        if self.t == 0:
            self.mean_q[:, :, :] = q_t
        else:
            self.mean_q = self.mean_q * (self.t / (self.t + 1.0)) + q_t / (self.t + 1.0)

        self.t += 1

    # -----------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "prefix_fft": self.prefix_fft,
            "mean_q": self.mean_q,
            "t": self.t,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.prefix_fft = state["prefix_fft"]
        self.mean_q = state["mean_q"]
        self.t = state["t"]


class HaarDWT(nn.Module):
    """Single-level 1-D Haar analysis / synthesis, depth-wise per channel."""

    def __init__(self):
        super().__init__()

        sqrt2_inv = 1.0 / math.sqrt(2.0)
        lp = torch.tensor([sqrt2_inv, sqrt2_inv])      # low-pass  [+ +]
        hp = torch.tensor([-sqrt2_inv, sqrt2_inv])     # high-pass [− +]

        # store one copy; we’ll replicate along channel dim at runtime
        self.register_buffer("lp_base", lp.view(1, 1, 2))   # (1,1,k)
        self.register_buffer("hp_base", hp.view(1, 1, 2))

    # -----------------------------------------------------------------
    def _repeat(self, kernel: torch.Tensor, C: int, *, dtype, device):
        """Make (C,1,k) kernel matching the input’s dtype / device."""
        return kernel.to(dtype=dtype, device=device).repeat(C, 1, 1)

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (B, C, L)  – returns (low, high) each (B, C, ⌊L/2⌋)
        """
        B, C, _ = x.shape
        lp = self._repeat(self.lp_base, C, dtype=x.dtype, device=x.device)
        hp = self._repeat(self.hp_base, C, dtype=x.dtype, device=x.device)

        low  = F.conv1d(x, lp, stride=2, groups=C)
        high = F.conv1d(x, hp, stride=2, groups=C)
        return low, high

    # -----------------------------------------------------------------
    def inverse(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """
        Inverse single-level Haar reconstruction.
        Inputs are (B, C, L/2); output (B, C, L) (L even).
        """
        B, C, _ = low.shape

        # Upsample by factor 2 (nearest + zero pad), shift for perfect reconstruction
        up_lp  = F.pad(torch.repeat_interleave(low , repeats=2, dim=-1), (1, 1))
        up_hp  = F.pad(torch.repeat_interleave(high, repeats=2, dim=-1), (1, 1))

        lp_k = self._repeat(self.lp_base.flip(-1), C, dtype=low.dtype , device=low.device)
        hp_k = self._repeat(self.hp_base.flip(-1), C, dtype=high.dtype, device=high.device)

        rec_lp = F.conv1d(up_lp, lp_k, groups=C)
        rec_hp = F.conv1d(up_hp, hp_k, groups=C)
        return rec_lp + rec_hp

# ---------------------------------------------------------------------
# Wavelet Refinement (single-level orthogonal Haar)
# ---------------------------------------------------------------------



class WaveletRefinement(nn.Module):
    """
    Lightweight, optionally skipped DWT branch that sharpens local detail.
    """

    def __init__(self, d_model: int, skip_init: float = 0.9):
        super().__init__()
        self.dwt = HaarDWT()

        self.gating = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        log_odds = math.log((1.0 - skip_init) / skip_init)
        self.skip_logit = nn.Parameter(torch.tensor(log_odds))

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor, global_q: torch.Tensor) -> torch.Tensor:
        """
        x        : (B, H, L, Dh)
        global_q : (B, H, Dh)
        """
        if self.training:
            do_skip = torch.rand((), device=x.device) < torch.sigmoid(self.skip_logit)
        else:
            do_skip = False

        if do_skip:
            return x  # fast path – branch skipped

        B, H, L, Dh = x.shape
        x_c = x.reshape(B * H, Dh, L)  # (BH, C, L)

        lp, hp = self.dwt(x_c)
        gates  = self.gating(global_q).view(B * H, Dh, 1)
        lp, hp = lp * gates, hp * gates

        recon = self.dwt.inverse(lp, hp)                    # (BH, C, L)
        recon = recon.view(B, H, Dh, L).permute(0, 1, 3, 2) # (B,H,L,Dh)
        return x + recon

# ---------------------------------------------------------------------
# SPECTRE Layer
# ---------------------------------------------------------------------


class SPECTRELayer(nn.Module):
    """
    Frequency-domain mixer that can replace a multi-head attention block.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: Optional[int] = None,
        max_seq_len: int = 8192,
        low_rank: Optional[int] = None,
        use_wavelet: bool = False,
        share_gates: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim or d_model // n_heads
        self.max_seq_len = max_seq_len
        self.rank = low_rank
        self.share_gates = share_gates

        # --------------------------------------------------------------
        # Linear projections
        # --------------------------------------------------------------
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # --------------------------------------------------------------
        # Gate MLP –-> 2 · N_freq real numbers  (Re, Im) per head
        # --------------------------------------------------------------
        N_freq = max_seq_len // 2 + 1            # unique RFFT bins
        gate_dim = 2 * N_freq                    # <-- fixed
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(self.head_dim),
            nn.Linear(self.head_dim, 4 * self.head_dim),
            nn.GELU(),
            nn.Linear(4 * self.head_dim, gate_dim),
        )

        # --------------------------------------------------------------
        # Optional low-rank outer-product parameters U, V
        # --------------------------------------------------------------
        if low_rank and low_rank > 0:
            uv_dim = 2 * N_freq * low_rank       # <-- fixed
            self.uv_mlp = nn.Sequential(
                nn.LayerNorm(self.head_dim),
                nn.Linear(self.head_dim, 4 * self.head_dim),
                nn.GELU(),
                nn.Linear(4 * self.head_dim, uv_dim),
            )
        else:
            self.uv_mlp = None

        # --------------------------------------------------------------
        # Optional wavelet refinement
        # --------------------------------------------------------------
        self.use_wavelet = use_wavelet
        if use_wavelet:
            self.wrm = WaveletRefinement(self.head_dim)

    # === split / merge heads helpers =================================

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:          # (B,L,D) ➜ (B,H,L,Dh)
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:          # (B,H,L,Dh) ➜ (B,L,D)
        B, H, L, Dh = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * Dh)

    # === complex gate creation =======================================

    def _freq_gate(
        self, mean_q: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        mean_q : (B, H, Dh)
        Returns:
            g     : (B, 1|H, N_freq) complex gate
            U, V  : (B, H, N_freq, r) complex   (if low-rank enabled)
        """
        B, H, _ = mean_q.shape
        N_freq = self.max_seq_len // 2 + 1

        g_raw = self.gate_mlp(mean_q)            # (B,H,2·N_freq)
        g_real, g_imag = g_raw.split(N_freq, dim=-1)
        
        g_raw = self.gate_mlp(mean_q)              # (B, H, 2·N_freq)
        g_raw = g_raw.view(B, H, 2, N_freq)        # split real/imag early
        g_real, g_imag = g_raw[..., 0, :], g_raw[..., 1, :]

        if self.share_gates:                       # average per-head in real domain
            g_real = g_real.mean(dim=1, keepdim=True)
            g_imag = g_imag.mean(dim=1, keepdim=True)

        g = torch.complex(g_real, g_imag)          # (B, 1|H, N_freq) – complex32

        if not self.rank:
            return g, None, None

        # ---- low-rank outer product ---------------------------------
        uv_raw = self.uv_mlp(mean_q)             # (B,H,2·N_freq·r)
        uv_raw = uv_raw.view(B, H, N_freq, self.rank * 2)
        u_real, u_imag, v_real, v_imag = torch.split(
            uv_raw, [self.rank, self.rank, self.rank, self.rank], dim=-1
        )
        U = torch.complex(u_real, u_imag)
        V = torch.complex(v_real, v_imag)
        return g, U, V
    # -----------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[PrefixFFTCache] = None,
        incremental_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[PrefixFFTCache]]:
        """
        x: (B, L, D)
        cache: PrefixFFTCache or None
        incremental_state:
            False (default) → full-sequence training / inference
            True            → step-wise generation (requires `cache`)
        Returns (output, updated_cache)
        """
        B, L, D = x.shape

        # 1) token projections
        q = self._split_heads(self.q_proj(x))  # (B, H, L, Dh)
        v = self._split_heads(self.v_proj(x))  # (B, H, L, Dh)

        if incremental_state and cache is None:
            raise ValueError("Incremental decoding requires a PrefixFFTCache.")

        if incremental_state:
            # Step-wise generation ------------------------------------------------
            cache.update(
                v_t=v[:, :, -1, :],  # newest token (B,H,Dh)
                q_t=q[:, :, -1, :],
            )

            mean_q = cache.mean_q  # (B,H,Dh)
            g, U, V = self._freq_gate(mean_q)  # (B,H_or_1,N_freq)

            # positional phase exp(j 2π k t / N_max)
            t = cache.t - 1  # current index (0-based)
            k = torch.arange(
                self.max_seq_len // 2 + 1, device=x.device, dtype=torch.float
            )
            phase = torch.exp(
                1j * 2 * math.pi * k * t / self.max_seq_len
            )  # (N_freq,)
            g = g * phase.view(1, 1, -1)

            # low-rank outer product
            fft_coeff = cache.prefix_fft  # (B,H,N_freq,Dh)
            if U is not None:
                outer = torch.einsum("bhkr,bhkr->bhk", U, V.conj())
                g = g + outer  # broadcast: (B,H,N_freq)

            # apply gate
            fft_coeff = fft_coeff * g.unsqueeze(-1)

            # inverse transform (full length), then slice visible prefix
            v_tilde = torch.fft.irfft(
                fft_coeff, n=self.max_seq_len, dim=-2
            )  # (B,H,L_max,Dh)
            v_tilde = v_tilde[:, :, : cache.t, :]  # visible prefix
            out = self._merge_heads(v_tilde)

            if self.use_wavelet:
                out = self._split_heads(out)
                out = self.wrm(out, mean_q)
                out = self._merge_heads(out)

            out = self.out_proj(out)
            return out[:, -1:, :], cache  # only the newest token

        # Full-sequence path -----------------------------------------------------
        mean_q = q.mean(dim=2)  # (B,H,Dh)
        g, U, V = self._freq_gate(mean_q)

        v_freq = torch.fft.rfft(v, n=L, dim=2)  # (B,H,N_freq,Dh)

        # low-rank outer product addition
        if U is not None:
            outer = torch.einsum("bhkr,bhkr->bhk", U, V.conj())
            g = g + outer  # (B,H,N_freq) or (B,1,N_freq)

        v_freq = v_freq * g.unsqueeze(-1)  # apply gate
        v_tilde = torch.fft.irfft(v_freq, n=L, dim=2)  # (B,H,L,Dh)

        if self.use_wavelet:
            v_tilde = self.wrm(v_tilde, mean_q)

        out = self.out_proj(self._merge_heads(v_tilde))
        return out, cache

    # -----------------------------------------------------------------

    def init_cache(self, device: torch.device) -> PrefixFFTCache:
        return PrefixFFTCache(self.max_seq_len, self.head_dim, device)


# ---------------------------------------------------------------------
# SPECTRE Block (with residual + LayerNorm)
# ---------------------------------------------------------------------


class SPECTREBlock(nn.Module):
    """
    [LayerNorm → SPECTRE → residual]  + position-wise FFN.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_hidden: int = 4,
        **spectre_kwargs,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.spectre = SPECTRELayer(d_model, n_heads, **spectre_kwargs)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden * d_model),
            nn.GELU(),
            nn.Linear(ffn_hidden * d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[PrefixFFTCache] = None,
        incremental_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[PrefixFFTCache]]:
        y, cache = self.spectre(self.ln1(x), cache=cache, incremental_state=incremental_state)
        x = x + y
        x = x + self.ffn(self.ln2(x))
        return x, cache