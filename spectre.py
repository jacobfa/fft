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

def _is_power_of_two(n: int) -> bool:
    """Returns True if *n* is a power-of-two (and > 0)."""
    return (n & (n - 1) == 0) and n > 0


def _safe_rfft(x: torch.Tensor, n: int, dim: int) -> torch.Tensor:
    """cuFFT limitation workaround.

    * If *x* is fp16 **and** *n* is not a power‑of‑two, promote to fp32 for the
      FFT.  We **keep** the complex output in *complex32* (the default result
      of an fp32/16 real‑to‑complex transform) because PyTorch does not support
      complex16 and casting down would silently drop the imaginary part.
    * Otherwise, delegate to the regular :func:`torch.fft.rfft`.
    """
    if x.dtype == torch.float16 and not _is_power_of_two(n):
        # nb: result is complex32 – do *not* cast back to float16!
        return torch.fft.rfft(x.float(), n=n, dim=dim)
    return torch.fft.rfft(x, n=n, dim=dim)



def _safe_irfft(x: torch.Tensor, n: int, dim: int) -> torch.Tensor:
    """Inverse of :func:`_safe_rfft`."""
    if x.dtype == torch.float16 and not _is_power_of_two(n):
        return torch.fft.irfft(x.float(), n=n, dim=dim).to(x.dtype)
    return torch.fft.irfft(x, n=n, dim=dim)

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

class SPECTRELayer(nn.Module):
    """Frequency‑domain mixer that can replace a multi‑head attention block.

    New compared to the original implementation
    -------------------------------------------
    * Toeplitz depth‑wise spectral convolution (``toeplitz_bandwidth``)
    * Complex **modReLU** activation
    * **_safe_rfft / _safe_irfft** to avoid the *“cuFFT only supports power‑of‑two”*
      runtime error when training with AMP/fp16.
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
        *,
        toeplitz_bandwidth: int = 0,
        use_modrelu: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim or d_model // n_heads
        self.max_seq_len = max_seq_len
        self.rank = low_rank
        self.share_gates = share_gates
        self.bandwidth = toeplitz_bandwidth
        self.use_modrelu = use_modrelu
        self.eps = 1e-6

        # ------------------------------------------------------------------
        # Projections
        # ------------------------------------------------------------------
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # ------------------------------------------------------------------
        # Gate MLP –> 2·N_freq values (Re, Im)
        # ------------------------------------------------------------------
        N_freq = max_seq_len // 2 + 1
        gate_dim = 2 * N_freq
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(self.head_dim),
            nn.Linear(self.head_dim, 4 * self.head_dim),
            nn.GELU(),
            nn.Linear(4 * self.head_dim, gate_dim),
        )

        # ------------------------------------------------------------------
        # Toeplitz convolution parameters (depth‑wise, spectral domain)
        # ------------------------------------------------------------------
        if self.bandwidth > 0:
            k = 2 * self.bandwidth + 1
            shape = (1, k) if share_gates else (n_heads, k)
            self.t_real = nn.Parameter(torch.zeros(shape))
            self.t_imag = nn.Parameter(torch.zeros(shape))
        else:
            self.register_parameter("t_real", None)
            self.register_parameter("t_imag", None)

        # ------------------------------------------------------------------
        # modReLU bias (real‑valued)
        # ------------------------------------------------------------------
        if self.use_modrelu:
            shape = (1, N_freq) if share_gates else (n_heads, N_freq)
            self.modrelu_bias = nn.Parameter(torch.zeros(shape))
        else:
            self.register_parameter("modrelu_bias", None)

        # ------------------------------------------------------------------
        # Optional low‑rank outer‑product parameters U, V
        # ------------------------------------------------------------------
        if low_rank and low_rank > 0:
            uv_dim = 2 * N_freq * low_rank
            self.uv_mlp = nn.Sequential(
                nn.LayerNorm(self.head_dim),
                nn.Linear(self.head_dim, 4 * self.head_dim),
                nn.GELU(),
                nn.Linear(4 * self.head_dim, uv_dim),
            )
        else:
            self.uv_mlp = None

        # ------------------------------------------------------------------
        # Optional wavelet refinement
        # ------------------------------------------------------------------
        self.use_wavelet = use_wavelet
        if use_wavelet:
            self.wrm = WaveletRefinement(self.head_dim)

    # == head helpers ==================================================

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:  # (B,L,D) → (B,H,L,Dh)
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:  # (B,H,L,Dh) → (B,L,D)
        B, H, L, Dh = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, L, H * Dh)

    # == gating utilities =============================================

    def _apply_toeplitz(self, g: torch.Tensor) -> torch.Tensor:
        if self.bandwidth == 0:
            return g
        t = torch.complex(self.t_real, self.t_imag)  # (1|H, K)
        offsets = range(-self.bandwidth, self.bandwidth + 1)
        out = g
        for idx, o in enumerate(offsets):
            coeff = t[..., idx].unsqueeze(-1)  # (1|H,1)
            out = out + coeff * torch.roll(g, shifts=o, dims=-1)
        return out

    def _apply_modrelu(self, g: torch.Tensor) -> torch.Tensor:
        if not self.use_modrelu:
            return g
        amp = torch.abs(g)
        bias = self.modrelu_bias.unsqueeze(0)
        act = F.relu(amp + bias)
        return g * act / (amp + self.eps)

    # == complex gate ==================================================

    def _freq_gate(
        self, mean_q: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, H, _ = mean_q.shape
        N_freq = self.max_seq_len // 2 + 1

        g_raw = self.gate_mlp(mean_q).view(B, H, 2, N_freq)
        g_real, g_imag = g_raw[..., 0, :], g_raw[..., 1, :]
        if self.share_gates:
            g_real = g_real.mean(1, keepdim=True)
            g_imag = g_imag.mean(1, keepdim=True)
        g = torch.complex(g_real, g_imag)  # (B,H_or_1,N_freq)
        g = self._apply_toeplitz(g)
        g = self._apply_modrelu(g)

        if not self.rank:
            return g, None, None
        uv_raw = self.uv_mlp(mean_q).view(B, H, N_freq, self.rank * 2)
        splits = torch.split(uv_raw, self.rank, dim=-1)
        U = torch.complex(splits[0], splits[1])
        V = torch.complex(splits[2], splits[3])
        return g, U, V

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional["PrefixFFTCache"] = None,
        incremental_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional["PrefixFFTCache"]]:

        B, L, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        v = self._split_heads(self.v_proj(x))

        if incremental_state and cache is None:
            raise ValueError("Incremental decoding requires a PrefixFFTCache.")

        # ---- incremental --------------------------------------------------
        if incremental_state:
            cache.update(v_t=v[:, :, -1, :], q_t=q[:, :, -1, :])
            mean_q = cache.mean_q
            g, U, V = self._freq_gate(mean_q)
            t_idx = cache.t - 1
            k = torch.arange(self.max_seq_len // 2 + 1, device=x.device, dtype=torch.float)
            phase = torch.exp(1j * 2 * math.pi * k * t_idx / self.max_seq_len)
            g = g * phase.view(1, 1, -1)
            fft_coeff = cache.prefix_fft
            if U is not None:
                g = g + torch.einsum("bhkr,bhkr->bhk", U, V.conj())
            fft_coeff = fft_coeff * g.unsqueeze(-1)
            v_tilde = _safe_irfft(fft_coeff, n=self.max_seq_len, dim=-2)[:, :, : cache.t, :]
            out = self._merge_heads(v_tilde)
            if self.use_wavelet:
                out = self._split_heads(out)
                out = self.wrm(out, mean_q)
                out = self._merge_heads(out)
            return self.out_proj(out)[:, -1:, :], cache

        # ---- full sequence ----------------------------------------------
        mean_q = q.mean(2)
        g, U, V = self._freq_gate(mean_q)
        v_freq = _safe_rfft(v, n=L, dim=2)
        if U is not None:
            v_freq = v_freq * (g + torch.einsum("bhkr,bhkr->bhk", U, V.conj())).unsqueeze(-1)
        else:
            v_freq = v_freq * g.unsqueeze(-1)
        v_tilde = _safe_irfft(v_freq, n=L, dim=2)
        if self.use_wavelet:
            v_tilde = self.wrm(v_tilde, mean_q)
        return self.out_proj(self._merge_heads(v_tilde)), cache


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
