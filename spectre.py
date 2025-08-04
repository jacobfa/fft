from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# For DCT pooling
try:
    import torch_dct as dct
    _HAVE_DCT = True
except ImportError:
    _HAVE_DCT = False

# ------------------------------------------------------------
# Helper layers
# ------------------------------------------------------------
class ComplexModReLU(nn.Module):
    """
    Complex modReLU:
        z ↦ ReLU(|z| + b) · z / |z|.
    A real bias b is learned per complex element.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, z: torch.Tensor) -> torch.Tensor:             # (..., F)
        mag  = torch.abs(z)
        scale = F.relu(mag + self.bias) / (mag + 1e-8)
        return z * scale


def hermitian_complete(g_half: torch.Tensor, n_fft: int) -> torch.Tensor:
    """
    Reconstruct a full length‑`n_fft` complex spectrum from its
    non‑redundant half.  Input shape (..., n_fft//2+1) → output (..., n_fft)
    """
    return torch.cat(
        [g_half, torch.conj(g_half[..., 1:-1].flip(-1))],
        dim=-1
    )


class DCTPooling(nn.Module):
    """DCT-based pooling for gate descriptor."""
    def __init__(self, embed_dim: int, dct_components: int = 64):
        super().__init__()
        self.dct_components = dct_components
        self.embed_dim = embed_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d)
        if _HAVE_DCT:
            # Apply DCT along sequence dimension
            x_dct = dct.dct(x.transpose(1, 2))  # (B, d, N)
            # Take first K components and average
            x_pool = x_dct[:, :, :self.dct_components].mean(dim=2)  # (B, d)
        else:
            # Fallback to mean pooling if DCT not available
            x_pool = x.mean(dim=1)
        return x_pool


class AttentionPooling(nn.Module):
    """2-layer attention pooling for gate descriptor."""
    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d)
        scores = self.w2(self.activation(self.w1(x)))  # (B, N, 1)
        weights = F.softmax(scores, dim=1)
        pooled = (x * weights).sum(dim=1)  # (B, d)
        return pooled


class MeanPool(nn.Module):
    """Simple mean pooling as nn.Module."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

# ============================================================
# SPECTRE **single head**
# ============================================================
class SpectreHead(nn.Module):
    """
    Frequency‑domain token mixer for one attention head.
    """
    def __init__(
        self,
        embed_dim:  int,
        fft_size:   int,
        d_gate:     int  = 256,
        use_toeplitz: bool = False,
        toeplitz_bw: int  = 4,
        dropout_p:  float = 0.0,
        pooling_type: str = "dct",  # "mean", "dct", or "attention"
    ):
        super().__init__()
        self.d      = embed_dim
        self.n_fft  = fft_size
        self.use_toeplitz = use_toeplitz
        self.pooling_type = pooling_type

        # projections
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # gate MLP → (real, imag) parts for half spectrum
        out_dim = (fft_size // 2 + 1) * 2  # Half spectrum
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim, d_gate),
            nn.GELU(),
            nn.Linear(d_gate, out_dim),
        )
        self.q_norm  = nn.LayerNorm(embed_dim)
        self.modrelu = ComplexModReLU(fft_size // 2 + 1)  # Half spectrum
        
        # Pooling layer
        if pooling_type == "dct":
            self.pooling = DCTPooling(embed_dim)
        elif pooling_type == "attention":
            self.pooling = AttentionPooling(embed_dim)
        else:
            self.pooling = MeanPool()  # fallback to mean

        if use_toeplitz:
            self.toeplitz_bw = toeplitz_bw
            # Toeplitz kernel for full spectrum
            self.toeplitz_kernel = nn.Parameter(
                torch.randn(2 * toeplitz_bw + 1, dtype=torch.cfloat)
                / math.sqrt(2 * toeplitz_bw + 1)
            )

        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    # ---------------------------------------------------------------------
    # Forward (training / full‑sequence inference)
    # ---------------------------------------------------------------------
    @torch.jit.ignore
    def forward(
        self,
        x: torch.Tensor,
        pos_phase: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, d)   tokens of one head
        pos_phase : optional complex phase of shape (..., F_half)
                    e^{j2π k p / N} for absolute position injection
        Returns
        -------
        (B, N, d)  mixed values
        """
        B, N, d = x.shape
        assert d == self.d

        # 1) projections
        Q = self.W_q(x)         # (B, N, d)
        V = self.W_v(x)

        # 2) half‑spectrum FFT of V
        V_fft = torch.fft.rfft(V, n=self.n_fft, dim=1)    # (B, F_half, d)

        # 3) build gate
        q_pool = self.q_norm(self.pooling(Q))             # (B, d)
        gate_rs = self.gate_mlp(q_pool)                    # (B, 2·F_half)
        gate_c_half = torch.view_as_complex(gate_rs.view(B, -1, 2))   # (B, F_half)

        if self.use_toeplitz:
            # Apply Toeplitz to half spectrum
            k = self.toeplitz_kernel.view(1, 1, -1)       # (1,1,2r+1)
            pad = (self.toeplitz_bw,)
            gate_c_half = gate_c_half + F.conv1d(
                gate_c_half.unsqueeze(1), k,
                padding=pad, groups=1
            ).squeeze(1)

        # Apply modReLU to half spectrum
        gate_c_half = self.modrelu(gate_c_half)           # (B, F_half)
        
        # We only need the half spectrum for mixing with V_fft
        gate_half = gate_c_half    # (B, F_half)

        # Positional phase should be provided by caller
        # If not provided, we skip it
        
        if pos_phase is not None:
            gate_half = gate_half * pos_phase            # inject position

        # 4) mix & inverse real FFT
        mixed_half = gate_half.unsqueeze(-1) * V_fft    # broadcast over d
        v_time = torch.fft.irfft(mixed_half, n=self.n_fft, dim=1)

        return self.dropout(v_time[:, :N])               # (B, N, d)

    # ---------------------------------------------------------------------
    # Single‑token decode step
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def decode_step(
        self,
        q_t:    torch.Tensor,          # (d,)
        v_t:    torch.Tensor,          # (d,)
        cache:  "PrefixFFTCache",
    ) -> torch.Tensor:
        """
        Incremental generation update for one head (batch‑size = 1).

        Returns the mixed vector ṽ_t (shape d,) for the current timestep.
        """
        prefix_fft, sum_q = cache.decode_step(q_t, v_t)   # state update

        # ----- build gate from running descriptor ------------------------
        descr = self.q_norm((sum_q / cache.N).unsqueeze(0)).squeeze(0)  # (d,)
        gate_rs = self.gate_mlp(descr)                     # (2·F_half,)
        gate_c_half = torch.view_as_complex(gate_rs.view(-1, 2))       # (F_half,)

        if self.use_toeplitz:
            # conv1d expects (B=1,C=1,L)
            gate_c_half = gate_c_half + F.conv1d(
                gate_c_half.view(1, 1, -1),
                self.toeplitz_kernel.view(1, 1, -1),
                padding=(self.toeplitz_bw,)
            ).view(-1)

        # Apply modReLU to half spectrum
        gate_c_half = self.modrelu(gate_c_half.unsqueeze(0)).squeeze(0)  # (F_half,)
        
        # We only need the half spectrum for mixing
        gate_half = gate_c_half

        # positional phase  e^{j2π k t / N}
        k = torch.arange(cache.prefix_fft.size(0), device=gate_half.device)
        pos_phase = torch.exp(1j * 2 * math.pi * k * cache.t / cache.N)
        gate_half = gate_half * pos_phase

        # ----- mix & pruned irfft ---------------------------------------
        mixed_half = gate_half.unsqueeze(-1) * prefix_fft  # (F_half, d)
        
        # Pruned iFFT: compute only the single output we need
        # This is O(F) instead of O(N log N)
        v_out = pruned_irfft_single(mixed_half, cache.N, cache.t % cache.N)

        return v_out


def pruned_irfft_single(X_half: torch.Tensor, n: int, pos: int) -> torch.Tensor:
    """
    Compute only a single output of the inverse real FFT at position 'pos'.
    Vectorized implementation - O(F) but uses efficient tensor operations.
    
    X_half: (F_half, d) where F_half = n//2 + 1
    Returns: (d,) - the output at position pos
    """
    F_half, d = X_half.shape
    
    # Create frequency indices
    k = torch.arange(F_half, device=X_half.device, dtype=X_half.real.dtype)
    
    # Compute phase for all frequencies at once
    phase = 2 * math.pi * k * pos / n
    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)
    
    # Vectorized computation of contributions
    # Shape: (F_half,) -> (F_half, 1) for broadcasting with (F_half, d)
    cos_phase = cos_phase.unsqueeze(1)
    sin_phase = sin_phase.unsqueeze(1)
    
    # Compute real part of X[k] * exp(j*phase) for all k
    contrib = X_half.real * cos_phase - X_half.imag * sin_phase
    
    # Sum contributions with proper scaling
    # DC component (k=0) is not doubled
    # Nyquist (if exists) is not doubled
    result = contrib[0]  # DC component
    
    # Add doubled contributions for k=1 to k=F_half-2 (or F_half-1 if n is odd)
    if n % 2 == 0:
        # Even n: double all except DC and Nyquist
        result += 2 * contrib[1:-1].sum(dim=0)
        # Add Nyquist with alternating sign
        result += contrib[-1] * ((-1) ** pos)
    else:
        # Odd n: double all except DC
        result += 2 * contrib[1:].sum(dim=0)
    
    return result / n  # Normalize

# ============================================================
# Multi‑head wrapper
# ============================================================
class SpectreMultiHead(nn.Module):
    """
    Groups several SpectreHead instances and concatenates outputs.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        n_fft:     int,
        d_gate:    int  = 256,
        use_toeplitz: bool = False,
        dropout_p: float = 0.0,
        pooling_type: str = "dct",
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        self.heads = nn.ModuleList([
            SpectreHead(
                self.head_dim,
                fft_size     = n_fft,
                d_gate       = d_gate,
                use_toeplitz = use_toeplitz,
                dropout_p    = dropout_p,
                pooling_type = pooling_type,
            )
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    # training / full inference
    def forward(self, x: torch.Tensor, pos_phase: Optional[torch.Tensor] = None):
        chunks = torch.chunk(x, self.num_heads, dim=-1)
        mixed  = [h(c, pos_phase) for h, c in zip(self.heads, chunks)]
        return self.out_proj(torch.cat(mixed, dim=-1))

# ============================================================
# Prefix‑FFT cache (one layer, one head)
# ============================================================
class PrefixFFTCache:
    """
    Sliding‑window frequency cache for autoregressive decoding.
    """
    def __init__(self, n_fft: int, embed_dim: int, device=None):
        self.N  = n_fft
        self.d  = embed_dim
        self.device = device

        self.prefix_fft = torch.zeros(n_fft // 2 + 1, embed_dim,
                                      dtype=torch.cfloat, device=device)
        self.V_buf = torch.zeros(n_fft, embed_dim, device=device)
        self.Q_buf = torch.zeros_like(self.V_buf)
        self.sum_q = torch.zeros(embed_dim, device=device)
        self.t = -1  # last filled position

        k = torch.arange(n_fft // 2 + 1, device=device)
        self.twiddle = torch.exp(-2j * math.pi * k / n_fft)

    # -------------------------------------------------------
    def prefill(self, Q: torch.Tensor, V: torch.Tensor):
        """
        Initialises the cache from a prompt.
        Q, V : (L, d)   where L ≤ N
        """
        L = V.size(0)
        pad_len = self.N - L
        V_pad = F.pad(V, (0, 0, 0, pad_len))              # (N, d)
        V_fft = torch.fft.rfft(V_pad, dim=0)               # (F_half, d)

        self.prefix_fft.copy_(V_fft)
        self.V_buf[:L].copy_(V)
        self.Q_buf[:L].copy_(Q)
        self.sum_q = Q.sum(dim=0)
        self.t = L - 1

    # -------------------------------------------------------
    def decode_step(
        self,
        q_t: torch.Tensor,          # (d,)
        v_t: torch.Tensor,          # (d,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with a single new token (q_t, v_t).
        Returns (updated prefix_fft, updated sum_q).
        """
        self.t += 1
        j = self.t % self.N

        v_old = self.V_buf[j]
        if self.t >= self.N:                                    # eviction
            phase = self.twiddle ** j                           # e^{-j2πk·j/N}
            self.prefix_fft -= phase.unsqueeze(-1) * v_old

        phase_new = self.twiddle ** self.t                      # add new
        self.prefix_fft += phase_new.unsqueeze(-1) * v_t

        # ring buffers
        self.V_buf[j] = v_t
        q_old = self.Q_buf[j]
        self.Q_buf[j] = q_t

        self.sum_q += q_t - (q_old if self.t >= self.N else 0.0)
        return self.prefix_fft, self.sum_q

# ============================================================
# Optional Wavelet Refinement Module
# ============================================================
try:
    import pywt
    _HAVE_PYWT = True
except ImportError:
    _HAVE_PYWT = False


class WaveletRefinement(nn.Module):
    """
    Lightweight (optional) wavelet‑domain refinement.
    Per‑channel gating; executed with probability `on_rate`.
    """
    def __init__(self, embed_dim: int, on_rate: float = 0.1):
        super().__init__()
        self.on_rate = on_rate
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, v: torch.Tensor, q_pool: torch.Tensor):
        """
        v      : (B, N, d)     mixed tokens
        q_pool : (B, d)        pooled query descriptor
        """
        if (not _HAVE_PYWT) or (torch.rand(1) > self.on_rate):
            return v

        B, N, d = v.shape
        gate = self.gate_mlp(q_pool).unsqueeze(1)          # (B,1,d)

        # pywt expects numpy; apply batch‑wise
        v_np = v.detach().cpu().numpy()
        outputs = []
        for b in range(B):
            coeffs = pywt.wavedec(v_np[b], wavelet="haar", axis=0)
            coeffs = [torch.tensor(c, device=v.device) * gate[b] for c in coeffs]
            v_ref = pywt.waverec([c.cpu().numpy() for c in coeffs],
                                 wavelet="haar", axis=0)
            outputs.append(torch.tensor(v_ref, device=v.device))
        v_ref = torch.stack(outputs, dim=0)                # (B, N, d)

        return v + v_ref

# ============================================================
# Transformer block with SPECTRE mixing
# ============================================================
class SpectreBlock(nn.Module):
    """
    Drop‑in replacement for an attention block.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        n_fft:     int,
        mlp_ratio: int   = 4,
        d_gate:    int   = 256,
        use_toeplitz: bool = False,
        dropout_p: float = 0.0,
        pooling_type: str = "dct",
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mix = SpectreMultiHead(
            embed_dim, num_heads, n_fft,
            d_gate=d_gate, use_toeplitz=use_toeplitz, dropout_p=dropout_p,
            pooling_type=pooling_type,
        )

        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor):
        x = x + self.mix(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
