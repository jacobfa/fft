from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants - grouped at top
try:
    import torch_dct as dct
    _HAVE_DCT = True
except ImportError:
    _HAVE_DCT = False

# Note: We implement custom DWT functions, no longer need pywt

# ------------------------------------------------------------
# Helper layers
# ------------------------------------------------------------
# Check PyTorch version for cubic interpolation workaround
_PYTORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
_USE_GRID_SAMPLE_CUBIC = _PYTORCH_VERSION >= (2, 2)

# Complex 1-D interpolation helper
def interp_complex_1d(
    x: torch.Tensor,                # (B, G, K) complex - anchors
    size: int,                      # target length  F_half = n_fft//2+1
    mode: str = "linear"
) -> torch.Tensor:                  # -> (B, G, size) complex
    """
    Interpolate complex data along dim=-1.
    For cubic mode, uses grid_sample which supports bicubic for proper 1D cubic interpolation.
    """
    B, G, K = x.shape
    device = x.device
    
    if mode == "cubic":
        if _USE_GRID_SAMPLE_CUBIC:
            # Use grid_sample for true cubic interpolation (PyTorch >= 2.2)
            # Reshape to 4D for grid_sample: (B*G, 2, 1, K) where 2 channels are real/imag
            real_imag = torch.stack([x.real, x.imag], dim=1).reshape(B*G, 2, 1, K)
            
            # Create 1D sampling grid
            # Grid values should be in [-1, 1], mapping to [0, K-1]
            grid_x = torch.linspace(-1, 1, size, device=device)
            grid = grid_x.view(1, 1, size, 1).expand(B*G, 1, size, 1)  # (N, 1, size, 1)
            # Add dummy y coordinate (required for grid_sample)
            grid_2d = torch.cat([grid, torch.zeros_like(grid)], dim=-1)  # (N, 1, size, 2)
            
            # Apply bicubic interpolation (works as cubic for 1D when height=1)
            interp = F.grid_sample(real_imag, grid_2d, mode='bicubic', 
                                  padding_mode='border', align_corners=True)
            
            # Extract real and imaginary parts: (B*G, 2, 1, size)
            real_up = interp[:, 0, 0, :]  # (B*G, size)
            imag_up = interp[:, 1, 0, :]  # (B*G, size)
            
            # Combine back to complex
            up = torch.complex(real_up, imag_up).view(B, G, size)
            return up
        else:
            # For PyTorch < 2.2, use F.interpolate with mode='linear' as fallback
            # Note: This is not true cubic interpolation, but avoids the bicubic->bilinear degradation
            import warnings
            warnings.warn(
                "PyTorch < 2.2 detected. Using linear interpolation instead of cubic for 1D complex interpolation. "
                "Consider upgrading PyTorch for better interpolation quality.",
                RuntimeWarning,
                stacklevel=2
            )
            # Fall through to linear mode
            mode = "linear"
    else:
        # Fallback to F.interpolate for linear/nearest mode
        assert mode in ('linear', 'nearest'), f"Unsupported interpolation mode: {mode}"
        
        real = x.real.reshape(B * G, 1, K)
        imag = x.imag.reshape(B * G, 1, K)
        
        # align_corners is only valid for linear mode in 1D
        if mode == 'linear':
            real_up = F.interpolate(real, size=size, mode=mode, align_corners=True)
            imag_up = F.interpolate(imag, size=size, mode=mode, align_corners=True)
        else:  # nearest
            real_up = F.interpolate(real, size=size, mode=mode)
            imag_up = F.interpolate(imag, size=size, mode=mode)
        
        up = torch.view_as_complex(
            torch.stack([real_up.squeeze(1), imag_up.squeeze(1)], dim=-1)
        )
        return up.view(B, G, size)


class ComplexModReLU(nn.Module):
    """
    Complex modReLU:
        z ↦ ReLU(|z| + b) · z / |z|.
    A real bias b is learned per complex element.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))
        # Initialize bias to -0.1 to make gate initially near-identity
        nn.init.constant_(self.bias, -0.1)
        # Register epsilon as buffer to avoid reallocation
        self.register_buffer("eps", torch.tensor(1e-4))

    def forward(self, z: torch.Tensor) -> torch.Tensor:             # (..., F) complex
        mag = torch.abs(z)
        
        # Smooth approximation for numerical stability at |z| ≈ 0
        # This ensures stable gradients by avoiding division by very small numbers
        # Using sqrt(|z|^2 + eps^2) instead of max(|z|, eps) for smoother gradients
        mag_stable = torch.sqrt(mag.square() + self.eps.square())
        
        # Compute gating scale with stable denominator
        scale = F.relu(mag + self.bias) / mag_stable
        
        # Apply scale to complex input
        return z * scale


# Note: hermitian_complete is not used - relying on irfft's implicit handling
# def hermitian_complete(g_half: torch.Tensor, n_fft: int) -> torch.Tensor:
#     """
#     Reconstruct a full length‑`n_fft` complex spectrum from its
#     non‑redundant half.  Input shape (..., n_fft//2+1) → output (..., n_fft)
#     """
#     return torch.cat(
#         [g_half, torch.conj(g_half[..., 1:-1].flip(-1))],
#         dim=-1
#     )


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
            import warnings
            warnings.warn("DCT pooling unavailable, falling back to mean pooling. "
                         "Consider installing torch_dct or re-tuning hyperparameters.")
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


class HaarDWT(nn.Module):
    """Haar DWT with pre-allocated filter buffers."""
    def __init__(self):
        super().__init__()
        # Register Haar filters as buffers to avoid runtime allocation
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        self.register_buffer('h0_base', torch.tensor([sqrt2_inv, sqrt2_inv]))  # low-pass
        self.register_buffer('h1_base', torch.tensor([-sqrt2_inv, sqrt2_inv])) # high-pass
    
    def forward(self, x, levels=1):
        """Batch-wise Haar DWT (channels last): x shape (B, L) or (B, C, L).
        Uses circular boundary handling to maintain perfect L → L/2 sizing."""
        if x.ndim == 2:
            x = x[:, None, :]              # add channel dim
        B, C, L = x.shape
        low, high = [], []

        # Expand filters for number of channels
        h0 = self.h0_base.repeat(C).view(C, 1, 2)
        h1 = self.h1_base.repeat(C).view(C, 1, 2)

        for _ in range(levels):
            # Pad to ensure perfect L → L/2 (circular boundary)
            x_padded = F.pad(x, (1, 0), mode='circular')  # Left pad by 1
            current_L = x.shape[-1]  # Track current length at this level
            
            # Convolution + stride-2 down-sample
            lo = F.conv1d(x_padded, h0, stride=2, groups=C)
            hi = F.conv1d(x_padded, h1, stride=2, groups=C)
            
            # Handle odd-length sequences
            if lo.shape[-1] * 2 > current_L:
                lo = lo[..., :-1]
                hi = hi[..., :-1]
                
            low.append(lo); high.append(hi)
            x = lo                              # iterate on coarse scale

        return low, high                        # list per level


# Create a global instance to use in place of the function
_haar_dwt = None

def dwt1d_harr(x, levels=1):
    """Wrapper function for backward compatibility."""
    global _haar_dwt
    if _haar_dwt is None:
        _haar_dwt = HaarDWT()
    # Move module to same device as input
    if _haar_dwt.h0_base.device != x.device:
        _haar_dwt = _haar_dwt.to(x.device)
    return _haar_dwt(x, levels)


class HaarIDWT(nn.Module):
    """Haar inverse DWT with pre-allocated filter buffers."""
    def __init__(self):
        super().__init__()
        # Register synthesis filters as buffers
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        self.register_buffer('g0_base', torch.tensor([sqrt2_inv, sqrt2_inv]))   # low-pass synthesis
        self.register_buffer('g1_base', torch.tensor([sqrt2_inv, -sqrt2_inv]))  # high-pass synthesis
    
    def forward(self, low, high):
        """Batch-wise Haar iDWT (reconstruction) from level coefficients."""
        # Start from coarsest level
        x = low[-1]
        
        # Reconstruct through levels (reverse order)
        for i in range(len(high)-1, -1, -1):
            B, C, L = x.shape
            lo_coeff = x  # Use current approximation (running result)
            hi_coeff = high[i]
            
            # Prepare filters for grouped conv
            g0_filter = self.g0_base.repeat(C).view(C, 1, 2)
            g1_filter = self.g1_base.repeat(C).view(C, 1, 2)
            
            # Upsample by 2 (insert zeros)
            lo_up = F.conv_transpose1d(lo_coeff, g0_filter, stride=2, groups=C)
            hi_up = F.conv_transpose1d(hi_coeff, g1_filter, stride=2, groups=C)
            
            # Handle odd-length reconstruction (symmetric to forward trim)
            target_length = L * 2  # Expected length after upsampling
            if lo_up.shape[-1] > target_length:
                lo_up = lo_up[..., :-1]
                hi_up = hi_up[..., :-1]
            
            # Sum low and high frequency components
            x = lo_up + hi_up
        
        return x


# Global instance for inverse transform
_haar_idwt = None

def idwt1d_harr(low, high):
    """Wrapper function for backward compatibility."""
    global _haar_idwt
    if _haar_idwt is None:
        _haar_idwt = HaarIDWT()
    # Move module to same device as input
    device = low[0].device if low else high[0].device
    if _haar_idwt.g0_base.device != device:
        _haar_idwt = _haar_idwt.to(device)
    return _haar_idwt(low, high)


def dwt_decompose(x, levels=None):
    """Multi-level Haar DWT decomposition. Returns all coefficients."""
    if x.ndim == 2:
        x = x.unsqueeze(1)  # Add channel dimension
    
    B, C, L = x.shape
    if levels is None:
        # Compute max decomposition levels
        levels = int(math.log2(L))
    
    coeffs = []
    for level in range(levels):
        # Apply one level of DWT
        low, high = dwt1d_harr(x, levels=1)
        coeffs.append(high[0])  # Store detail coefficients
        x = low[0]  # Continue with approximation
        
        # Stop if we can't decompose further
        if x.shape[-1] <= 1:
            break
    
    # Add final approximation coefficients
    coeffs.append(x)
    return coeffs


def dwt_reconstruct(coeffs):
    """Multi-level Haar DWT reconstruction from coefficients."""
    # Start with approximation (last coefficient)
    x = coeffs[-1]
    
    # Reconstruct through levels in reverse order
    for i in range(len(coeffs)-2, -1, -1):
        low = [x]
        high = [coeffs[i]]
        x = idwt1d_harr(low, high)
    
    return x.squeeze(1) if x.shape[1] == 1 else x


# ------------------------------------------------------------
# Helper for complex convolution
# ------------------------------------------------------------
def complex_conv1d(x: torch.Tensor, kernel: torch.Tensor, padding: int) -> torch.Tensor:
    """
    Perform 1D convolution on complex tensors.
    x: (..., L) complex
    kernel: (K,) complex where K = 2*padding + 1
    Returns: (..., L) complex
    """
    # Split into real/imag parts
    x_r, x_i = x.real, x.imag
    k_r, k_i = kernel.real, kernel.imag
    
    # Get shape for batching
    *batch_shape, L = x.shape
    batch_size = math.prod(batch_shape) if batch_shape else 1
    
    # Reshape for conv1d
    x_r_flat = x_r.reshape(batch_size, 1, L)
    x_i_flat = x_i.reshape(batch_size, 1, L)
    k_r_view = k_r.view(1, 1, -1)
    k_i_view = k_i.view(1, 1, -1)
    
    # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    # Use circular padding to handle complex kernels properly
    
    # Check PyTorch version for circular padding support
    if hasattr(F.conv1d, '__wrapped__'):  # Check if padding_mode is supported
        try:
            # PyTorch >= 2.2 supports padding_mode='circular' in conv1d
            conv_ac = F.conv1d(x_r_flat, k_r_view, padding=padding, padding_mode='circular')
            conv_bd = F.conv1d(x_i_flat, k_i_view, padding=padding, padding_mode='circular')
            conv_ad = F.conv1d(x_r_flat, k_i_view, padding=padding, padding_mode='circular')
            conv_bc = F.conv1d(x_i_flat, k_r_view, padding=padding, padding_mode='circular')
        except (RuntimeError, TypeError):
            # Fallback for older PyTorch versions - pre-compute padded tensors
            import warnings
            warnings.warn(
                "PyTorch < 2.2 detected: Using manual circular padding in complex_conv1d. "
                "This is slower than native padding_mode='circular'. "
                "Consider upgrading to PyTorch >= 2.2 for better performance.",
                RuntimeWarning,
                stacklevel=2
            )
            x_r_pad = F.pad(x_r_flat, (padding, padding), mode='circular')
            x_i_pad = F.pad(x_i_flat, (padding, padding), mode='circular')
            conv_ac = F.conv1d(x_r_pad, k_r_view, padding=0)
            conv_bd = F.conv1d(x_i_pad, k_i_view, padding=0)
            conv_ad = F.conv1d(x_r_pad, k_i_view, padding=0)
            conv_bc = F.conv1d(x_i_pad, k_r_view, padding=0)
    else:
        # Older PyTorch version without padding_mode support
        x_r_pad = F.pad(x_r_flat, (padding, padding), mode='circular')
        x_i_pad = F.pad(x_i_flat, (padding, padding), mode='circular')
        conv_ac = F.conv1d(x_r_pad, k_r_view, padding=0)
        conv_bd = F.conv1d(x_i_pad, k_i_view, padding=0)
        conv_ad = F.conv1d(x_r_pad, k_i_view, padding=0)
        conv_bc = F.conv1d(x_i_pad, k_r_view, padding=0)
    
    # Combine results
    real_part = (conv_ac - conv_bd).reshape(*batch_shape, L)
    imag_part = (conv_ad + conv_bc).reshape(*batch_shape, L)
    
    return torch.complex(real_part, imag_part)

# ============================================================
# SPECTRE **single head**
# ============================================================
class SpectreHead(nn.Module):
    """
    Frequency‑domain token mixer for one attention head.
    """
    def __init__(
        self,
        embed_dim:   int,
        fft_size:    int,
        *,
        num_groups:  int = 4,
        num_buckets: Optional[int] = None,
        d_gate:      int  = 256,
        use_toeplitz: bool = False,
        toeplitz_bw: int  = 4,
        dropout_p:   float = 0.0,
        pooling_type: str = "dct",
    ):
        super().__init__()
        self.d       = embed_dim
        self.n_fft   = fft_size
        self.G       = num_groups
        self.d_g     = embed_dim // num_groups
        assert embed_dim % num_groups == 0, "embed_dim must be divisible by num_groups"

        # --------------- gate size after bucketing ---------------------
        self.F_half  = fft_size // 2 + 1
        self.B       = max(4, num_buckets or int(math.sqrt(self.F_half)))

        # projections ---------------------------------------------------
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # gate MLP  (anchors × groups  × {Re,Im})
        out_dim = self.B * self.G * 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim, d_gate),
            nn.GELU(),
            nn.Linear(d_gate, out_dim),
        )

        self.q_norm  = nn.LayerNorm(embed_dim)
        self.modrelu = ComplexModReLU(self.F_half * self.G)   # after up‑sampling

        # pooling layer ------------------------------------
        if pooling_type == "dct":
            self.pooling = DCTPooling(embed_dim)
        elif pooling_type == "attention":
            self.pooling = AttentionPooling(embed_dim)
        else:
            self.pooling = MeanPool()

        # optional Toeplitz kernel -------------------------
        self.use_toeplitz = use_toeplitz
        self.toeplitz_kernel = None
        if use_toeplitz:
            self.toeplitz_bw = toeplitz_bw
            # Register as None parameter first, will be initialized in _reset_parameters
            self.register_parameter('toeplitz_kernel', None)
            
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        
        # Initialize parameters after all modules are created
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters with proper device handling."""
        if self.use_toeplitz and self.toeplitz_kernel is None:
            # Get device from existing parameters
            device = self.W_q.weight.device
            dtype = torch.cfloat
            # Initialize Toeplitz kernel on the correct device
            self.toeplitz_kernel = nn.Parameter(
                torch.randn(2 * self.toeplitz_bw + 1, device=device, dtype=dtype)
                / math.sqrt(2 * self.toeplitz_bw + 1)
            )

    # ---------------------------------------------------------------------
    # Forward (training / full‑sequence inference)
    # ---------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        pos_phase: Optional[torch.Tensor] = None,
        return_q_pool: bool = False,
        memory_fft: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, d)   tokens of one head
        pos_phase : optional complex phase of shape (..., F_half)
                    e^{j2π k p / N} for absolute position injection
        return_q_pool : if True, returns tuple (mixed_values, q_pool)
        memory_fft : optional spectral memory of shape (F_half, head_dim) complex
        Returns
        -------
        (B, N, d)  mixed values, or tuple ((B, N, d), (B, d)) if return_q_pool
        """
        B, N, d = x.shape
        assert d == self.d

        # 1) projections
        Q = self.W_q(x)         # (B, N, d)
        V = self.W_v(x)

        # 2) half‑spectrum FFT of V
        V_fft = torch.fft.rfft(V, n=self.n_fft, dim=1)    # (B, F_half, d)

        # -----------------------------------------------------------------
        # 3) grouped bucket gate
        # -----------------------------------------------------------------
        q_pool = self.q_norm(self.pooling(Q))          # (B, d)
        Bsz, d_pool = q_pool.shape

        # Predict anchors: (B, G, Bk, {Re,Im}) → complex (B, G, Bk)
        gate_rs = self.gate_mlp(q_pool).view(Bsz, self.G, self.B, 2)
        gate_anchor = torch.view_as_complex(gate_rs)   # (B, G, Bk)

        # Optional Toeplitz on anchors (per group)
        if self.use_toeplitz:
            conv_result = complex_conv1d(gate_anchor, self.toeplitz_kernel, self.toeplitz_bw)
            gate_anchor = gate_anchor + conv_result

        # Interpolate anchors to full F_half
        # Using cubic interpolation as specified in the paper, 
        # implemented via grid_sample for proper 1D cubic interpolation
        gate_half = interp_complex_1d(
            gate_anchor, size=self.F_half, mode="cubic"   # (B, G, F_half)
        )

        # modReLU operates over flattened freq×group axis
        gate_half = self.modrelu(gate_half.reshape(Bsz, -1)).view_as(gate_half)

        # Apply absolute-position phase if provided
        if pos_phase is not None:
            # pos_phase shape: (1, F_half) or (B, F_half)
            gate_half = gate_half * pos_phase.unsqueeze(1 if pos_phase.dim() == 2 else 0)

        # -----------------------------------------------------------------
        # broadcast gate to channels & mix
        # -----------------------------------------------------------------
        # gate_half : (B, G, F_half) → (B, F_half, d) via repeat
        gate_broadcast = gate_half.permute(0, 2, 1)          # (B, F_half, G)
        gate_broadcast = gate_broadcast.repeat_interleave(self.d_g, dim=-1)

        mixed_half = gate_broadcast * V_fft                  # (B, F_half, d)
        
        # Add persistent spectral memory if present
        if memory_fft is not None:
            mixed_half = mixed_half + memory_fft.unsqueeze(0)  # (F_half, d) -> (1, F_half, d)
        
        v_time = torch.fft.irfft(mixed_half, n=self.n_fft, dim=1)

        result = self.dropout(v_time[:, :N])              # (B, N, d)
        
        if return_q_pool:
            return result, q_pool
        return result

    # ---------------------------------------------------------------------
    # Single‑token decode step
    # ---------------------------------------------------------------------
    @torch.jit.ignore
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

        # ----- build gate from running descriptor (anchors) ------------
        descr = self.q_norm((sum_q / cache.N).unsqueeze(0)).squeeze(0)  # (d,)
        gate_rs = self.gate_mlp(descr).view(self.G, self.B, 2)
        gate_anchor = torch.view_as_complex(gate_rs)                    # (G, B)

        if self.use_toeplitz:
            conv_result = complex_conv1d(gate_anchor, self.toeplitz_kernel, self.toeplitz_bw)
            gate_anchor = gate_anchor + conv_result

        gate_half = interp_complex_1d(
            gate_anchor.unsqueeze(0),       # (1, G, B)
            size=self.F_half, mode="cubic"  # Using cubic interpolation as in forward()
        ).squeeze(0)                        # (G, F_half)

        gate_half = self.modrelu(gate_half.flatten()).view_as(gate_half)

        # positional phase: (F_half,) → broadcast to (G, F_half)
        k = torch.arange(self.F_half, device=gate_half.device)
        # Correct phase formula: exp(j*2π*k*(t-j)/N) where j = t mod N
        j = cache.t % cache.N
        phase = torch.exp(1j * 2 * math.pi * k * (cache.t - j) / cache.N)
        gate_half = gate_half * phase.unsqueeze(0)          # (G, F_half)

        # broadcast to channels
        gate_broadcast = gate_half.permute(1, 0)            # (F_half, G)
        gate_broadcast = gate_broadcast.repeat_interleave(self.d_g, dim=1)

        # ----- mix & pruned irfft ---------------------------------------
        mixed_half = gate_broadcast * prefix_fft  # (F_half, d)
        
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
        num_groups:   int = 4,
        num_buckets:  Optional[int] = None,
        wavelet_on_rate: float = 0.1,
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
                num_groups   = num_groups,
                num_buckets  = num_buckets,
            )
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Add wavelet refinement
        self.wavelet_refinement = WaveletRefinement(embed_dim, on_rate=wavelet_on_rate)

    # training / full inference
    def forward(self, x: torch.Tensor, pos_phase: Optional[torch.Tensor] = None, 
                memory_fft: Optional[torch.Tensor] = None):
        chunks = torch.chunk(x, self.num_heads, dim=-1)
        
        # Slice memory per head if provided
        if memory_fft is not None:
            mem_chunks = torch.chunk(memory_fft, self.num_heads, dim=-1)
        else:
            mem_chunks = [None] * self.num_heads
        
        # Get mixed values and q_pool from each head
        mixed_and_pools = [h(c, pos_phase, return_q_pool=True, memory_fft=m) 
                          for h, c, m in zip(self.heads, chunks, mem_chunks)]
        mixed = [m for m, _ in mixed_and_pools]
        q_pools = [q for _, q in mixed_and_pools]
        
        # Concatenate mixed values
        mixed_concat = torch.cat(mixed, dim=-1)
        
        # Concatenate q_pools across heads for wavelet refinement
        q_pool_concat = torch.cat(q_pools, dim=-1)  # List of (B, head_dim) -> (B, embed_dim)
        
        # Apply wavelet refinement
        mixed_refined = self.wavelet_refinement(mixed_concat, q_pool_concat)
        
        return self.out_proj(mixed_refined)

# ============================================================
# Prefix‑FFT cache (one layer, one head)
# ============================================================
class PrefixFFTCache:
    """
    Sliding‑window frequency cache for autoregressive decoding.
    
    For spectral memory injection during generation:
    ```python
    cache = PrefixFFTCache(n_fft, embed_dim, device=x.device)
    cache.prefill(Q_prompt, V_prompt)
    # CRITICAL: Add spectral memory after prefill for decode to see it
    if hasattr(block, "memory_fft"):
        cache.prefix_fft += block.memory_fft  # One-time O(F*d) addition
    ```
    Note: Without this step, decode_step won't have access to the persistent memory.
    """
    def __init__(self, n_fft: int, embed_dim: int, device=None):
        self.N  = n_fft
        self.d  = embed_dim
        
        # If no device specified, require explicit device from caller tensors
        if device is None:
            raise ValueError("PrefixFFTCache requires an explicit device parameter. "
                           "Pass device=tensor.device from your input tensors.")
        
        self.device = device

        self.prefix_fft = torch.zeros(n_fft // 2 + 1, embed_dim,
                                      dtype=torch.cfloat, device=device)
        self.V_buf = torch.zeros(n_fft, embed_dim, device=device)
        self.Q_buf = torch.zeros_like(self.V_buf)
        self.sum_q = torch.zeros(embed_dim, device=device)
        self.t = -1  # last filled position

        # Store frequencies for stable phase computation
        # Use float32 to avoid mixed-precision issues
        self.freq_k = torch.arange(n_fft // 2 + 1, device=device, dtype=torch.float32)
        self.omega = -2 * math.pi / n_fft  # Base frequency

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
            # Compute phase directly for numerical stability
            phase = torch.exp(1j * self.omega * self.freq_k * j)
            self.prefix_fft -= phase.unsqueeze(-1) * v_old

        # Compute phase for new token
        phase_new = torch.exp(1j * self.omega * self.freq_k * self.t)
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
        # Sample on/off mask per batch element
        B, N, d = v.shape
        on_mask = torch.rand(B, 1, 1, device=v.device) < self.on_rate  # (B, 1, 1)
        
        # Early exit if all batches are off
        if not on_mask.any():
            return v
        
        # Compute gate
        gate = self.gate_mlp(q_pool).unsqueeze(1)          # (B, 1, d)
        
        # Process wavelet transform using our custom implementation
        outputs = []
        
        for b in range(B):
            if on_mask[b]:
                # Apply wavelet decomposition on each channel
                v_b = v[b]  # (N, d)
                # Transpose to (d, N) for channel-wise processing
                v_b_t = v_b.t().unsqueeze(0)  # (1, d, N)
                
                # Decompose
                coeffs = dwt_decompose(v_b_t)
                
                # Reconstruct
                v_ref = dwt_reconstruct(coeffs)  # (1, d, N)
                
                # With proper trimming in dwt/idwt, length is now guaranteed to match
                v_ref = v_ref.squeeze(0).t()  # (N, d)
                
                outputs.append(v_ref)
            else:
                # No wavelet processing needed, just use original
                outputs.append(v[b])
                
        v_ref = torch.stack(outputs, dim=0)  # (B, N, d)
        
        # Apply gate AFTER reconstruction to preserve gradient flow to gate_mlp
        # while still detaching the wavelet path (straight-through estimator)
        #
        # DESIGN CHOICE: We detach v_ref but not gate. This means:
        # 1. The wavelet transform (DWT/IDWT) doesn't receive gradients (straight-through)
        # 2. The gate MLP CAN learn to modulate the wavelet refinement
        # 3. If you want the wavelet path to also train, remove .detach() from v_ref
        #
        # Current behavior: gate learns, wavelet transform is fixed (recommended for stability)
        residual = (v_ref.detach() * gate) * on_mask
        
        return v + residual

# ============================================================
# Transformer block with SPECTRE mixing
# ============================================================
class SpectreBlock(nn.Module):
    """
    Drop‑in replacement for an attention block.
    
    Args:
        embed_dim: Model dimension
        num_heads: Number of attention heads
        n_fft: FFT size for frequency-domain processing
        mlp_ratio: MLP expansion ratio
        d_gate: Hidden dimension for gate MLP
        use_toeplitz: Whether to use Toeplitz convolution
        dropout_p: Dropout probability
        pooling_type: Type of pooling for gate descriptor ("dct", "attention", or "mean")
        num_groups: Number of groups for grouped gating
        num_buckets: Number of frequency buckets (default: sqrt(n_fft//2+1))
        wavelet_on_rate: Probability of applying wavelet refinement
        memory_size: Size of spectral memory bank. If 0, no memory. If 1, uses full FFT size.
                    If > 1, uses that many frequency bins (truncated).
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
        num_groups:   int = 4,
        num_buckets:  Optional[int] = None,
        wavelet_on_rate: float = 0.1,
        memory_size: int = 0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mix = SpectreMultiHead(
            embed_dim, num_heads, n_fft,
            d_gate=d_gate, use_toeplitz=use_toeplitz, dropout_p=dropout_p,
            pooling_type=pooling_type,
            num_groups=num_groups,
            num_buckets=num_buckets,
            wavelet_on_rate=wavelet_on_rate,
        )

        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim),
        )
        
        # Frequency-space memory (as described in the paper)
        # This adds no per-token computational cost
        if memory_size > 0:
            # Use memory_size to control the number of frequency bins
            # Default to full FFT size if memory_size is just used as a boolean flag
            mem_freq_bins = min(memory_size, n_fft // 2 + 1) if memory_size > 1 else n_fft // 2 + 1
            
            self.register_parameter(
                "memory_fft",
                nn.Parameter(
                    torch.randn(mem_freq_bins, embed_dim, dtype=torch.cfloat) 
                    / math.sqrt(embed_dim)
                )
            )
            # Freeze for persistent factual memory bank
            self.memory_fft.requires_grad_(False)
            
            # If using truncated memory, we'll need to pad during forward
            self.memory_freq_bins = mem_freq_bins
            self.full_freq_bins = n_fft // 2 + 1
        else:
            self.memory_fft = None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (B, N, d)
        """
        # Prepare memory_fft with padding if truncated
        memory_fft = self.memory_fft
        if memory_fft is not None and self.memory_freq_bins < self.full_freq_bins:
            # Pad truncated memory to full frequency size
            pad_size = self.full_freq_bins - self.memory_freq_bins
            memory_fft = F.pad(memory_fft, (0, 0, 0, pad_size), mode='constant', value=0)
        
        # Apply mixing with spectral memory and MLP
        x = x + self.mix(self.ln1(x), memory_fft=memory_fft)
        x = x + self.mlp(self.ln2(x))
        return x
