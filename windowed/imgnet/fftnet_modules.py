import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # Generate binary tensor mask; shape: (batch_size, 1, 1, ..., 1)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    DropPath module that performs stochastic depth.
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MultiScaleSpectralAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        seq_len,
        num_heads=4,
        dropout=0.1,
        adaptive=True,
        combine_mode='gate',  # 'gate' or 'concat'
    ):
        """
        Spectral attention module that introduces additional nonlinearity and adaptivity
        while maintaining an n log n computational overhead. Also adds a local wavelet
        transform to capture local dependencies.

        Parameters:
          - embed_dim: Total embedding dimension.
          - seq_len: Sequence length (e.g. number of tokens, including class token).
          - num_heads: Number of attention heads.
          - dropout: Dropout rate applied after iFFT.
          - adaptive: If True, uses an MLP to generate both multiplicative and additive
                      adaptive modulations for the FFT.
          - combine_mode: 'gate' to blend local and global features via a learned scalar,
                          or 'concat' to concatenate and project back down.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        self.adaptive = adaptive
        self.combine_mode = combine_mode

        # Frequency bins for rFFT: (seq_len//2 + 1)
        self.freq_bins = seq_len // 2 + 1

        # ---- FFT parameters ----
        # Base multiplicative filter: one per head and frequency bin.
        self.base_filter = nn.Parameter(torch.ones(num_heads, self.freq_bins, 1))
        # Base additive bias.
        self.base_bias = nn.Parameter(torch.full((num_heads, self.freq_bins, 1), -0.1))

        # ---- Adaptive MLP (if enabled) ----
        if adaptive:
            # Produces scale and bias for each (head, freq_bin).
            self.adaptive_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_heads * self.freq_bins * 2)
            )

        # ---- Wavelet gating or concatenation ----
        # A scalar gate for 'gate' mode
        if self.combine_mode == 'gate':
            # We'll learn a single gate that is broadcast across batch, heads, and positions.
            self.gate_param = nn.Parameter(torch.tensor(0.0))
            self.proj_concat = None  # Not used in gating mode
        elif self.combine_mode == 'concat':
            # A projection to bring concatenated (local + global) features back to embed_dim
            self.proj_concat = nn.Linear(2 * embed_dim, embed_dim)
            self.gate_param = None
        else:
            raise ValueError("combine_mode must be either 'gate' or 'concat'")

        self.dropout = nn.Dropout(dropout)
        # Pre-normalization for improved stability before FFT.
        self.pre_norm = nn.LayerNorm(embed_dim)

    def complex_activation(self, z):
        """
        Applies a nonlinear activation to a complex tensor.
        This function takes the magnitude of z, passes it through GELU, and rescales z accordingly,
        preserving the phase.

        Args:
          z: complex tensor of shape (B, num_heads, freq_bins, head_dim)
        Returns:
          Activated complex tensor of the same shape.
        """
        mag = torch.abs(z)
        # Nonlinear transformation on the magnitude; using GELU for smooth nonlinearity.
        mag_act = F.gelu(mag)
        # Compute scaling factor; add a small epsilon to avoid division by zero.
        scale = mag_act / (mag + 1e-6)
        return z * scale

    def wavelet_transform(self, x_heads):
        """
        Applies a single-level Haar wavelet transform (decomposition + reconstruction)
        to capture local dependencies along the sequence dimension.

        Args:
          x_heads: Tensor of shape (B, num_heads, seq_len, head_dim)

        Returns:
          Reconstructed wavelet-based features of the same shape (B, num_heads, seq_len, head_dim).
        """
        B, H, N, D = x_heads.shape

        # For simplicity, if N is odd, truncate by one
        N_even = N if (N % 2) == 0 else (N - 1)
        x_heads = x_heads[:, :, :N_even, :]  # shape -> (B, H, N_even, D)

        # Split even and odd positions along sequence dimension
        x_even = x_heads[:, :, 0::2, :]  # (B, H, N_even/2, D)
        x_odd  = x_heads[:, :, 1::2, :]  # (B, H, N_even/2, D)

        # Haar wavelet decomposition
        # approx = 0.5*(even + odd), detail = 0.5*(even - odd)
        approx = 0.5 * (x_even + x_odd)
        detail = 0.5 * (x_even - x_odd)

        # A nonlinearity can optionally be applied to approx/detail
        approx = F.gelu(approx)
        detail = F.gelu(detail)

        # Haar wavelet reconstruction
        # even' = approx + detail, odd' = approx - detail
        x_even_recon = approx + detail
        x_odd_recon  = approx - detail

        # Interleave even/odd back to original shape
        out = torch.zeros_like(x_heads)
        out[:, :, 0::2, :] = x_even_recon
        out[:, :, 1::2, :] = x_odd_recon

        # If we truncated one position, pad it back with zeros
        if N_even < N:
            pad = torch.zeros((B, H, 1, D), device=out.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=2)

        return out

    def forward(self, x):
        """
        Forward pass of the enhanced spectral attention module.

        Args:
          x: Input tensor of shape (B, seq_len, embed_dim)

        Returns:
          Tensor of shape (B, seq_len, embed_dim) with combined wavelet (local) and
          FFT-based (global) modulation, plus a residual connection.
        """
        B, N, D = x.shape

        # Pre-normalize input for more stable frequency transformations.
        x_norm = self.pre_norm(x)

        # Reshape to separate heads: (B, num_heads, seq_len, head_dim)
        x_heads = x_norm.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # ---- (1) FFT-based global features ----
        F_fft = torch.fft.rfft(x_heads, dim=2, norm='ortho')  # shape (B, num_heads, freq_bins, head_dim)

        # Adaptive scale/bias if enabled
        if self.adaptive:
            # Global context: average over tokens (B, embed_dim)
            context = x_norm.mean(dim=1)  # (B, embed_dim)
            # Produce 2 values per (head, freq_bin) -> (scale, bias)
            adapt_params = self.adaptive_mlp(context)  # (B, num_heads*freq_bins*2)
            adapt_params = adapt_params.view(B, self.num_heads, self.freq_bins, 2)
            adaptive_scale = adapt_params[..., 0:1]  # shape (B, num_heads, freq_bins, 1)
            adaptive_bias  = adapt_params[..., 1:2]  # shape (B, num_heads, freq_bins, 1)
        else:
            adaptive_scale = torch.zeros(B, self.num_heads, self.freq_bins, 1, device=x.device)
            adaptive_bias  = torch.zeros(B, self.num_heads, self.freq_bins, 1, device=x.device)

        # Combine base parameters with adaptive modulations
        effective_filter = self.base_filter * (1 + adaptive_scale)  # (num_heads, freq_bins, 1) broadcast with (B, ...)
        effective_bias   = self.base_bias + adaptive_bias

        # Apply modulations in the frequency domain
        F_fft_mod = F_fft * effective_filter + effective_bias

        # Nonlinear activation in the frequency domain
        F_fft_nl = self.complex_activation(F_fft_mod)

        # Inverse FFT to bring data back to token space
        x_fft = torch.fft.irfft(F_fft_nl, dim=2, n=self.seq_len, norm='ortho')  # (B, num_heads, seq_len, head_dim)

        # ---- (2) Wavelet-based local features ----
        x_wavelet = self.wavelet_transform(x_heads)  # (B, num_heads, seq_len, head_dim)

        # ---- (3) Combine local/global ----
        if self.combine_mode == 'gate':
            # Gate in [0,1] after a sigmoid
            alpha = torch.sigmoid(self.gate_param)
            # Blend wavelet and FFT features
            x_combined = alpha * x_wavelet + (1.0 - alpha) * x_fft
        else:
            # Concatenate along the embedding dimension
            # First, reshape each to (B, seq_len, num_heads*head_dim) = (B, N, D)
            x_wavelet_reshaped = x_wavelet.permute(0, 2, 1, 3).reshape(B, N, D)
            x_fft_reshaped     = x_fft.permute(0, 2, 1, 3).reshape(B, N, D)
            x_cat = torch.cat([x_wavelet_reshaped, x_fft_reshaped], dim=-1)  # (B, N, 2*D)
            # Project back down to D
            x_combined = self.proj_concat(x_cat)
            # Reshape back to (B, num_heads, seq_len, head_dim) if we want to keep the same path
            x_combined = x_combined.view(B, N, -1).view(B, N, self.num_heads, self.head_dim)
            # Permute back to (B, num_heads, seq_len, head_dim)
            x_combined = x_combined.permute(0, 2, 1, 3)

        # ---- (4) Reshape + dropout + residual ----
        # Merge heads back into the embedding dimension
        x_out = x_combined.permute(0, 2, 1, 3).reshape(B, N, D)
        return x + self.dropout(x_out)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1, attention_module=None, drop_path=0.0):
        """
        A generic Transformer encoder block with integrated drop path (stochastic depth).
          - embed_dim: embedding dimension.
          - mlp_ratio: expansion factor for the MLP.
          - dropout: dropout rate.
          - attention_module: a module handling self-attention (or spectral attention, etc.).
          - drop_path: drop path rate for stochastic depth.
        """
        super().__init__()
        if attention_module is None:
            raise ValueError("Must provide an attention module!")
        self.attention = attention_module
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(embed_dim)
        # Drop path layer for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # (1) Attention + drop path
        x = x + self.drop_path(self.attention(x))
        # (2) MLP (after layer norm) + drop path
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x
