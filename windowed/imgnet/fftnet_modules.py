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
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (batch_size, 1, 1, ..., 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    DropPath module that performs stochastic depth.
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MultiScaleSpectralAttention(nn.Module):
    """
    Multi-scale spectral attention module with:
      1) Global FFT over the entire sequence.
      2) Local STFT (windowed FFT).
      3) A gating mechanism to fuse global and local branches.
    """
    def __init__(
        self,
        embed_dim,
        seq_len,
        num_heads=4,
        dropout=0.1,
        adaptive=True,
        local_window_size=8,
        use_local_branch=True
    ):
        """
        Args:
          embed_dim: Total embedding dimension.
          seq_len:   Sequence length (#tokens, includes class token).
          num_heads: Number of attention heads.
          dropout:   Dropout rate after iFFT.
          adaptive:  If True, uses an MLP to produce (scale, bias) for frequency modulation.
          local_window_size: Size of local STFT windows (e.g., 8).
          use_local_branch: Whether to enable local STFT branch at all.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        self.adaptive = adaptive
        self.use_local_branch = use_local_branch
        self.local_window_size = local_window_size

        # -------------------------
        # Global (full sequence) branch
        # -------------------------
        self.freq_bins_global = seq_len // 2 + 1
        self.base_filter_global = nn.Parameter(
            torch.ones(num_heads, self.freq_bins_global, 1)
        )
        self.base_bias_global = nn.Parameter(
            torch.full((num_heads, self.freq_bins_global, 1), -0.1)
        )

        # -------------------------
        # Local (STFT) branch
        # -------------------------
        if use_local_branch:
            self.freq_bins_local = local_window_size // 2 + 1
            self.base_filter_local = nn.Parameter(
                torch.ones(num_heads, self.freq_bins_local, 1)
            )
            self.base_bias_local = nn.Parameter(
                torch.full((num_heads, self.freq_bins_local, 1), -0.1)
            )
        else:
            self.freq_bins_local = None

        # Adaptive MLPs for global/local if needed
        if adaptive:
            # For the global branch
            self.adaptive_mlp_global = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_heads * self.freq_bins_global * 2)
            )
            # For the local branch
            if use_local_branch:
                self.adaptive_mlp_local = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, num_heads * self.freq_bins_local * 2)
                )

        self.dropout = nn.Dropout(dropout)
        # Pre-normalization for stability
        self.pre_norm = nn.LayerNorm(embed_dim)

        # Gating parameter to fuse global/local
        self.gate = nn.Parameter(torch.zeros(1))  # scalar

    def complex_activation(self, z):
        """
        Nonlinear activation in frequency domain (magnitude-based GELU).
        """
        mag = torch.abs(z)
        mag_act = F.gelu(mag)
        scale = mag_act / (mag + 1e-6)
        return z * scale

    def _transform_global(self, x_heads, adapt_params_global):
        """
        Perform global rFFT and (optional) adaptive modulation.
        x_heads shape: (B, H, N, head_dim)
        Returns: (B, H, N, head_dim)
        """
        B, H, N, D = x_heads.shape
        F_fft = torch.fft.rfft(x_heads, dim=2, norm='ortho')  # -> (B, H, freq_bins_global, D), complex

        if self.adaptive and adapt_params_global is not None:
            adaptive_scale = adapt_params_global[..., 0:1]  # (B, H, freq_bins_global, 1)
            adaptive_bias  = adapt_params_global[..., 1:2]  # (B, H, freq_bins_global, 1)
        else:
            adaptive_scale = torch.zeros(
                B, H, self.freq_bins_global, 1, device=x_heads.device, dtype=x_heads.dtype
            )
            adaptive_bias = torch.zeros_like(adaptive_scale)

        # base_filter_global: (H, freq_bins_global, 1)
        # We rely on broadcasting to match (B,H,freq_bins_global,D).
        effective_filter = self.base_filter_global * (1 + adaptive_scale)  # still (B,H,freq_bins_global,1) after broadcast
        effective_bias   = self.base_bias_global + adaptive_bias

        # Multiply & add in frequency domain
        F_fft_mod = F_fft * effective_filter + effective_bias
        # Nonlinear activation
        F_fft_nl = self.complex_activation(F_fft_mod)

        # iFFT
        x_global = torch.fft.irfft(F_fft_nl, dim=2, n=N, norm='ortho')
        return x_global

    def _transform_local(self, x_heads, adapt_params_local):
        """
        Perform local STFT in windows of size local_window_size.
        x_heads shape: (B, H, N, D)
        """
        B, H, N, D = x_heads.shape
        win_size = self.local_window_size

        # Pad if not multiple of win_size
        pad_len = (win_size - (N % win_size)) % win_size
        if pad_len > 0:
            pad_tensor = torch.zeros((B, H, pad_len, D), device=x_heads.device, dtype=x_heads.dtype)
            x_heads_padded = torch.cat([x_heads, pad_tensor], dim=2)  # (B,H,N+pad_len,D)
        else:
            x_heads_padded = x_heads

        total_len = x_heads_padded.shape[2]
        num_windows = total_len // win_size  # integer

        # Reshape to (B, H, #windows, win_size, D)
        x_heads_win = x_heads_padded.view(B, H, num_windows, win_size, D)
        # Local FFT over dim=3
        F_fft_local = torch.fft.rfft(x_heads_win, dim=3, norm='ortho')  
        # shape: (B, H, #windows, freq_bins_local, D)

        # Expand adapt_params_local to match #windows
        if self.adaptive and adapt_params_local is not None:
            # (B, H, freq_bins_local, 2) -> unsqueeze #windows dimension
            adaptive_scale = adapt_params_local[..., 0:1]  # (B,H,freq_bins_local,1)
            adaptive_bias  = adapt_params_local[..., 1:2]  # (B,H,freq_bins_local,1)

            # Expand across #windows dimension
            adaptive_scale = adaptive_scale.unsqueeze(2).expand(-1, -1, num_windows, -1, -1)
            adaptive_bias  = adaptive_bias.unsqueeze(2).expand(-1, -1, num_windows, -1, -1)
        else:
            adaptive_scale = torch.zeros(
                B, H, num_windows, self.freq_bins_local, 1,
                device=x_heads.device, dtype=x_heads.dtype
            )
            adaptive_bias = torch.zeros_like(adaptive_scale)

        # base_filter_local: (H, freq_bins_local, 1)
        # To broadcast with (B,H,#windows,freq_bins_local,D), we unsqueeze to (1,H,1,freq_bins_local,1).
        base_filter_local = self.base_filter_local.unsqueeze(0).unsqueeze(2)
        base_bias_local   = self.base_bias_local.unsqueeze(0).unsqueeze(2)
        # Now shape: (1, H, 1, freq_bins_local, 1)

        # Combine base + adaptive
        effective_filter_local = base_filter_local * (1 + adaptive_scale)
        effective_bias_local   = base_bias_local + adaptive_bias

        # Multiply and add
        # F_fft_local: (B,H,#windows,freq_bins_local,D)
        F_fft_local_mod = F_fft_local * effective_filter_local + effective_bias_local
        # Nonlinear activation
        F_fft_local_nl = self.complex_activation(F_fft_local_mod)

        # iFFT
        x_local_win = torch.fft.irfft(
            F_fft_local_nl, dim=3, n=win_size, norm='ortho'
        )  # (B,H,#windows,win_size,D)

        # Reshape back
        x_local_padded = x_local_win.view(B, H, total_len, D)
        # Remove padding
        if pad_len > 0:
            x_local = x_local_padded[:, :, :N, :]
        else:
            x_local = x_local_padded

        return x_local

    def forward(self, x):
        """
        x shape: (B, seq_len, embed_dim)
        Returns: (B, seq_len, embed_dim) with residual.
        """
        B, N, D = x.shape
        # Normalize
        x_norm = self.pre_norm(x)
        # Reshape into heads
        x_heads = x_norm.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Context for adaptive MLPs
        context = x_norm.mean(dim=1)  # (B, embed_dim)

        # Global adapt
        if self.adaptive:
            adapt_params_global = self.adaptive_mlp_global(context)
            adapt_params_global = adapt_params_global.view(
                B, self.num_heads, self.freq_bins_global, 2
            )
        else:
            adapt_params_global = None

        x_global = self._transform_global(x_heads, adapt_params_global)

        # Local adapt (optional)
        if self.use_local_branch:
            if self.adaptive:
                adapt_params_local = self.adaptive_mlp_local(context)
                adapt_params_local = adapt_params_local.view(
                    B, self.num_heads, self.freq_bins_local, 2
                )
            else:
                adapt_params_local = None

            x_local = self._transform_local(x_heads, adapt_params_local)

            # Fuse
            alpha = torch.sigmoid(self.gate)  # scalar in [0,1]
            x_fused_heads = alpha * x_global + (1.0 - alpha) * x_local
        else:
            x_fused_heads = x_global

        # Merge heads
        x_fused = x_fused_heads.permute(0, 2, 1, 3).reshape(B, N, D)

        # Residual + dropout
        out = x + self.dropout(x_fused)
        return out


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with:
      - A spectral attention module (MultiScaleSpectralAttention).
      - An MLP.
      - LayerNorm and optional DropPath.
    """
    def __init__(
        self,
        embed_dim,
        mlp_ratio=4.0,
        dropout=0.1,
        attention_module=None,
        drop_path=0.0
    ):
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
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # Attention
        x = x + self.drop_path(self.attention(x))
        # MLP
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x
