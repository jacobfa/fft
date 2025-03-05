import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (batch_size, 1, 1, ..., 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MultiScaleSpectralAttention(nn.Module):
    """
    Multi-scale spectral attention module with:
      1) Global FFT
      2) Local STFT
      3) Learned gating MLP
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
        # Global branch
        # -------------------------
        self.freq_bins_global = seq_len // 2 + 1
        self.base_filter_global = nn.Parameter(
            torch.ones(num_heads, self.freq_bins_global, 1)
        )
        self.base_bias_global = nn.Parameter(
            torch.full((num_heads, self.freq_bins_global, 1), -0.1)
        )

        # -------------------------
        # Local branch
        # -------------------------
        if use_local_branch:
            self.freq_bins_local = local_window_size // 2 + 1
            self.base_filter_local = nn.Parameter(
                torch.ones(num_heads, self.freq_bins_local, 1)
            )
            self.base_bias_local = nn.Parameter(
                torch.full((num_heads, self.freq_bins_local, 1), -0.1)
            )

            hann = torch.hann_window(local_window_size, periodic=False)
            hann = hann.view(1, 1, 1, local_window_size, 1)
            self.register_buffer("hann_window", hann)
        else:
            self.freq_bins_local = None

        # Adaptive MLPs
        if adaptive:
            self.adaptive_mlp_global = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_heads * self.freq_bins_global * 2)
            )
            if use_local_branch:
                self.adaptive_mlp_local = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, num_heads * self.freq_bins_local * 2)
                )

        # Dropout on final output
        self.dropout = nn.Dropout(dropout)
        # A "pre-norm" for stability inside the FFT
        self.pre_norm = nn.LayerNorm(embed_dim)

        # Fusion MLP for combining global+local
        if use_local_branch:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, num_heads)  # => (B, H)
            )
        else:
            self.fusion_mlp = None

    def complex_activation(self, z):
        """Magnitude-based GELU."""
        mag = torch.abs(z)
        mag_act = F.gelu(mag)
        scale = mag_act / (mag + 1e-6)
        return z * scale

    def _transform_global(self, x_heads, adapt_params_global):
        B, H, N, D = x_heads.shape
        F_fft = torch.fft.rfft(x_heads, dim=2, norm='ortho')  

        if self.adaptive and adapt_params_global is not None:
            adaptive_scale = adapt_params_global[..., 0:1]
            adaptive_bias  = adapt_params_global[..., 1:2]
        else:
            adaptive_scale = 0.0
            adaptive_bias  = 0.0

        # base_filter_global: (H, freq_bins_global, 1)
        # adapt_params: (B, H, freq_bins_global, 2)
        effective_filter = self.base_filter_global * (1 + adaptive_scale)
        effective_bias   = self.base_bias_global + adaptive_bias

        F_fft_mod = F_fft * effective_filter + effective_bias
        F_fft_nl  = self.complex_activation(F_fft_mod)
        x_global  = torch.fft.irfft(F_fft_nl, dim=2, n=N, norm='ortho')
        return x_global

    def _transform_local(self, x_heads, adapt_params_local):
        B, H, N, D = x_heads.shape
        win_size = self.local_window_size

        # Pad if needed
        pad_len = (win_size - (N % win_size)) % win_size
        if pad_len > 0:
            pad_tensor = torch.zeros((B, H, pad_len, D), device=x_heads.device, dtype=x_heads.dtype)
            x_heads_padded = torch.cat([x_heads, pad_tensor], dim=2)
        else:
            x_heads_padded = x_heads

        total_len = x_heads_padded.shape[2]
        num_windows = total_len // win_size

        # Reshape to (B,H,#windows,win_size,D)
        x_heads_win = x_heads_padded.view(B, H, num_windows, win_size, D)
        x_heads_win = x_heads_win * self.hann_window  # apply Hann window

        # FFT
        F_fft_local = torch.fft.rfft(x_heads_win, dim=3, norm='ortho')  
        # => (B, H, #windows, freq_bins_local, D)

        if self.adaptive and adapt_params_local is not None:
            # (B, H, freq_bins_local, 2)
            adaptive_scale = adapt_params_local[..., 0:1]
            adaptive_bias  = adapt_params_local[..., 1:2]
            # Expand to #windows
            adaptive_scale = adaptive_scale.unsqueeze(2).expand(-1, -1, num_windows, -1, -1)
            adaptive_bias  = adaptive_bias.unsqueeze(2).expand(-1, -1, num_windows, -1, -1)
        else:
            adaptive_scale = 0.0
            adaptive_bias  = 0.0

        base_filter_local = self.base_filter_local.unsqueeze(0).unsqueeze(2)
        base_bias_local   = self.base_bias_local.unsqueeze(0).unsqueeze(2)

        effective_filter_local = base_filter_local * (1 + adaptive_scale)
        effective_bias_local   = base_bias_local + adaptive_bias

        F_fft_local_mod = F_fft_local * effective_filter_local + effective_bias_local
        F_fft_local_nl  = self.complex_activation(F_fft_local_mod)

        x_local_win = torch.fft.irfft(F_fft_local_nl, dim=3, n=win_size, norm='ortho')
        x_local_padded = x_local_win.reshape(B, H, total_len, D)

        if pad_len > 0:
            x_local = x_local_padded[:, :, :N, :]
        else:
            x_local = x_local_padded
        return x_local

    def forward(self, x):
        """
        x: (B, seq_len, embed_dim)
        return: (B, seq_len, embed_dim), *no extra skip addition here*.
        """
        B, N, D = x.shape

        # Internal LN
        x_norm = self.pre_norm(x)
        x_heads = x_norm.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Global adaptation
        if self.adaptive:
            context_global = x_norm.mean(dim=1)  # (B, D)
            adapt_params_global = self.adaptive_mlp_global(context_global)
            adapt_params_global = adapt_params_global.view(
                B, self.num_heads, self.freq_bins_global, 2
            )
        else:
            adapt_params_global = None

        # Global transform
        x_global = self._transform_global(x_heads, adapt_params_global)

        # Local branch if used
        if self.use_local_branch:
            if self.adaptive:
                context_local = x_norm.mean(dim=1)
                adapt_params_local = self.adaptive_mlp_local(context_local)
                adapt_params_local = adapt_params_local.view(
                    B, self.num_heads, self.freq_bins_local, 2
                )
            else:
                adapt_params_local = None

            x_local = self._transform_local(x_heads, adapt_params_local)

            # Gating
            alpha = self.fusion_mlp(x_norm.mean(dim=1))  # => (B, num_heads)
            alpha = torch.sigmoid(alpha).view(B, self.num_heads, 1, 1)
            x_fused_heads = alpha * x_global + (1 - alpha) * x_local
        else:
            x_fused_heads = x_global

        x_fused = x_fused_heads.permute(0, 2, 1, 3).reshape(B, N, D)
        # **No** skip-add inside. Just dropout for the final.
        return self.dropout(x_fused)


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with:
      - spectral attention
      - MLP
      - DropPath
      - (pre)-LayerNorm
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
        # x + attention
        x = x + self.drop_path(self.attention(x))
        # x + mlp
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x
