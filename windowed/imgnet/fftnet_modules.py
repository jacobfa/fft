import math
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# DropPath (Stochastic Depth) Module
###############################################################################
class DropPath(nn.Module):
    """
    DropPath as described in "Deep Networks with Stochastic Depth".
    During training, randomly drop entire residual paths.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Generate random tensor in shape (B, 1, 1, ..., 1)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor

###############################################################################
# 1) ComplexLinear: Linear layer for complex-valued inputs.
###############################################################################
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_real = nn.Parameter(torch.empty(out_features))
            self.bias_imag = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
        if self.bias_real is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_real)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(self, x):
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        a, b = x.real, x.imag
        out_real = F.linear(a, self.weight_real) - F.linear(b, self.weight_imag)
        out_imag = F.linear(a, self.weight_imag) + F.linear(b, self.weight_real)
        if self.bias_real is not None:
            out_real = out_real + self.bias_real
            out_imag = out_imag + self.bias_imag
        return torch.complex(out_real, out_imag)

###############################################################################
# 2) MultiHeadSpectralAttention: Combines global FFT and local STFT-based attention.
###############################################################################
class MultiHeadSpectralAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        seq_len,
        num_heads=4,
        dropout=0.1,
        adaptive=True,
        window_size=16,
        hop_size=8
    ):
        """
        Args:
            embed_dim: total embedding dimension (e.g., 192).
            seq_len: number of tokens (e.g., 64 patches + 1 CLS = 65).
            num_heads: number of attention heads.
            dropout: dropout probability.
            adaptive: if True, uses an MLP to modulate a base filter.
            window_size: local STFT window size.
            hop_size: step (hop) size for the STFT windows.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        self.adaptive = adaptive

        self.global_freq_bins = seq_len // 2 + 1
        self.local_freq_bins = window_size // 2 + 1

        self.base_filter_global = nn.Parameter(torch.ones(num_heads, self.global_freq_bins, 1))
        self.base_filter_local = nn.Parameter(torch.ones(num_heads, self.local_freq_bins, 1))

        if adaptive:
            self.adaptive_mlp_global = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_heads * self.global_freq_bins)
            )
            self.adaptive_mlp_local = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_heads * self.local_freq_bins)
            )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.modrelu_bias_global = nn.Parameter(torch.full((num_heads, self.global_freq_bins, 1), -0.1))
        self.modrelu_bias_local = nn.Parameter(torch.full((num_heads, self.local_freq_bins, 1), -0.1))

        fused_dim = self.local_freq_bins + self.global_freq_bins
        self.freq_fusion = ComplexLinear(fused_dim, self.global_freq_bins, bias=True)
        self.time_fusion = nn.Linear(2, 1, bias=False)

        self.window_size = window_size
        self.hop_size = hop_size
        self.register_buffer("hann_window", torch.hann_window(self.window_size))

    def modrelu(self, z, bias):
        z_abs = torch.abs(z)
        z_abs_clamped = torch.clamp(z_abs, min=1e-3)
        activated = torch.relu(z_abs + bias)
        scale = activated / z_abs_clamped
        return z * scale

    def forward(self, x):
        # x: (B, seq_len, embed_dim)
        B, N, D = x.shape
        assert N == self.seq_len, f"Expected seq_len={self.seq_len}, got {N}."
        # Reshape to (B, num_heads, seq_len, head_dim)
        x_heads = x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        #######################################################################
        # 1) Global Path: rFFT along seq_len dimension
        #######################################################################
        F_global = torch.fft.rfft(x_heads, dim=2, norm="ortho")  # (B, num_heads, global_freq_bins, head_dim)
        if self.adaptive:
            context_global = x.mean(dim=1)  # (B, embed_dim)
            mod_global = self.adaptive_mlp_global(context_global)  # (B, num_heads*global_freq_bins)
            mod_global = torch.tanh(mod_global).view(B, self.num_heads, self.global_freq_bins, 1)
            filter_used_global = self.base_filter_global.unsqueeze(0) + mod_global
        else:
            filter_used_global = self.base_filter_global.unsqueeze(0)
        F_global = F_global * filter_used_global
        F_global = self.modrelu(F_global, self.modrelu_bias_global.unsqueeze(0))

        #######################################################################
        # 2) Local Path: Windowed STFT over the token dimension.
        #######################################################################
        # Unfold along dimension=2 (the seq_len dimension)
        x_unfold = x_heads.unfold(dimension=2, size=self.window_size, step=self.hop_size)
        # Default shape: (B, num_heads, num_windows, head_dim, window_size)
        # Swap last two dimensions so that window_size is at dim=3:
        x_unfold = x_unfold.permute(0, 1, 2, 4, 3)  # => (B, num_heads, num_windows, window_size, head_dim)
        w = self.hann_window.view(1, 1, 1, self.window_size, 1)
        x_win = x_unfold * w  # (B, num_heads, num_windows, window_size, head_dim)
        # rFFT along dim=3 (the window dimension)
        F_local = torch.fft.rfft(x_win, dim=3, norm="ortho")  # (B, num_heads, num_windows, local_freq_bins, head_dim)
        if self.adaptive:
            context_local = x.mean(dim=1)
            mod_local = self.adaptive_mlp_local(context_local)
            mod_local = torch.tanh(mod_local).view(B, self.num_heads, self.local_freq_bins, 1)
            filter_used_local = self.base_filter_local.unsqueeze(0) + mod_local
        else:
            filter_used_local = self.base_filter_local.unsqueeze(0)
        filter_used_local = filter_used_local.unsqueeze(2)  # (B, num_heads, 1, local_freq_bins, 1)
        F_local = F_local * filter_used_local
        F_local = self.modrelu(F_local, self.modrelu_bias_local.unsqueeze(0).unsqueeze(2))
        # Average over the window dimension (num_windows)
        F_local_agg = F_local.mean(dim=2)  # (B, num_heads, local_freq_bins, head_dim)

        #######################################################################
        # 3) Frequency Fusion: Concatenate local and global frequencies.
        #######################################################################
        F_cat = torch.cat([F_local_agg, F_global], dim=2)  # (B, num_heads, local+global, head_dim)
        F_cat = F_cat.permute(0, 1, 3, 2)  # (B, num_heads, head_dim, local+global)
        B_, H_, HD_, fused_dim = F_cat.shape
        F_cat_2d = F_cat.reshape(B_ * H_ * HD_, fused_dim)
        F_fused_2d = self.freq_fusion(F_cat_2d)  # (B_*H_*HD_, global_freq_bins) as complex
        F_fused = F_fused_2d.view(B_, H_, HD_, -1).permute(0, 1, 3, 2)

        #######################################################################
        # 4) Inverse FFT: Global reconstruction and local iSTFT with overlap-add.
        #######################################################################
        x_time_fused = torch.fft.irfft(F_fused, dim=2, n=self.seq_len, norm="ortho")  # (B, num_heads, seq_len, head_dim)
        x_local_win = torch.fft.irfft(F_local, dim=3, n=self.window_size, norm="ortho")
        x_local_win = x_local_win * w  # (B, num_heads, num_windows, window_size, head_dim)
        x_local_reconstructed = x_heads.new_zeros((B, self.num_heads, self.seq_len, self.head_dim))
        num_windows = x_local_win.shape[2]
        for w_idx in range(num_windows):
            start = w_idx * self.hop_size
            end = start + self.window_size
            x_local_reconstructed[:, :, start:end] += x_local_win[:, :, w_idx]

        #######################################################################
        # 5) Time-domain Fusion: Combine global and local reconstructions.
        #######################################################################
        x_time_cat = torch.stack([x_time_fused, x_local_reconstructed], dim=-1)  # (B, num_heads, seq_len, head_dim, 2)
        x_fused_time = self.time_fusion(x_time_cat).squeeze(-1)  # (B, num_heads, seq_len, head_dim)
        x_out = x_fused_time.permute(0, 2, 1, 3).reshape(B, N, D)  # merge heads back

        return self.norm(x + self.dropout(x_out))

###############################################################################
# 3) TransformerEncoderBlock with DropPath.
###############################################################################
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1, attention_module=None, drop_path_rate=0.0):
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
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attention(x))
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x
