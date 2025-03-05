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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# 1) ComplexLinear (unchanged)
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
        # Ensure input is complex
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        a, b = x.real, x.imag  # real and imaginary parts
        # Linear transforms
        out_real = F.linear(a, self.weight_real) - F.linear(b, self.weight_imag)
        out_imag = F.linear(a, self.weight_imag) + F.linear(b, self.weight_real)
        # Bias
        if self.bias_real is not None:
            out_real = out_real + self.bias_real
            out_imag = out_imag + self.bias_imag
        return torch.complex(out_real, out_imag)

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
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        self.adaptive = adaptive

        # Frequency bins for global and local
        self.global_freq_bins = seq_len // 2 + 1
        self.local_freq_bins = window_size // 2 + 1

        # Base filters
        self.base_filter_global = nn.Parameter(torch.ones(num_heads, self.global_freq_bins, 1))
        self.base_filter_local = nn.Parameter(torch.ones(num_heads, self.local_freq_bins, 1))

        # (Optional) MLP to adapt the filters based on the global context
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

        # modReLU biases
        self.modrelu_bias_global = nn.Parameter(torch.full((num_heads, self.global_freq_bins, 1), -0.1))
        self.modrelu_bias_local = nn.Parameter(torch.full((num_heads, self.local_freq_bins, 1), -0.1))

        # Simple linear to fuse the two time-domain branches
        # We'll stack them along an extra dimension and do a linear from 2 -> 1
        self.time_fusion = nn.Linear(2, 1, bias=False)

        # STFT windowing
        self.window_size = window_size
        self.hop_size = hop_size
        self.register_buffer("hann_window", torch.hann_window(self.window_size))

    def modrelu(self, z, bias):
        """
        z is complex: modReLU => ReLU(|z| + bias) * z / |z|
        """
        z_abs = torch.abs(z)
        z_abs_clamped = torch.clamp(z_abs, min=1e-6)  # avoid division by zero
        activated = torch.relu(z_abs + bias)
        scale = activated / z_abs_clamped
        return z * scale

    def forward(self, x):
        """
        x: (B, seq_len, embed_dim)
        """
        B, N, D = x.shape
        assert N == self.seq_len, f"Expected seq_len={self.seq_len}, got {N}."

        # Reshape to (B, num_heads, seq_len, head_dim)
        x_heads = x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # shape => (B, num_heads, seq_len, head_dim)

        # ---------------------------------------------------------------------
        # 1) Global Path: single FFT across all tokens
        # ---------------------------------------------------------------------
        F_global = torch.fft.rfft(x_heads, dim=2, norm="ortho")
        # shape => (B, num_heads, global_freq_bins, head_dim)

        if self.adaptive:
            # use global average pooling for context
            context_global = x.mean(dim=1)  # shape (B, embed_dim)
            mod_global = self.adaptive_mlp_global(context_global)
            mod_global = torch.tanh(mod_global).view(B, self.num_heads, self.global_freq_bins, 1)
            # Add to base filter
            filter_used_global = self.base_filter_global.unsqueeze(0) + mod_global
        else:
            filter_used_global = self.base_filter_global.unsqueeze(0)

        # Apply filter + modReLU
        F_global = F_global * filter_used_global
        F_global = self.modrelu(F_global, self.modrelu_bias_global.unsqueeze(0))

        # Inverse FFT => time-domain representation
        x_global_time = torch.fft.irfft(F_global, n=self.seq_len, dim=2, norm="ortho")
        # shape => (B, num_heads, seq_len, head_dim)

        # ---------------------------------------------------------------------
        # 2) Local Path: windowed STFT
        # ---------------------------------------------------------------------
        # chunk up tokens in frames of size=window_size, step=hop_size
        # x_heads shape => (B, num_heads, seq_len, head_dim)
        x_unfold = x_heads.unfold(dimension=2, size=self.window_size, step=self.hop_size)
        # shape => (B, num_heads, num_windows, head_dim, window_size)
        # reorder to => (B, num_heads, num_windows, window_size, head_dim)
        x_unfold = x_unfold.permute(0, 1, 2, 4, 3)

        # Multiply by Hann window => shape (1,1,1,window_size,1)
        w = self.hann_window.view(1, 1, 1, self.window_size, 1)
        x_win = x_unfold * w

        # rFFT over dimension=3 => window_size
        F_local = torch.fft.rfft(x_win, dim=3, norm="ortho")
        # shape => (B, num_heads, num_windows, local_freq_bins, head_dim)

        if self.adaptive:
            context_local = x.mean(dim=1)  # (B, embed_dim)
            mod_local = self.adaptive_mlp_local(context_local)
            mod_local = torch.tanh(mod_local).view(B, self.num_heads, self.local_freq_bins, 1)
            filter_used_local = self.base_filter_local.unsqueeze(0) + mod_local
        else:
            filter_used_local = self.base_filter_local.unsqueeze(0)

        # Expand across num_windows dimension (index=2)
        filter_used_local = filter_used_local.unsqueeze(2)  # => shape (B, num_heads, 1, local_freq_bins, 1)
        F_local = F_local * filter_used_local
        # modReLU
        F_local = self.modrelu(F_local, self.modrelu_bias_local.unsqueeze(0).unsqueeze(2))

        # iFFT each window => (B, num_heads, num_windows, window_size, head_dim)
        x_local_win = torch.fft.irfft(F_local, n=self.window_size, dim=3, norm="ortho")
        # multiply by Hann for typical iSTFT
        x_local_win = x_local_win * w

        # Overlap-add to reconstruct local time path
        x_local_time = x_heads.new_zeros((B, self.num_heads, self.seq_len, self.head_dim))
        num_windows = x_local_win.shape[2]
        for w_idx in range(num_windows):
            start = w_idx * self.hop_size
            end = start + self.window_size
            x_local_time[:, :, start:end] += x_local_win[:, :, w_idx]

        # ---------------------------------------------------------------------
        # 3) Fuse global + local time
        # ---------------------------------------------------------------------
        # shape => (B, num_heads, seq_len, head_dim, 2)
        x_time_stack = torch.stack([x_global_time, x_local_time], dim=-1)
        # linear from 2 -> 1 across last dim
        x_fused_time = self.time_fusion(x_time_stack).squeeze(-1)  # => (B, num_heads, seq_len, head_dim)

        # Merge heads back => (B, seq_len, embed_dim)
        x_out = x_fused_time.permute(0, 2, 1, 3).reshape(B, N, D)

        # Final residual & layer norm
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
