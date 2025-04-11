import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# 1) DropPath (Stochastic Depth)
###############################################################################
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

###############################################################################
# 2) MultiHeadSpectralWaveletAttention: Purely "spectral" approach
#    (1D FFT + single-level Haar wavelet transform) - no for loops, all on GPU
###############################################################################
class MultiHeadSpectralWaveletAttention(nn.Module):
    def __init__(self, embed_dim, seq_len, num_heads=4, dropout=0.1, adaptive=True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        self.adaptive = adaptive
        self.return_intermediates = False  # <-- new flag
        self.last_intermediates = {}       # <--- where we'll store outputs

        # ----- FFT filters -----
        self.freq_bins = self.seq_len // 2 + 1
        self.base_filter = nn.Parameter(torch.ones(num_heads, self.freq_bins, 1))
        self.base_bias   = nn.Parameter(torch.full((num_heads, self.freq_bins, 1), -0.1))
        
        if self.adaptive:
            self.adaptive_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_heads * self.freq_bins * 2)
            )

        sqrt2 = 1.0 / (2.0 ** 0.5)
        self.register_buffer("haar_low",  torch.tensor([sqrt2,  sqrt2]).reshape(1,1,2))
        self.register_buffer("haar_high", torch.tensor([sqrt2, -sqrt2]).reshape(1,1,2))

        self.dropout = nn.Dropout(dropout)
        self.pre_norm = nn.LayerNorm(embed_dim)

        # A place to store outputs if needed:
        self.last_intermediates = {}

    def complex_activation(self, z):
        mag = torch.abs(z)
        mag_act = F.gelu(mag)
        scale = mag_act / (mag + 1e-6)
        return z * scale

    def wavelet_transform_1d(self, x_in):
        B, H, N, D = x_in.shape
        grouped = x_in.permute(0,1,3,2).reshape(B*H*D, 1, N)
        cA = F.conv1d(grouped, self.haar_low, stride=2)
        cD = F.conv1d(grouped, self.haar_high, stride=2)
        cA_up = F.interpolate(cA, scale_factor=2, mode='nearest')
        cD_up = F.interpolate(cD, scale_factor=2, mode='nearest')
        x_rec_A = F.conv1d(cA_up, self.haar_low, stride=1, padding=1)
        x_rec_D = F.conv1d(cD_up, self.haar_high, stride=1, padding=1)
        x_rec = x_rec_A + x_rec_D

        if x_rec.size(-1) > N:
            x_rec = x_rec[..., :N]

        x_out = x_rec.reshape(B, H, D, -1).permute(0,1,3,2)
        return x_out

    def complex_activation(self, z):
        """
        Nonlinear activation on complex data: uses GELU on magnitude, then rescales.
        """
        mag = torch.abs(z)
        mag_act = F.gelu(mag)
        scale = mag_act / (mag + 1e-6)
        return z * scale

    def wavelet_transform_1d(self, x_in):
        """
        Single-level Haar wavelet transform + inverse done via convolutions.
        x_in: (B, H, N, D)
          B = batch, H = #heads, N = seq_len, D = head_dim
        Returns x_out: shape (B, H, N, D) after wavelet transform & reconstruction.
        """
        B, H, N, D = x_in.shape

        # Reshape for grouped convolution:
        # We treat each (b,h,d) triple as a separate channel for the wavelet operation
        # => shape (B*H*D, 1, N)
        grouped = x_in.permute(0,1,3,2).reshape(B*H*D, 1, N)  # (B*H*D, 1, N)

        # 1) Forward DWT (stride=2) for low-pass (cA) and high-pass (cD)
        cA = F.conv1d(grouped, self.haar_low, stride=2)    # shape ~ (B*H*D, 1, N/2)
        cD = F.conv1d(grouped, self.haar_high, stride=2)   # shape ~ (B*H*D, 1, N/2)

        # 2) Upsample cA, cD back to length ~ N
        #    (nearest interpolation to double the temporal dimension)
        cA_up = F.interpolate(cA, scale_factor=2, mode='nearest')
        cD_up = F.interpolate(cD, scale_factor=2, mode='nearest')

        # 3) Inverse DWT with same Haar filters:
        #    x_rec = conv1d(cA_up, low) + conv1d(cD_up, high)
        #    We'll do padding=1 to mimic 'same' convolution for kernel_size=2.
        x_rec_A = F.conv1d(cA_up, self.haar_low, stride=1, padding=1)
        x_rec_D = F.conv1d(cD_up, self.haar_high, stride=1, padding=1)
        x_rec = x_rec_A + x_rec_D  # shape ~ (B*H*D, 1, N+1) if N is even, or close.

        # 4) Ensure the final length is at least N; slice if there's an off-by-one
        if x_rec.size(-1) > N:
            x_rec = x_rec[..., :N]

        # 5) Reshape back to (B,H,N,D)
        x_out = x_rec.reshape(B, H, D, -1).permute(0,1,3,2)
        # => (B, H, N, D)

        return x_out

    def forward(self, x):
        """
        x: (B, seq_len, embed_dim)
        returns: (B, seq_len, embed_dim), with FFT + wavelet transform + residual.
        """
        B, N, D = x.shape
        # Pre-norm
        x_norm = self.pre_norm(x)

        # Reshape for heads => (B, H, N, head_dim)
        x_heads = x_norm.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        #####################
        # 1) FFT-based path
        #####################
        # rFFT along dim=2 (the seq_len dimension)
        F_fft = torch.fft.rfft(x_heads, dim=2, norm='ortho')  # (B, H, freq_bins, head_dim)

        if self.adaptive:
            # Global context => adaptive scale/bias
            context = x_norm.mean(dim=1)  # (B, embed_dim)
            adapt_params = self.adaptive_mlp(context)  # (B, H*freq_bins*2)
            adapt_params = adapt_params.view(B, self.num_heads, self.freq_bins, 2)
            adaptive_scale = adapt_params[..., 0:1]  # (B,H,freq_bins,1)
            adaptive_bias  = adapt_params[..., 1:2]  # (B,H,freq_bins,1)
        else:
            adaptive_scale = torch.zeros_like(self.base_filter).unsqueeze(0)  # (1,H,freq_bins,1)
            adaptive_bias  = torch.zeros_like(self.base_bias).unsqueeze(0)    # (1,H,freq_bins,1)

        # Combine base + adaptive
        effective_filter = self.base_filter.unsqueeze(0) * (1 + adaptive_scale)  # (B,H,freq_bins,1)
        effective_bias   = self.base_bias.unsqueeze(0) + adaptive_bias

        # Modulate in frequency domain
        F_fft_mod = F_fft * effective_filter + effective_bias
        # Nonlinear activation in frequency domain
        F_fft_nl = self.complex_activation(F_fft_mod)
        # Inverse FFT => (B, H, N, head_dim)
        x_fft = torch.fft.irfft(F_fft_nl, n=self.seq_len, dim=2, norm='ortho')

        # wavelet path
        x_wav = self.wavelet_transform_1d(x_heads)
        x_wav_nl = F.gelu(x_wav)

        # fuse paths
        x_fused = x_fft + x_wav_nl
        x_fused = x_fused.permute(0, 2, 1, 3).reshape(B, N, D)
        out = x + self.dropout(x_fused)

        # If we want to save intermediate outputs for visualization:
        if self.return_intermediates:
            # Store wavelet activations (before the final fuse) on CPU
            self.last_intermediates["x_wav"] = x_wav.detach().cpu()
            # Similarly, you could store the FFT path, e.g.:
            # self.last_intermediates["F_fft_mod"] = F_fft_mod.detach().cpu()

        return out

###############################################################################
# 3) TransformerEncoderBlock
###############################################################################
class TransformerEncoderBlock(nn.Module):
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
            raise ValueError("Must provide an attention module.")
        self.attention = attention_module

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """
        Because attention_module returns (x + residual), we do x + drop_path(...) only for MLP part.
        """
        # "Spectral" attention
        x = x + self.drop_path(self.attention(self.norm(x)))
        # Feed-forward MLP
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x
