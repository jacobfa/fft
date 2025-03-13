import math
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
      1) Global FFT over the entire sequence (with optional Hann window).
      2) Local wavelet transform (Haar DWT -> scale/bias -> iDWT).
      3) A learned gating MLP to fuse global and local branches.
    """
    def __init__(
        self,
        embed_dim,
        seq_len,
        num_heads=4,
        dropout=0.1,
        adaptive=True,
        use_local_branch=True,
        use_global_hann=True
    ):
        """
        Args:
          embed_dim: Total embedding dimension.
          seq_len:   Sequence length (#tokens).
          num_heads: Number of attention heads.
          dropout:   Dropout rate after iFFT/iDWT.
          adaptive:  If True, uses an MLP to produce (scale, bias) for freq/wavelet modulation.
          use_local_branch: Whether to enable local wavelet transform branch at all.
          use_global_hann:  Whether to apply a Hann window over the entire sequence in the global branch.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        self.adaptive = adaptive
        self.use_local_branch = use_local_branch
        self.use_global_hann = use_global_hann

        # ---------------------------------------------------
        #  Global (full-sequence) FFT branch
        # ---------------------------------------------------
        self.freq_bins_global = seq_len // 2 + 1
        self.base_filter_global = nn.Parameter(
            torch.ones(num_heads, self.freq_bins_global, 1)
        )
        self.base_bias_global = nn.Parameter(
            torch.full((num_heads, self.freq_bins_global, 1), -0.1)
        )

        # Optional Hann window for the entire sequence
        if use_global_hann:
            hann_g = torch.hann_window(seq_len, periodic=False)  # (seq_len,)
            hann_g = hann_g.view(1, 1, seq_len, 1)               # (1,1,N,1)
            self.register_buffer("hann_window_global", hann_g)

        # ---------------------------------------------------
        #  Local (wavelet) branch
        # ---------------------------------------------------
        if use_local_branch:
            # Single-level Haar wavelet => 2 subbands: approx & detail
            self.freq_bins_local = 2
            # Base scale & bias for wavelet subbands => shape = (num_heads, 2, 1)
            self.base_filter_local = nn.Parameter(
                torch.ones(num_heads, self.freq_bins_local, 1)
            )
            self.base_bias_local = nn.Parameter(
                torch.full((num_heads, self.freq_bins_local, 1), -0.1)
            )

            # Haar wavelet filters (low-pass, high-pass), shape (2,).
            lp = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)], dtype=torch.float32)
            hp = torch.tensor([1/math.sqrt(2),-1/math.sqrt(2)], dtype=torch.float32)
            wavelet_kernel = torch.stack([lp, hp], dim=0).unsqueeze(1)  # (2,1,2)
            self.register_buffer("wavelet_kernel", wavelet_kernel)

        else:
            self.freq_bins_local = None

        # ---------------------------------------------------
        #  Adaptive MLPs
        # ---------------------------------------------------
        if adaptive:
            # Global branch
            self.adaptive_mlp_global = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_heads * self.freq_bins_global * 2)
            )
            # Local (wavelet) branch
            if use_local_branch:
                self.adaptive_mlp_local = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, num_heads * self.freq_bins_local * 2)
                )

        self.dropout = nn.Dropout(dropout)
        # Pre-normalization for stability
        self.pre_norm = nn.LayerNorm(embed_dim)

        # If local branch used, we create a gating MLP to fuse local & global:
        # alpha ∈ [0,1]. Final = alpha*global + (1-alpha)*local
        if use_local_branch:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, num_heads)  # (B, num_heads)
            )
        else:
            self.fusion_mlp = None

    # ---------------------------------------------------
    #  Complex activation for FFT branch
    # ---------------------------------------------------
    def complex_activation(self, z):
        """
        Nonlinear activation in frequency domain (magnitude-based GELU).
        z: complex tensor
        """
        mag = torch.abs(z)
        mag_act = F.gelu(mag)
        scale = mag_act / (mag + 1e-6)
        return z * scale

    # ---------------------------------------------------
    #  Global branch = full-sequence FFT
    # ---------------------------------------------------
    def _transform_global(self, x_heads, adapt_params_global):
        """
        Perform global rFFT with optional Hann window and (optional) adaptive modulation.
        
        x_heads: (B, H, N, head_dim)
        Returns: (B, H, N, head_dim)
        """
        B, H, N, D = x_heads.shape

        # Optionally multiply by Hann window
        if self.use_global_hann:
            x_heads = x_heads * self.hann_window_global  # shape (1,1,N,1), broadcasts

        # Global rFFT
        F_fft = torch.fft.rfft(x_heads, dim=2, norm='ortho')  # (B,H,freq_bins_global,D), complex

        # Adaptive freq modulation
        if self.adaptive and adapt_params_global is not None:
            adaptive_scale = adapt_params_global[..., 0:1]  # (B,H,freq_bins_global,1)
            adaptive_bias  = adapt_params_global[..., 1:2]  # (B,H,freq_bins_global,1)
        else:
            adaptive_scale = torch.zeros(
                B, H, self.freq_bins_global, 1,
                device=x_heads.device, dtype=x_heads.dtype
            )
            adaptive_bias = torch.zeros_like(adaptive_scale)

        # Combine base + adaptive
        effective_filter = self.base_filter_global * (1 + adaptive_scale)
        effective_bias   = self.base_bias_global + adaptive_bias

        # Multiply & add in frequency domain
        F_fft_mod = F_fft * effective_filter + effective_bias
        # Nonlinear activation (complex)
        F_fft_nl = self.complex_activation(F_fft_mod)

        # iFFT
        x_global = torch.fft.irfft(F_fft_nl, dim=2, n=N, norm='ortho')
        return x_global

    # ---------------------------------------------------
    #  Local branch = wavelet transform (single-level Haar)
    # ---------------------------------------------------
    def _transform_wavelet(self, x_heads, adapt_params_local):
        """
        Single-level Haar wavelet transform for each head:
          1) DWT -> produces [approx, detail]
          2) scale/bias + GELU in wavelet domain
          3) iDWT -> reconstruct
        x_heads: (B, H, N, D)
        Returns: (B, H, N, D)
        """
        B, H, N, D = x_heads.shape
        original_N = N  # remember for unpadding

        # If N is odd, pad by 1 so we have an even length for stride=2 transform
        pad_len = (2 - (N % 2)) % 2
        if pad_len > 0:
            # pad shape => (B,H,pad_len,D)
            pad_zeros = torch.zeros(B, H, pad_len, D, dtype=x_heads.dtype, device=x_heads.device)
            x_heads = torch.cat([x_heads, pad_zeros], dim=2)
            N = N + pad_len  # update N

        # Reshape so that we treat 'D' as channels and 'N' as the length => (B*H, D, N)
        x_reshape = x_heads.permute(0,1,3,2).reshape(B*H, D, N)  # (B*H, D, N)

        # Forward DWT (conv1d w stride=2). We want 2 subbands per channel -> out_channels=2*D.
        # groups=D so each channel is filtered independently.
        kernel = self.wavelet_kernel.repeat(D, 1, 1)  # shape (2D,1,2)
        approx_detail = F.conv1d(
            x_reshape,
            kernel,
            stride=2,
            padding=0,
            groups=D
        )  # (B*H, 2D, N//2)

        half_len = approx_detail.shape[-1]  # = N//2
        # Reshape to (B,H,D,2,N//2)
        approx_detail = approx_detail.view(B, H, 2*D, half_len)
        approx_detail = approx_detail.view(B, H, D, 2, half_len)

        # Adaptive scale/bias for subbands => (B,H,2,2)
        if self.adaptive and adapt_params_local is not None:
            # shape => scale/bias in [B,H,2,1]
            adaptive_scale = adapt_params_local[..., 0:1]
            adaptive_bias  = adapt_params_local[..., 1:2]

            # Insert exactly one new dim to become (B,H,1,2,1),
            # then expand to (B,H,D,2,half_len).
            adaptive_scale = adaptive_scale.unsqueeze(2)  # => (B,H,1,2)
            adaptive_scale = adaptive_scale.expand(-1, -1, D, -1, half_len)  # => (B,H,D,2,half_len)
            adaptive_bias  = adaptive_bias.unsqueeze(2)   # => (B,H,1,2)
            adaptive_bias  = adaptive_bias.expand(-1, -1, D, -1, half_len)   # => (B,H,D,2,half_len)
        else:
            adaptive_scale = torch.zeros(
                B, H, D, 2, half_len,
                device=x_heads.device, dtype=x_heads.dtype
            )
            adaptive_bias = torch.zeros_like(adaptive_scale)

        # Base filters for wavelet subbands => shape (num_heads,2,1)
        # expand to (B,H,D,2,half_len)
        base_filter_local = self.base_filter_local.view(1, self.num_heads, 1, 2, 1)
        base_filter_local = base_filter_local.expand(B, -1, D, -1, half_len)
        base_bias_local   = self.base_bias_local.view(1, self.num_heads, 1, 2, 1)
        base_bias_local   = base_bias_local.expand(B, -1, D, -1, half_len)

        # Multiply & add in wavelet domain
        effective_filter_local = base_filter_local * (1 + adaptive_scale)
        effective_bias_local   = base_bias_local + adaptive_bias
        approx_detail_mod = approx_detail * effective_filter_local + effective_bias_local

        # Nonlinear activation (real)
        approx_detail_act = F.gelu(approx_detail_mod)

        # Now inverse wavelet (conv_transpose1d):
        # Reshape back to (B*H, 2D, N//2)
        approx_detail_act = approx_detail_act.view(B, H, D*2, half_len)  # (B,H,2D,N//2)
        approx_detail_act = approx_detail_act.view(B*H, 2*D, half_len)

        # out_channels = D, in_channels=2D, groups=D
        kernel_t = self.wavelet_kernel.repeat(D,1,1)  # shape (2D,1,2)
        x_recon = F.conv_transpose1d(
            approx_detail_act,
            kernel_t,
            stride=2,
            padding=0,
            groups=D
        )  # => (B*H, D, N) but note N = padded length

        # Reshape back to (B,H,N,D)
        x_recon = x_recon.view(B, H, D, N).permute(0,1,3,2)

        # If we padded, remove the extra from the length dimension
        if pad_len > 0:
            x_recon = x_recon[:, :, :original_N, :]

        return x_recon

    # ---------------------------------------------------
    #  Forward
    # ---------------------------------------------------
    def forward(self, x):
        """
        x shape: (B, seq_len, embed_dim)
        Returns: (B, seq_len, embed_dim) with residual.
        """
        B, N, D = x.shape
        x_norm = self.pre_norm(x)  # pre-LayerNorm
        # Reshape into heads => (B,H,N,head_dim)
        x_heads = x_norm.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Adaptive MLP context
        context = x_norm.mean(dim=1)  # (B, embed_dim)

        # Global branch adapt
        if self.adaptive:
            adapt_params_global = self.adaptive_mlp_global(context)  # (B, num_heads*freq_bins_global*2)
            adapt_params_global = adapt_params_global.view(
                B, self.num_heads, self.freq_bins_global, 2
            )
        else:
            adapt_params_global = None

        # Global FFT branch
        x_global = self._transform_global(x_heads, adapt_params_global)

        # Local wavelet branch
        if self.use_local_branch:
            if self.adaptive:
                adapt_params_local = self.adaptive_mlp_local(context)  # (B, num_heads*2*2)
                adapt_params_local = adapt_params_local.view(
                    B, self.num_heads, self.freq_bins_local, 2  # freq_bins_local=2
                )
            else:
                adapt_params_local = None

            x_local = self._transform_wavelet(x_heads, adapt_params_local)

            # Gating alpha in [0,1] per head
            alpha = self.fusion_mlp(context)  # (B, num_heads)
            alpha = torch.sigmoid(alpha).view(B, self.num_heads, 1, 1)
            x_fused_heads = alpha * x_global + (1 - alpha) * x_local
        else:
            x_fused_heads = x_global

        # Merge heads back
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
        # Spectral attention + residual
        x = x + self.drop_path(self.attention(x))
        # MLP + residual
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x
