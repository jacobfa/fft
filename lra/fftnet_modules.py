import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSpectralAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        seq_len,
        num_heads=4,
        dropout=0.1,
        adaptive=True,
        fuse_method="gating",  # or "concat"
        wavelet="haar"         # only "haar" implemented here
    ):
        """
        A multi‐head spectral attention block that combines:
          1) Global interactions via FFT.
          2) Local interactions via a single-level Wavelet transform.

        The results of these two paths are fused either by:
          - gating (default): alpha * FFT_path + (1 - alpha) * Wavelet_path
          - concatenation: [FFT_path, Wavelet_path] -> linear projection

        Args:
            embed_dim: total embedding dimension.
            seq_len: sequence length (e.g. number of patches + 1 for a class token).
            num_heads: number of attention heads.
            dropout: dropout rate.
            adaptive: if True, uses MLPs to modulate base filters for both FFT and wavelet paths.
            fuse_method: how to fuse FFT-based output and wavelet-based output:
                         "gating" or "concat"
            wavelet: wavelet type, only "haar" is implemented.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.adaptive = adaptive
        self.fuse_method = fuse_method

        # -- FFT-related parameters --
        # Real rFFT reduces the frequency dimension to seq_len // 2 + 1.
        self.freq_bins = seq_len // 2 + 1

        # Base filter for FFT path: shape (num_heads, freq_bins, 1)
        self.base_filter_fft = nn.Parameter(torch.ones(num_heads, self.freq_bins, 1))

        # modReLU bias for the complex FFT coefficients
        self.modrelu_bias_fft = nn.Parameter(
            torch.full((num_heads, self.freq_bins, 1), -0.1)
        )

        # -- Wavelet-related parameters --
        # We implement a single-level Haar transform for local dependencies.
        # We no longer require seq_len to be even; we will pad if needed.
        # We'll store separate filters for approx and detail.
        # Shape for each: (num_heads, wavelet_len, 1), though wavelet_len = ceil(seq_len/2) if odd
        # However, we will just create them for the largest possible half = (seq_len+1)//2
        self.wavelet_len = (seq_len + 1) // 2  # maximum half-length for odd seq_len

        self.base_filter_wavelet_approx = nn.Parameter(
            torch.ones(num_heads, self.wavelet_len, 1)
        )
        self.base_filter_wavelet_detail = nn.Parameter(
            torch.ones(num_heads, self.wavelet_len, 1)
        )

        # -- Adaptive MLPs (optional) --
        # If adaptive=True, we learn how to modulate both FFT and wavelet filters
        # from a global context vector (mean-pooled input).
        if adaptive:
            # For FFT
            self.adaptive_mlp_fft = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_heads * self.freq_bins),
            )
            # For wavelet: need to modulate both approx & detail
            # => 2 * wavelet_len * num_heads total
            self.adaptive_mlp_wavelet = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 2 * num_heads * self.wavelet_len),
            )

        # -- Fusion method --
        if fuse_method not in ["gating", "concat"]:
            raise ValueError('fuse_method must be "gating" or "concat"')

        if fuse_method == "gating":
            # A learnable scalar (or vector) gate can also be used; here we use a single scalar:
            self.alpha = nn.Parameter(torch.tensor(0.0))
        else:
            # If we concat, we need a linear to go back to embed_dim
            self.proj_concat = nn.Linear(embed_dim * 2, embed_dim)

    def modrelu(self, z):
        """
        modReLU activation for complex-valued input z.
        z: (B, num_heads, freq_bins, head_dim) [complex tensor]
        Returns the same shape, with modReLU applied channel-wise.
        """
        z_abs = torch.abs(z)
        bias = self.modrelu_bias_fft.unsqueeze(0)  # shape (1, num_heads, freq_bins, 1)
        # Clamp the magnitude to avoid division by small numbers
        z_abs_clamped = torch.clamp(z_abs, min=1e-3)
        # modReLU => ReLU(|z| + bias) or GELU(|z| + bias). We'll keep it consistent with code:
        activated = F.gelu(z_abs + bias)
        # Scale factor
        scale = activated / z_abs_clamped
        return z * scale

    def wavelet_transform(self, x):
        """
        Single-level Haar wavelet transform that supports odd sequence length by zero-padding.
        x: shape (B, L), real.

        Returns:
          approx: (B, L_pad//2)
          detail: (B, L_pad//2)
          pad_len: how many zeros we added on the right if L is odd (0 or 1).
        """
        B, L = x.shape
        pad_len = 0
        # If length is odd, pad one zero on the right
        if L % 2 != 0:
            pad_len = 1
            x = F.pad(x, (0, pad_len), mode="constant", value=0.0)  # shape (B, L+1)
            L = L + pad_len

        # L is now even; do single-level Haar
        half = L // 2
        x_even = x[:, 0::2]
        x_odd = x[:, 1::2]
        approx = (x_even + x_odd) / math.sqrt(2)
        detail = (x_even - x_odd) / math.sqrt(2)
        return approx, detail, pad_len

    def wavelet_inverse(self, approx, detail, pad_len):
        """
        Inverse single-level Haar wavelet transform that removes padding if it was applied.
        approx, detail: (B, half)
        pad_len: how many zeros were added (0 or 1).
        
        Returns:
          x: (B, original_L)  # i.e., without the padding if present.
        """
        B, half = approx.shape
        L = half * 2  # the padded length
        x = approx.new_zeros(B, L)

        # x_even = (approx + detail)/sqrt(2)
        # x_odd  = (approx - detail)/sqrt(2)
        x[:, 0::2] = (approx + detail) / math.sqrt(2)
        x[:, 1::2] = (approx - detail) / math.sqrt(2)

        # If we padded, remove the extra columns at the end
        if pad_len > 0:
            x = x[:, :-pad_len]
        return x

    def forward(self, x):
        """
        x: (B, seq_len, embed_dim)
        Returns:
          (B, seq_len, embed_dim) after combining global (FFT) and local (Wavelet) paths.
        """
        B, N, D = x.shape
        # x_heads => (B, num_heads, seq_len, head_dim)
        x_heads = x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 1) Global path: Real FFT -> filter -> modReLU -> iFFT
        # -----------------------------------------------------
        F_fft = torch.fft.rfft(x_heads, dim=2, norm="ortho")  
        # shape (B, num_heads, freq_bins, head_dim)

        # Adaptive filter (FFT)
        if self.adaptive:
            # Global context vector: (B, embed_dim)
            context = x.mean(dim=1)  # mean over tokens
            mod_fft = self.adaptive_mlp_fft(context)  # => (B, num_heads * freq_bins)
            mod_fft = torch.tanh(mod_fft)
            mod_fft = mod_fft.view(B, self.num_heads, self.freq_bins, 1)
            filter_fft = self.base_filter_fft.unsqueeze(0) + mod_fft
            # shape (B, num_heads, freq_bins, 1)
        else:
            # Static base filter
            filter_fft = self.base_filter_fft.unsqueeze(0)  # (1, num_heads, freq_bins, 1)

        # Multiply FFT coefficients by filter
        F_fft = F_fft * filter_fft
        # Complex modReLU
        F_fft = self.modrelu(F_fft)
        # iFFT
        x_fft = torch.fft.irfft(F_fft, dim=2, n=self.seq_len, norm="ortho")
        # x_fft => (B, num_heads, seq_len, head_dim)
        x_fft = x_fft.permute(0, 2, 1, 3).reshape(B, N, D)

        # 2) Local path: Wavelet transform -> filter -> activation -> inverse wavelet
        # ---------------------------------------------------------------------------
        # Flatten (B, num_heads, seq_len, head_dim) => (B*num_heads*head_dim, seq_len)
        x_heads_flat = x_heads.reshape(B * self.num_heads * self.head_dim, N)

        # Single-level Haar wavelet (with optional padding)
        wave_approx, wave_detail, pad_len = self.wavelet_transform(x_heads_flat)

        # wave_approx, wave_detail => each shape: (B*num_heads*head_dim, N_pad//2)
        # Now reshape to (B, num_heads, head_dim, wavelet_len) to broadcast filters
        wave_approx = wave_approx.view(B, self.num_heads, self.head_dim, -1)
        wave_detail = wave_detail.view(B, self.num_heads, self.head_dim, -1)

        # We want to multiply across wavelet_len dimension => reorder to (B, num_heads, wavelet_len, head_dim)
        wave_approx = wave_approx.permute(0, 1, 3, 2)
        wave_detail = wave_detail.permute(0, 1, 3, 2)

        # Adaptive wavelet filter
        if self.adaptive:
            # Re-use the same global context
            wave_out = self.adaptive_mlp_wavelet(context)  # (B, 2 * num_heads * wavelet_len)
            wave_out = torch.tanh(wave_out)

            wave_out = wave_out.view(B, self.num_heads, 2, -1)  
            # wave_out[:, :, 0, :] => approx filter portion
            # wave_out[:, :, 1, :] => detail filter portion
            wave_mod_approx = wave_out[:, :, 0, :]  # (B, num_heads, wavelet_len)
            wave_mod_detail = wave_out[:, :, 1, :]  # (B, num_heads, wavelet_len)

            wave_mod_approx = wave_mod_approx.unsqueeze(-1)  # => (B, num_heads, wavelet_len, 1)
            wave_mod_detail = wave_mod_detail.unsqueeze(-1)  # => (B, num_heads, wavelet_len, 1)

            filter_wavelet_approx = (
                self.base_filter_wavelet_approx.unsqueeze(0)[:, :, : wave_approx.size(2), :] + wave_mod_approx
            )  
            filter_wavelet_detail = (
                self.base_filter_wavelet_detail.unsqueeze(0)[:, :, : wave_detail.size(2), :] + wave_mod_detail
            )
        else:
            # Static base filters, slice them to the current wavelet_len
            filter_wavelet_approx = self.base_filter_wavelet_approx.unsqueeze(0)[:, :, : wave_approx.size(2), :]
            filter_wavelet_detail = self.base_filter_wavelet_detail.unsqueeze(0)[:, :, : wave_detail.size(2), :]

        # Apply wavelet filters
        wave_approx = wave_approx * filter_wavelet_approx  # (B, num_heads, wavelet_len, head_dim)
        wave_detail = wave_detail * filter_wavelet_detail  # (B, num_heads, wavelet_len, head_dim)

        # Optional nonlinearity on wavelet coefficients
        wave_approx = F.gelu(wave_approx)
        wave_detail = F.gelu(wave_detail)

        # Inverse wavelet
        # => first restore shape to (B*num_heads*head_dim, wavelet_len)
        wave_approx = wave_approx.permute(0, 1, 3, 2).reshape(
            B * self.num_heads * self.head_dim, -1
        )
        wave_detail = wave_detail.permute(0, 1, 3, 2).reshape(
            B * self.num_heads * self.head_dim, -1
        )

        x_wave_flat = self.wavelet_inverse(wave_approx, wave_detail, pad_len=pad_len)
        # => shape (B*num_heads*head_dim, seq_len)
        x_wave = x_wave_flat.view(B, self.num_heads, self.head_dim, N)
        x_wave = x_wave.permute(0, 3, 1, 2).reshape(B, N, D)

        # 3) Fuse global (FFT) and local (Wavelet) paths
        # ---------------------------------------------
        if self.fuse_method == "gating":
            # A simple learned scalar gate alpha in [-∞, +∞], we apply sigmoid to keep it in [0,1].
            alpha = torch.sigmoid(self.alpha)
            x_fused = alpha * x_fft + (1.0 - alpha) * x_wave
        else:
            # Concat along the embedding dimension => (B, N, 2*D), then project back to (B, N, D)
            x_fused = torch.cat([x_fft, x_wave], dim=-1)
            x_fused = self.proj_concat(x_fused)

        # 4) Residual connection + dropout + layernorm
        return self.norm(x + self.dropout(x_fused))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1, attention_module=None):
        """
        A generic Transformer encoder block.
          - embed_dim: embedding dimension.
          - mlp_ratio: expansion factor for the MLP.
          - dropout: dropout rate.
          - attention_module: a module handling self-attention (or spectral attention).
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
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # First, do the multi-head spectral/wavelet-based attention step
        x_attn = self.attention(x)  # (B, seq_len, embed_dim)
        # Then, pass the result through an MLP
        x_out = x_attn + self.mlp(self.norm(x_attn))
        return x_out
