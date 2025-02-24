import torch
import torch.nn as nn

class MultiHeadSpectralAttention(nn.Module):
    def __init__(self, embed_dim, seq_len, num_heads=4, dropout=0.1, adaptive=True):
        """
        Memory-efficient multi‐head spectral attention using real FFT.
          - embed_dim: total embedding dimension.
          - seq_len: sequence length (e.g. number of patches + 1 for the class token).
          - num_heads: number of attention heads.
          - dropout: dropout rate.
          - adaptive: if True, uses an MLP to modulate a base filter.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        self.adaptive = adaptive

        # When using rfft, the frequency dimension is reduced to seq_len//2+1.
        self.freq_bins = seq_len // 2 + 1

        # Base filter: shape (num_heads, freq_bins, 1)
        self.base_filter = nn.Parameter(torch.ones(num_heads, self.freq_bins, 1))
        if adaptive:
            # Adaptive MLP outputs (B, num_heads * freq_bins) reshaped to (B, num_heads, freq_bins, 1)
            self.adaptive_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_heads * self.freq_bins)
            )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize modReLU bias with a small negative value for stability.
        self.modrelu_bias = nn.Parameter(torch.full((num_heads, self.freq_bins, 1), -0.1))

    def modrelu(self, z):
        """
        Applies the modReLU activation on a complex tensor.
          - z: complex tensor of shape (B, num_heads, freq_bins, head_dim)
        Returns:
          - Activated complex tensor with the same shape.
        """
        z_abs = torch.abs(z)
        bias = self.modrelu_bias.unsqueeze(0)
        # Clamp the magnitude to avoid division by very small numbers.
        z_abs_clamped = torch.clamp(z_abs, min=1e-3)
        # Compute the activated magnitude: ReLU(|z| + bias)
        activated = torch.relu(z_abs + bias)
        # Compute scale safely.
        scale = activated / z_abs_clamped
        return z * scale

    def forward(self, x):
        # x shape: (B, seq_len, embed_dim)
        B, N, D = x.shape

        # Reshape to (B, num_heads, seq_len, head_dim)
        x_heads = x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute the real FFT along the token dimension using orthogonal normalization.
        F_fft = torch.fft.rfft(x_heads, dim=2, norm='ortho')
        
        # Compute the frequency filter.
        if self.adaptive:
            # Global context: (B, embed_dim)
            context = x.mean(dim=1)
            # Adaptive modulation: (B, num_heads * freq_bins)
            mod = self.adaptive_mlp(context)
            # Bound the MLP outputs to keep them in a reasonable range.
            mod = torch.tanh(mod)
            mod = mod.view(B, self.num_heads, self.freq_bins, 1)
            filter_used = self.base_filter.unsqueeze(0) + mod  # (B, num_heads, freq_bins, 1)
        else:
            filter_used = self.base_filter.unsqueeze(0)  # (1, num_heads, freq_bins, 1)

        # Apply the frequency filter.
        F_fft = F_fft * filter_used
        
        # Apply modReLU activation to the complex FFT coefficients.
        F_fft = self.modrelu(F_fft)
        
        # Inverse FFT back to the token domain using orthogonal normalization.
        x_filtered = torch.fft.irfft(F_fft, dim=2, n=self.seq_len, norm='ortho')  # (B, num_heads, seq_len, head_dim)
        
        # Merge heads back to (B, seq_len, embed_dim).
        x_filtered = x_filtered.permute(0, 2, 1, 3).reshape(B, N, D)
        
        # Residual connection with dropout and layer normalization.
        return self.norm(x + self.dropout(x_filtered))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1, attention_module=None):
        """
        A generic Transformer encoder block.
          - embed_dim: embedding dimension.
          - mlp_ratio: expansion factor for the MLP.
          - dropout: dropout rate.
          - attention_module: a module handling self-attention.
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

    def forward(self, x):
        x = self.attention(x)
        x = x + self.mlp(self.norm(x))
        return x
