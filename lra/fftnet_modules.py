import torch
import torch.nn as nn

class MultiHeadSpectralAttention(nn.Module):
    def __init__(self, embed_dim, seq_len, num_heads=4, dropout=0.1, adaptive=True):
        """
        Improved multi‐head spectral attention.
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

        # Base filter for each head (learnable parameter)
        self.base_filter = nn.Parameter(torch.ones(num_heads, seq_len, self.head_dim))
        if adaptive:
            # MLP to generate modulation for each head based on global context.
            self.adaptive_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_heads * seq_len * self.head_dim)
            )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        # x shape: (B, seq_len, embed_dim)
        B, N, D = x.shape

        # Reshape x to (B, num_heads, seq_len, head_dim) without extra copies
        x_heads = x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute FFT along the token dimension; returns a complex tensor.
        F_fft = torch.fft.fft(x_heads, dim=2)
        
        # Compute and prepare the filter.
        if self.adaptive:
            # Global context from all tokens
            context = x.mean(dim=1)  # (B, embed_dim)
            mod = self.adaptive_mlp(context).view(B, self.num_heads, self.seq_len, self.head_dim)
            filter_used = self.base_filter.unsqueeze(0) + mod  # (B, num_heads, seq_len, head_dim)
        else:
            filter_used = self.base_filter.unsqueeze(0)  # (1, num_heads, seq_len, head_dim)

        # Apply the filter in the frequency domain.
        F_fft = F_fft * filter_used
        
        # Non-linear activation for the frequency domain
        real_part = F_fft.real
        imag_part = F_fft.imag
        
        F_fft = self.activation(real_part) + 1j * self.activation(imag_part)

        # Inverse FFT back to the token domain; take only the real part.
        x_filtered = torch.fft.ifft(F_fft, dim=2).real  # (B, num_heads, seq_len, head_dim)
        
        # Merge heads back: (B, seq_len, embed_dim)
        x_filtered = x_filtered.permute(0, 2, 1, 3).reshape(B, N, D)
        
        # Residual connection, dropout, and layer normalization
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
