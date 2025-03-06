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

class MultiHeadSpectralAttention(nn.Module):
    def __init__(self, embed_dim, seq_len, num_heads=4, dropout=0.1, adaptive=True):
        """
        Spectral attention module that introduces additional nonlinearity and adaptivity
        while maintaining an n log n computational overhead.

        Parameters:
          - embed_dim: Total embedding dimension.
          - seq_len: Sequence length (e.g. number of tokens, including class token).
          - num_heads: Number of attention heads.
          - dropout: Dropout rate applied after iFFT.
          - adaptive: If True, uses an MLP to generate both multiplicative and additive adaptive modulations.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        self.adaptive = adaptive

        # Frequency bins for rFFT: (seq_len//2 + 1)
        self.freq_bins = seq_len // 2 + 1

        # Base multiplicative filter: one per head and frequency bin.
        self.base_filter = nn.Parameter(torch.ones(num_heads, self.freq_bins, 1))
        # Base additive bias (think of it as a learned offset on the frequency magnitudes).
        self.base_bias = nn.Parameter(torch.full((num_heads, self.freq_bins, 1), -0.1))

        if adaptive:
            # Adaptive MLP: produces 2 values per head & frequency bin (scale and bias modulation).
            self.adaptive_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_heads * self.freq_bins * 2)
            )

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
        # Nonlinear transformation on the magnitude; using GELU introduces smooth nonlinearity.
        mag_act = F.gelu(mag)
        # Compute scaling factor; add a small epsilon to avoid division by zero.
        scale = mag_act / (mag + 1e-6)
        return z * scale

    def forward(self, x):
        """
        Forward pass of the enhanced spectral attention module.

        Args:
          x: Input tensor of shape (B, seq_len, embed_dim)
        Returns:
          Tensor of shape (B, seq_len, embed_dim) with spectral modulation and a residual connection.
        """
        B, N, D = x.shape

        # Pre-normalize input for more stable frequency transformations.
        x_norm = self.pre_norm(x)

        # Reshape to separate heads: (B, num_heads, seq_len, head_dim)
        x_heads = x_norm.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute the FFT along the sequence (token) dimension.
        # Resulting shape: (B, num_heads, freq_bins, head_dim), complex-valued.
        F_fft = torch.fft.rfft(x_heads, dim=2, norm='ortho')

        # Compute adaptive modulation parameters if enabled.
        if self.adaptive:
            # Global context: average over tokens (B, embed_dim)
            context = x_norm.mean(dim=1)
            # Adaptive MLP outputs 2 values per (head, frequency bin).
            adapt_params = self.adaptive_mlp(context)  # (B, num_heads*freq_bins*2)
            adapt_params = adapt_params.view(B, self.num_heads, self.freq_bins, 2)
            # Split into multiplicative and additive modulations.
            adaptive_scale = adapt_params[..., 0:1]  # shape: (B, num_heads, freq_bins, 1)
            adaptive_bias  = adapt_params[..., 1:2]  # shape: (B, num_heads, freq_bins, 1)
        else:
            # If not adaptive, set modulation to neutral (scale=0, bias=0).
            adaptive_scale = torch.zeros(B, self.num_heads, self.freq_bins, 1, device=x.device)
            adaptive_bias  = torch.zeros(B, self.num_heads, self.freq_bins, 1, device=x.device)

        # Combine base parameters with adaptive modulations.
        # effective_filter: scales the frequency response.
        effective_filter = self.base_filter * (1 + adaptive_scale)
        # effective_bias: shifts the frequency response.
        effective_bias = self.base_bias + adaptive_bias

        # Apply adaptive modulation in the frequency domain.
        # Multiplicatively modulate F_fft then add the bias (broadcast along head_dim).
        F_fft_mod = F_fft * effective_filter + effective_bias

        # Apply a nonlinear activation in the frequency domain.
        F_fft_nl = self.complex_activation(F_fft_mod)

        # Inverse FFT to bring the data back to token space.
        # Specify n=self.seq_len to ensure the output length matches the input.
        x_filtered = torch.fft.irfft(F_fft_nl, dim=2, n=self.seq_len, norm='ortho')
        # Reshape: merge heads back into the embedding dimension.
        x_filtered = x_filtered.permute(0, 2, 1, 3).reshape(B, N, D)

        # Residual connection with dropout.
        return x + self.dropout(x_filtered)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1, attention_module=None, drop_path=0.0):
        """
        A generic Transformer encoder block with integrated drop path (stochastic depth).
          - embed_dim: embedding dimension.
          - mlp_ratio: expansion factor for the MLP.
          - dropout: dropout rate.
          - attention_module: a module handling self-attention.
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
        # Apply attention with drop path in the residual connection.
        x = x + self.drop_path(self.attention(x))
        # Apply MLP (after layer normalization) with drop path in the residual connection.
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x
