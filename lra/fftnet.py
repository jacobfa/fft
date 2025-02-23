import torch
import torch.nn as nn
from fftnet_modules import MultiHeadSpectralAttention, TransformerEncoderBlock

class TokenEmbed(nn.Module):
    def __init__(self, input_dim, embed_dim):
        """
        Projects an input sequence of features (B, seq_len, input_dim) 
        into a higher-dimensional embedding space (B, seq_len, embed_dim).
        """
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        return self.proj(x)

class FFTNetLRA(nn.Module):
    def __init__(self, seq_len, input_dim, num_classes,
                 embed_dim=768, depth=12, mlp_ratio=4.0, dropout=0.1,
                 num_heads=12, adaptive_spectral=True):
        """
        FFTNet-based Transformer for the Long Range Arena benchmark.
        
        Args:
            seq_len (int): The length of the input sequence.
            input_dim (int): The dimensionality of input features.
            num_classes (int): Number of target classes.
            embed_dim (int): Embedding dimension.
            depth (int): Number of transformer encoder blocks.
            mlp_ratio (float): Expansion ratio for the MLP inside encoder blocks.
            dropout (float): Dropout rate.
            num_heads (int): Number of attention heads.
            adaptive_spectral (bool): Whether to use adaptive spectral attention.
        """
        super().__init__()
        self.token_embed = TokenEmbed(input_dim, embed_dim)
        n_tokens = seq_len

        # Learnable class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Build Transformer encoder blocks with spectral (FFT-based) attention
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            spectral_attn = MultiHeadSpectralAttention(
                embed_dim, seq_len=n_tokens + 1,
                num_heads=num_heads, dropout=dropout, adaptive=adaptive_spectral
            )
            block = TransformerEncoderBlock(embed_dim, mlp_ratio, dropout, attention_module=spectral_attn)
            self.blocks.append(block)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        x = self.token_embed(x)
        B = x.shape[0]
        # Expand the class token to the batch size and concatenate with token embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_out = x[:, 0]  # Use the class token representation for classification
        out = self.head(cls_out)
        return out
