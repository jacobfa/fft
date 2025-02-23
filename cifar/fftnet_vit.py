# fftnet_vit.py
import torch
import torch.nn as nn
from fftnet_modules import MultiHeadSpectralAttention, TransformerEncoderBlock

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """
        Splits the image into patches and projects them into an embedding space.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class FFTNetViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, mlp_ratio=4.0, dropout=0.1,
                 num_heads=12, adaptive_spectral=True):
        """
        Vision Transformer with FFTNet (spectral) attention.
        Hyperparameters are set for ImageNet.
        """
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.n_patches

        # Learnable class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Build Transformer encoder blocks with spectral attention
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            spectral_attn = MultiHeadSpectralAttention(
                embed_dim, seq_len=n_patches + 1,
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
        # x: (B, C, H, W)
        x = self.patch_embed(x)
        B = x.shape[0]
        # Expand the class token to the batch size and concatenate with patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        out = self.head(cls_out)
        return out
