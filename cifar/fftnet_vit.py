import torch
import torch.nn as nn
from fftnet_modules import MultiHeadSpectralAttention, TransformerEncoderBlock
###############################################################################
# 4) PatchEmbed: 2D image -> (B, num_patches, embed_dim)
###############################################################################
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # e.g. 8 for 32/4
        self.num_patches = self.grid_size * self.grid_size
        # Conv2d with stride=patch_size => effectively patchify
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, in_channels=3, H=32, W=32)
        x = self.proj(x)            # (B, embed_dim, grid_size, grid_size)
        x = x.flatten(2)            # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)       # (B, num_patches, embed_dim)
        return x

###############################################################################
# 5) FFTNetsViT: Full model for CIFAR-10
###############################################################################
class FFTNetViT(nn.Module):
    """
    A ViT-like model that uses FFT-based spectral attention.
    Defaults are set for CIFAR-10 (32x32 images).
    """
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=192,
        depth=6,
        mlp_ratio=4.0,
        dropout=0.1,
        num_heads=3,
        adaptive_spectral=True,
        window_size=16,
        hop_size=8
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches  # e.g. 64 for CIFAR-10
        self.seq_len = num_patches + 1              # +1 for CLS token => 65
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Build transformer blocks with spectral attention
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            attn_mod = MultiHeadSpectralAttention(
                embed_dim=embed_dim,
                seq_len=self.seq_len,
                num_heads=num_heads,
                dropout=dropout,
                adaptive=adaptive_spectral,
                window_size=window_size,
                hop_size=hop_size
            )
            block = TransformerEncoderBlock(embed_dim, mlp_ratio, dropout, attention_module=attn_mod)
            self.blocks.append(block)

        # Final norm & head
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
        # x: (B, 3, 32, 32)
        B = x.shape[0]
        x = self.patch_embed(x)                # (B, 64, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 65, embed_dim)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # (B, embed_dim)
        logits = self.head(cls_out)  # (B, num_classes)
        return logits

