import torch
import torch.nn as nn
from fftnet_modules import MultiScaleSpectralAttention, TransformerEncoderBlock

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        """
        For CIFAR-10, use img_size=32, patch_size=4 -> 8x8=64 patches + 1 class token = 65 tokens.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # total number of patches
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H=32, W=32)
        x = self.proj(x)  # -> (B, embed_dim, H/patch_size, W/patch_size)
        # Flatten spatial dims and transpose to (B, n_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class FFTNetViT(nn.Module):
    """
    Vision Transformer for CIFAR-10, using multi-scale FFT-based attention.
    """
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=192,
        depth=6,
        mlp_ratio=4.0,
        dropout=0.1,
        num_heads=3,
        adaptive_spectral=True,
        local_window_size=8,
        use_local_branch=True
    ):
        """
        Default config for CIFAR-10:
          - 32×32 images
          - Patch size 4 => 8×8 = 64 patches
          - +1 for class token => sequence length = 65
          - embed_dim=192, num_heads=3 => 64 dims per head
          - depth=6 => fewer blocks for speed
        """
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        n_patches = self.patch_embed.n_patches  # 64

        # Class token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))  # shape (1, 65, D)
        self.pos_drop = nn.Dropout(dropout)

        # Build Transformer encoder blocks
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            spectral_attn = MultiScaleSpectralAttention(
                embed_dim=embed_dim,
                seq_len=n_patches + 1,  # 64 + 1 = 65
                num_heads=num_heads,
                dropout=dropout,
                adaptive=adaptive_spectral,
                local_window_size=local_window_size,
                use_local_branch=use_local_branch,
            )
            block = TransformerEncoderBlock(
                embed_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_module=spectral_attn
            )
            self.blocks.append(block)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        # Initialize embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 3, 32, 32)
        x = self.patch_embed(x)  # (B, 64, embed_dim)
        B = x.shape[0]

        # Class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)          # (B,65,embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer
        for blk in self.blocks:
            x = blk(x)

        # Final norm
        x = self.norm(x)
        cls_out = x[:, 0]  # (B, embed_dim)

        # Classification head
        out = self.head(cls_out)  # (B, num_classes)
        return out
