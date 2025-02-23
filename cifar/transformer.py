import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ---------------------------
# Patch Embedding Module
# ---------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        # A convolution with stride=patch_size acts as a patch embedder.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # shape: (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2)  # shape: (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # shape: (B, num_patches, embed_dim)
        return x

# ---------------------------
# MLP Module for Transformer Block
# ---------------------------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# ---------------------------
# Transformer Encoder Block
# ---------------------------
class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=3.0, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        # Using batch_first=True so the input shape is (B, N, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_dim, dropout=dropout)

    def forward(self, x):
        # Self-attention sub-layer
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP sub-layer
        x = x + self.mlp(self.norm2(x))
        return x

# ---------------------------
# Vision Transformer Model
# ---------------------------
class VisionTransformer(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=192, depth=6, mlp_ratio=3.0, num_heads=6, dropout=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Create transformer encoder blocks
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_layer)

    def _init_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        # Prepare class token and add to the patch embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embed_dim)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.blocks(x)
        x = self.norm(x)
        cls_x = x[:, 0]  # use the class token for classification
        out = self.head(cls_x)
        return out
