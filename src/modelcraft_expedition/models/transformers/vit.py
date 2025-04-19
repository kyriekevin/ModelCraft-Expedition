import torch
from torch import nn
from einops import rearrange


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.

    Splits an image into patches and projects each patch into an embedding space.
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: Input images (B, C, H, W)
        Returns:
            Patch embedding (B, N, D)
        """
        _, _, H, W = x.shape
        assert H == W == self.img_size, (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        )

        # (B, C, H, W) -> (B, D, H//P, W//P) -> (B, D, N) -> (B, N, D)
        x = self.proj(x)
        x = rearrange(x, "b d h w -> b (h w) d")

        return x


class Attention(nn.Module):
    """
    Multi-head Self Attention module.
    """

    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, (
            "Embedding dimension must be divisible by number of heads"
        )

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # Combined projection for query, key, value
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: Input tokens (B, N, D)
        Returns:
            Attention output (B, N, D)
        """

        # Project and split heads using einops
        q, k, v = rearrange(
            self.qkv(x),
            "b n (qkv h d) -> qkv b h n d",
            qkv=3,
            h=self.num_heads,
            d=self.head_dim,
        ).unbind(0)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention and combine heads
        x = attn @ v
        x = rearrange(x, "b h n d -> b n (h d)")

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """
    MLP module with two linear layers.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x: Input features (B, N, D)
        Returns:
            Transformed features (B, N, D)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class Block(nn.Module):
    """
    Transformer encoder block.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)

        # Multi-head attention
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer normalization
        self.norm2 = nn.LayerNorm(dim)

        # MLP block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        """
        Args:
            x: Input tokens (B, N, D)
        Returns:
            Output tokens (B, N, D)
        """
        # Attention block with residual connection and drop path
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # MLP block with residual connection and drop path
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT) model.

    Implementation based on the paper:
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.n_patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize model weights.
        """
        # Initialize patch embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

        # Apply to all modules

        self.apply(self._init_weights_recursive)

    def _init_weights_recursive(self, m):
        """
        Initialize weights recursively for all layers.
        """
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x):
        """
        Extract features from input images.

        Args:
            x: Input images (B, C, H, W)
        Returns:
            Class token features (B, D)
        """
        B = x.shape[0]

        # Extract patch embeddings
        x = self.patch_embed(x)  # (B, N, D)

        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_token, x), dim=1)  # (B, N+1, D)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        x = self.blocks(x)

        # Apply layer normalization
        x = self.norm(x)

        # Return class token
        return x[:, 0]

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input images (B, C, H, W)
        Returns:
            Class predictions (B, num_classes)
        """
        # Extract features
        x = self.forward_features(x)

        # Classification head
        x = self.head(x)

        return x
