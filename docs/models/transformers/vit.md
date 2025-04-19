# Vision Transformer (ViT)

Paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

The Vision Transformer (ViT) applies the transformer architecture, originally designed for natural language processing, to image classification tasks. Instead of using convolutional layers, ViT splits an image into fixed-size patches, linearly embeds each patch, adds position embeddings, and feeds the resulting sequence of vectors to a standard transformer encoder.

## Model Architecture

The ViT architecture consists of the following components:

1. Patch Embedding: Divides the input image into fixed-size patches and projects them into an embedding space
2. Position Embedding: Adds learnable position embeddings to provide spatial information
3. Transformer Encoder: Processes the sequence of patch embeddings through multiple layers of self-attention and feed-forward networks
4. Classification Head: A simple MLP that takes the transformer output corresponding to a special [CLS] token and produces class predictions

## Implementation Parameters

| Parameter        | Description                              | Default Value | Notes                                      |
| ---------------- | ---------------------------------------- | ------------- | ------------------------------------------ |
| `img_size`       | Input image size                         | 224           | Must be divisible by `patch_size`          |
| `patch_size`     | Size of image patches                    | 16            | Determines the sequence length             |
| `in_channels`    | Number of input channels                 | 3             | 3 for RGB images, 1 for grayscale          |
| `num_classes`    | Number of output classes                 | 1000          | For classification tasks                   |
| `embed_dim`      | Embedding dimension                      | 768           | Dimension of patch and position embeddings |
| `depth`          | Number of transformer layers             | 12            | More layers = higher capacity              |
| `num_heads`      | Number of attention heads                | 12            | Should divide `embed_dim` evenly           |
| `mlp_ratio`      | Ratio of MLP hidden dim to embedding dim | 4.0           | Controls capacity of feed-forward network  |
| `qkv_bias`       | Whether to use bias in QKV projections   | True          | Improves training stability                |
| `drop_rate`      | Dropout rate                             | 0.0           | For regularization                         |
| `attn_drop_rate` | Attention dropout rate                   | 0.0           | For regularization of attention maps       |
| `drop_path_rate` | Stochastic depth rate                    | 0.0           | For regularization via path dropping       |

## Mathematical Formulation

The ViT model processes images as follows:

1. Patch Embedding:

   - For an image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$, extract $N = HW/P^2$ patches of size $P \times P$
   - Flatten each patch and project it to dimension $D$: $\mathbf{z}_0 = [\mathbf{x}_p^1\mathbf{E}; \mathbf{x}_p^2\mathbf{E}; \cdots; \mathbf{x}p^N\mathbf{E}] + \mathbf{E}{pos}$
   - Where $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ is the patch embedding projection and $\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$ are the position embeddings

2. Transformer Encoder:

   - For each layer $l = 1...L$:
     - $\mathbf{z}'l = \text{MSA}(\text{LN}(\mathbf{z}{l-1})) + \mathbf{z}_{l-1}$
     - $\mathbf{z}_l = \text{MLP}(\text{LN}(\mathbf{z}'_l)) + \mathbf{z}'_l$
   - Where MSA is multi-head self-attention, LN is layer normalization, and MLP is a feed-forward network

3. Classification Head:

   - $\mathbf{y} = \text{LN}(\mathbf{z}_L^0)$ where $\mathbf{z}_L^0$ is the [CLS] token representation from the final layer

## Usage Example

```python
import torch
from modelcraft_expedition.models.transformers.vit import ViT

# Initialize model
model = ViT(
    img_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12
)

# Generate a random input tensor
x = torch.randn(1, 3, 224, 224)

# Forward pass
output = model(x)  # Shape: [1, 1000]
```

## Model Variants

The original paper proposed several ViT variants:

| Variant   | Layers | Hidden Size | MLP Size | Heads | Params |
| --------- | ------ | ----------- | -------- | ----- | ------ |
| ViT-Base  | 12     | 768         | 3072     | 12    | 86M    |
| ViT-Large | 24     | 1024        | 4096     | 16    | 307M   |
| ViT-Huge  | 32     | 1280        | 5120     | 16    | 632M   |

Our implementation supports all these variants through parameter configuration.

## Performance Considerations

- ViT typically requires more data than CNNs to achieve good performance when trained from scratch
- Pre-training on large datasets is recommended for most applications
- The model's computational complexity scales quadratically with the number of patches
