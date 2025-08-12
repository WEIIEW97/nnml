import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        """
        fn: represent one of `MSA(multihead self attention)` or `MLP`
        """
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(FFN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MSA(nn.Module):
    """Implementation of Multihead self attention."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super(MSA, self).__init__()
        inner_dim = heads * dim_head
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # *3 means [Q, K, V]

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """
        b: batch size
        n: sequence length(number of tokens)
        h: number of attention heads
        d: dimensionality of each head
        Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})*V
        """
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # (b, n, dim*3) --> 3*(b, n, dim)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv
        )  # q, k, v (b, h, n, dim)

        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = self.attend(dots)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, num_layers, heads, dim_head, mlp_dim, dropout=0.1):
        """A standard transformer model with multi-head self-attention
        and feed-forward layers.

        Args:
            dim (int): The dimensionality of the input embeddings.
            num_layers (int): The number of transformer layers.
            heads (int): The number of attention heads.
            dim_head (int): The dimensionality of each attention head.
            mlp_dim (int): The dimensionality of the hidden layer in the feed-forward network.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            MSA(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        ),
                        PreNorm(dim, FFN(dim, hidden_dim=mlp_dim, dropout=dropout)),
                    ]
                )
            )
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for all layers."""
        for layer in self.layers:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass for the transformer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim).
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim,
        num_layers,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1,
    ):
        super(ViT, self).__init__()
        image_h, image_w = pair(image_size)  # usually will reshape to a square size
        patch_h, patch_w = pair(patch_size)  # as above

        assert (
            image_h % patch_h == 0 and image_w % patch_w == 0
        ), "patch h, w must be divisible by image h, w"

        num_patches = (image_h // patch_h) * (image_w // patch_w)
        patch_dim = channels * patch_h * patch_w

        assert pool in ("cls", "mean")

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_h, p2=patch_w),
            nn.Linear(patch_dim, dim),
        )

        # self.pos_embedding = nn.Embedding(
        #     num_patches, embedding_dim=dim
        # )  # emb shape (p, p, vocab_dim), is a learnable parameter (typically a tensor) that stores the positional embeddings.
        # self.cls_token = nn.Embedding(1, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=dim,
            num_layers=num_layers,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.pool = pool
        self.to_latent = (
            nn.Identity()
        )  # In machine learning and deep learning, the term "latent" refers to a hidden or abstract representation of the data. For example:
        # In autoencoders, the latent space is a compressed representation of the input data.
        # In transformers, the latent representation might refer to the intermediate embeddings produced by the model.
        # The idea is that the latent representation captures the essential features of the input data in a way that is useful for downstream tasks (e.g., classification, generation, etc.).

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        x = self.to_patch_embedding(
            x
        )  # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim
        b, n, _ = x.shape  # b: batch size, n: patch resolution

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat(
            (cls_tokens, x), dim=1
        )  # x shape: (B, n+1, dim), concat the labels
        x += self.pos_embedding[
            :, : (n + 1)
        ]  # provide the model with information about the order of tokens in a sequence
        x = self.dropout(x)

        x = self.transformer(x)  # (b, n+1, dim)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]  # (b, dim)
        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == "__main__":
    devide = "cuda" if torch.cuda.is_available() else "cpu"
    model_vit = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        num_layers=8,
        heads=16,
        mlp_dim=2048,
    ).to(devide)

    img = torch.randn(16, 3, 256, 256).to(devide)
    preds = model_vit(img)
    print(preds.shape)
    print(preds)
