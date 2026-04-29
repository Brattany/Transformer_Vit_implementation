import torch
import torch.nn as nn
from .transformer import TransformerEncoder

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (Batch_size, embed_dim, grid_size, grid_size)
        x = x.flatten(2)  # (Batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (Batch_size, num_patches, embed_dim)
        return x

class CLS_Token(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (Batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (Batch_size, num_patches + 1, embed_dim)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        return x + self.pos_embedding

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        cls_token = x[:, 0]  # (B, embed_dim)
        return self.fc(cls_token)


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_classes=1000, depth=12, num_heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        #vit图像输入部分
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = CLS_Token(embed_dim)
        self.pos_embedding = PositionalEmbedding(self.patch_embedding.num_patches, embed_dim)
        #复用已有TransformerEncoder模块
        self.encoder = TransformerEncoder(
            N = depth,
            d_model = embed_dim,
            h = num_heads,
            d_ff = mlp_dim,
            dropout = dropout
        )
        self.classification_head = ClassificationHead(embed_dim, num_classes)

    def forward(self, x, return_attention=False):
        x = self.patch_embedding(x)
        x = self.cls_token(x)
        x = self.pos_embedding(x)
        x, attention_weights = self.encoder(x)
        x = self.classification_head(x)
        if return_attention:
            return x, attention_weights
        return x
