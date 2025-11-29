# clip_lab4/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImageEncoder(nn.Module):
    """
    ResNet50 backbone + projection head to 512-D CLIP space.
    """

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove classifier
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # (B, 2048, 1, 1)
        self.proj = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, embed_dim),
        )

    def forward(self, x):
        feats = self.backbone(x)        # (B, 2048, 1, 1)
        feats = feats.flatten(1)        # (B, 2048)
        emb = self.proj(feats)          # (B, 512)
        emb = F.normalize(emb, dim=-1)  # unit-norm
        return emb


class ClipLoss(nn.Module):
    """
    Symmetric InfoNCE loss for CLIP-style training.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / temperature))

    def forward(self, image_embeds, text_embeds):
        # Normalize (should already be normalized but just in case)
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        logits_per_image = self.logit_scale.exp() * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()

        targets = torch.arange(image_embeds.size(0), device=image_embeds.device)
        loss_i = F.cross_entropy(logits_per_image, targets)
        loss_t = F.cross_entropy(logits_per_text, targets)
        return (loss_i + loss_t) / 2
