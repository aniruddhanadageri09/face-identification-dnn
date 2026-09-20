import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ArcMarginProduct(nn.Module):
    """ArcFace margin layer."""

    def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.5, easy_margin: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor, label: torch.Tensor):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class FaceIDModel(nn.Module):
    """ResNet-50 face embedding network trained with ArcFace loss."""

    def __init__(self, num_classes: int, embedding_dim: int = 512):
        super().__init__()
        self.backbone = resnet50(weights=None)
        self.backbone.fc = nn.Identity()

        self.embedding_layer = nn.Linear(2048, embedding_dim)
        self.batch_norm = nn.BatchNorm1d(embedding_dim)
        self.arc_margin = ArcMarginProduct(
            in_features=embedding_dim,
            out_features=num_classes,
            s=64.0,
            m=0.5,
            easy_margin=False,
        )

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        features = self.backbone(x)
        embedding = self.embedding_layer(features)
        embedding = self.batch_norm(embedding)
        embedding = F.normalize(embedding, p=2, dim=1)

        if labels is not None:
            logits = self.arc_margin(embedding, labels)
            return embedding, logits

        return embedding
