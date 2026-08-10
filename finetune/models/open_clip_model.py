import torch
import torch.nn as nn
from timm.models.layers import DropPath


class OpenCLIPClassifier(nn.Module):
    """Linear head on top of an open_clip vision tower, fine-tuned end to end."""

    def __init__(self, backbone, num_classes, drop_path_rate=0.2, embed_dim=512):
        super().__init__()
        self.backbone = backbone  # visual encoder
        self.embed_dim = embed_dim

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        # O-MAKE's vision tower returns (pooled_features, patch_embeddings); the
        # classifier only consumes the pooled features.
        if isinstance(features, (tuple, list)):
            features = features[0]
        features = self.drop_path(features)
        logits = self.head(features)
        return logits
