"""Frozen DINOv2 CLS features. Input: BCHW RGB uint8 or float [0,1]."""
import torch
from torch import nn
from torch.nn import functional as F

class FrozenVisionEncoder(nn.Module):
    def __init__(self, name='dinov2_vits14', image_size=224, backbone=None):
        super().__init__()
        self.backbone = backbone if backbone is not None else torch.hub.load(
            'facebookresearch/dinov2', name, pretrained=True, trust_repo=True)
        self.backbone.requires_grad_(False).eval()
        self.image_size = image_size
        self.register_buffer('mean', torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std', torch.tensor([.229,.224,.225])[None,:,None,None])

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError('Expected BCHW RGB')
        x = images.float() / (255 if images.dtype == torch.uint8 else 1)
        if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1.001:
            raise ValueError('RGB must be uint8 or float in [0,1]')
        x = F.interpolate(x, (self.image_size,self.image_size), mode='bilinear', align_corners=False, antialias=True)
        return self.backbone.forward_features((x-self.mean)/self.std)['x_norm_clstoken']
