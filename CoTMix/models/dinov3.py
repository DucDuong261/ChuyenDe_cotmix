"""DINOv3 backbone adapter for CoTMix time-series experiments.

CoTMix supplies tensors shaped [batch, channels, time]. DINOv3 expects RGB
images. This module converts each sensor window to a 3-channel Gramian Angular
Field (GAF) image, then uses the DINOv3 CLS token as the feature embedding.
Temporal Mixup remains in the original time-series space because conversion
happens inside forward(), after CoTMix has created the mixed windows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DINOv3Encoder(nn.Module):
    """Backbone = Encoder for CoTMix.

    Input:  [B, C, T] raw UCI-HAR windows (normally C=9, T=128).
    Output: [B, 384] CLS embeddings from DINOv3 ViT-S/16.
    """

    def __init__(self, configs=None):
        super().__init__()
        model_name = getattr(
            configs,
            "dinov3_model_name",
            "facebook/dinov3-vits16-pretrain-lvd1689m",
        )
        self.image_size = int(getattr(configs, "dinov3_image_size", 224))
        self.freeze_backbone = bool(getattr(configs, "freeze_dinov3", True))
        self.backbone = AutoModel.from_pretrained(model_name)
        self.feature_dim = int(getattr(self.backbone.config, "hidden_size", 384))

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    @staticmethod
    def _gaf_per_channel(x):
        """Create a GAF per sensor channel. x: [B, C, T] -> [B, C, T, T]."""
        # Normalize each channel/window to [-1, 1], avoiding division by zero.
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        x = 2.0 * (x - x_min) / (x_max - x_min).clamp_min(1e-6) - 1.0
        x = x.clamp(-1.0, 1.0)
        sin_phi = torch.sqrt((1.0 - x.square()).clamp_min(1e-6))
        # cos(phi_i + phi_j) = cos(phi_i)cos(phi_j) - sin(phi_i)sin(phi_j)
        return (
            x.unsqueeze(-1) * x.unsqueeze(-2)
            - sin_phi.unsqueeze(-1) * sin_phi.unsqueeze(-2)
        )

    def time_series_to_rgb(self, x):
        """Map sensor channels into 3 GAF image channels and resize for DINOv3.

        UCI-HAR has 9 channels. They are split into three groups of three;
        each RGB plane is the mean GAF of one group. This is a practical adapter
        for an experiment, not a representation defined by the original paper.
        """
        if x.ndim != 3:
            raise ValueError(
                f"DINOv3Encoder expects [B, C, T] time-series input; got {tuple(x.shape)}"
            )
        gaf = self._gaf_per_channel(x.float())  # [B, C, T, T]
        channel_groups = torch.tensor_split(gaf, 3, dim=1)
        rgb = []
        for group in channel_groups:
            if group.shape[1] == 0:
                # This branch supports datasets with fewer than 3 sensor channels.
                group = gaf.mean(dim=1, keepdim=True)
            rgb.append(group.mean(dim=1))
        image = torch.stack(rgb, dim=1)  # [B, 3, T, T]
        image = F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        # Map GAF's [-1, 1] range to [0, 1] before pretrained normalization.
        image = (image + 1.0) / 2.0
        # DINOv3 pretrained image normalization (ImageNet-style).
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (image - mean) / std

    def forward(self, x):
        images = self.time_series_to_rgb(x)
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=images)
        else:
            outputs = self.backbone(pixel_values=images)
        return outputs.last_hidden_state[:, 0, :]
