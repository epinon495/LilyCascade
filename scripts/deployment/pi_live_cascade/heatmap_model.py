"""Exact inference architecture and peak extraction from the training project."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def conv_block(in_channels, out_channels):
    # GroupNorm works with small batches and has no running batch statistics.
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(4, out_channels), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(4, out_channels), nn.ReLU(),
    )

class TinyUNet(nn.Module):
    """Three downsampling stages; skip connections retain spatial detail.

    Input: B x 3 x H x W. Output: B x 1 x H x W raw logits.
    Apply sigmoid to obtain heatmap values in [0, 1]. No pretrained weights.
    """

    def __init__(self, width=16):
        super().__init__()
        if width % 4:
            raise ValueError("Width must be divisible by four for GroupNorm.")
        self.enc1 = conv_block(3, width)
        self.enc2 = conv_block(width, width * 2)
        self.enc3 = conv_block(width * 2, width * 4)
        self.bottom = conv_block(width * 4, width * 8)
        self.dec3 = conv_block(width * 12, width * 4)
        self.dec2 = conv_block(width * 6, width * 2)
        self.dec1 = conv_block(width * 3, width)
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.constant_(self.head.bias, -2.0)

    def forward(self, image):
        a = self.enc1(image)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.enc3(F.max_pool2d(b, 2))
        z = self.bottom(F.max_pool2d(c, 2))
        z = self.dec3(torch.cat((self.up(z, c), c), dim=1))
        z = self.dec2(torch.cat((self.up(z, b), b), dim=1))
        z = self.dec1(torch.cat((self.up(z, a), a), dim=1))
        return self.head(z)

    @staticmethod
    def up(tensor, skip):
        return F.interpolate(tensor, size=skip.shape[-2:], mode="bilinear",
                             align_corners=False)

def find_peaks(heatmap, threshold=0.3, min_distance=4):
    """Local maxima, highest score first; suppress neighbors within 4 pixels."""
    tensor = torch.from_numpy(np.asarray(heatmap, dtype=np.float32)).view(
        1, 1, *heatmap.shape)
    maxima = F.max_pool2d(tensor, 3, stride=1, padding=1)[0, 0].numpy()
    yy, xx = np.where((heatmap >= threshold) & (heatmap == maxima))
    order = np.argsort(-heatmap[yy, xx], kind="stable")
    peaks = []
    for i in order:
        x, y = int(xx[i]), int(yy[i])
        if all((x - px) ** 2 + (y - py) ** 2 > min_distance ** 2
               for px, py, _ in peaks):
            peaks.append((x, y, float(heatmap[y, x])))
    return peaks
