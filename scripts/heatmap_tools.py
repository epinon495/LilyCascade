"""Shared dataset, U-Net, and point-evaluation code for the anther baseline."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class AntherDataset(Dataset):
    """Return a photo tensor, its Gaussian target, and its manifest index."""

    def __init__(self, root, split, augment=False):
        self.augment = augment and split == "train"
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.image_size = manifest["image_size"]
        if manifest["target_channels"] != ["anther_tip"]:
            raise ValueError("This baseline expects one anther_tip channel.")
        self.records = [r for r in manifest["samples"] if r["split"] == split]
        if not self.records:
            raise ValueError(f"No samples found for {split}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record["image"]) as opened:
            image = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
        target = np.load(self.root / record["heatmap"], allow_pickle=False)
        size = self.image_size
        if image.shape != (size, size, 3) or target.shape != (size, size):
            raise ValueError(f"Invalid image or heatmap shape: {record['id']}")
        if not np.isfinite(target).all() or target.min() < 0 or target.max() > 1:
            raise ValueError(f"Invalid target values: {record['id']}")
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())
        target_tensor = torch.from_numpy(target.astype(np.float32))[None]

        if self.augment:
            # Horizontal flip: transform image and target together.
            if torch.rand(()).item() < 0.5:
                image_tensor = image_tensor.flip(-1)
                target_tensor = target_tensor.flip(-1)

            # Vertical flip.
            if torch.rand(()).item() < 0.5:
                image_tensor = image_tensor.flip(-2)
                target_tensor = target_tensor.flip(-2)

            # Rotate by 0, 90, 180, or 270 degrees.
            turns = torch.randint(0, 4, ()).item()
            image_tensor = torch.rot90(image_tensor, turns, (-2, -1))
            target_tensor = torch.rot90(target_tensor, turns, (-2, -1))

            # Mild contrast and brightness changes affect only the photograph.
            contrast = 0.8 + 0.4 * torch.rand(()).item()
            brightness = 0.8 + 0.4 * torch.rand(()).item()

            mean = image_tensor.mean(dim=(-2, -1), keepdim=True)
            image_tensor = (image_tensor - mean) * contrast + mean
            image_tensor = (image_tensor * brightness).clamp(0, 1)

        return image_tensor, target_tensor, index


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


def heatmap_loss(logits, target, foreground_weight):
    """Weighted MSE: give the sparse Gaussian peaks more influence.

    This weight is a baseline heuristic, not a class-probability correction.
    Plain average pixel error can reward predicting nearly zero everywhere.
    """
    error = (logits.sigmoid() - target).square()
    return (error * (1 + foreground_weight * target)).mean()


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


def match_points(peaks, targets, radius=6):
    """Confidence-ordered greedy, one-to-one matching in 256-pixel crops.

    An unmatched prediction is a false positive; an unmatched tip is a miss.
    Radius is a pixel tolerance, not a physical-distance claim or YOLO IoU.
    """
    available = set(range(len(targets)))
    distances = []
    for x, y, _ in peaks:
        if not available:
            break
        closest = min(available, key=lambda i:
                      (x - targets[i][0]) ** 2 + (y - targets[i][1]) ** 2)
        distance = math.hypot(x - targets[closest][0], y - targets[closest][1])
        if distance <= radius:
            distances.append(distance)
            available.remove(closest)
    tp = len(distances)
    return tp, len(peaks) - tp, len(targets) - tp, distances
