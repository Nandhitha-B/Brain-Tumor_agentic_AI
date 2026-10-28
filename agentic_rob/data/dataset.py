"""In-memory dataset access. Images are stored uint8 (N,S,S); converted to [0,1] 3-channel on the fly."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


class DataStore:
    def __init__(self, processed_dir, max_per_split=None, seed=0):
        d = Path(processed_dir)
        self.images = np.load(d / "images.npy", mmap_mode="r")
        self.man = pd.read_csv(d / "manifest.csv")
        self.max_per_split, self.seed = max_per_split, seed

    def split(self, name: str):
        idx = self.man.index[self.man.split == name].values
        if self.max_per_split and len(idx) > self.max_per_split:
            idx = np.sort(np.random.RandomState(self.seed).choice(idx, self.max_per_split, replace=False))
        return np.ascontiguousarray(self.images[idx]), self.man.label.values[idx].astype(np.int64)


def make_augment():
    from torchvision.transforms import v2
    return v2.Compose([v2.RandomHorizontalFlip(),
                       v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))])


def iterate(X, y, batch_size, device, shuffle=False, seed=0, augment=None, drop_last=False):
    """Yield (x in [0,1] of shape B,3,H,W ; y) batches."""
    import torch
    n = len(X)
    order = np.random.RandomState(seed).permutation(n) if shuffle else np.arange(n)
    for s in range(0, n, batch_size):
        ib = order[s:s + batch_size]
        if drop_last and len(ib) < batch_size:
            break
        xb = torch.from_numpy(X[ib]).to(device).float().div_(255).unsqueeze(1).repeat(1, 3, 1, 1)
        yb = torch.from_numpy(y[ib]).long().to(device)
        if augment is not None:
            xb = torch.stack([augment(xi) for xi in xb]).clamp_(0, 1)
        yield xb, yb
