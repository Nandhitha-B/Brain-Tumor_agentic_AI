"""Distribution-shift robustness: natural MRI-style corruptions at 3 severities (clean images, no attacker)."""
from __future__ import annotations
import math
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from agentic_rob.config import base_parser, load_config, Paths, set_seed, get_device
from agentic_rob.data.dataset import DataStore, iterate
from agentic_rob.evaluate import targets_for
from agentic_rob.models import build_model
from agentic_rob.registry import ModelRegistry


def noise(x, s): return (x + s * torch.randn_like(x)).clamp(0, 1)
def blur(x, s):
    k = 2 * math.ceil(3 * s) + 1
    return TF.gaussian_blur(x, [k, k], [s, s])
def contrast(x, c):
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - m) * c + m).clamp(0, 1)
def bias(x, s):
    f = F.interpolate(torch.rand(x.shape[0], 1, 4, 4, device=x.device) * 2 - 1, size=x.shape[-2:], mode="bicubic", align_corners=False)
    return (x * (1 + s * f)).clamp(0, 1)
def rotate(x, a): return TF.rotate(x, a if torch.rand(1) < 0.5 else -a)

CORR = {"clean": ([0], lambda x, s: x), "gaussian_noise": ([0.02, 0.05, 0.10], noise),
        "gaussian_blur": ([1.0, 2.0, 3.0], blur), "contrast": ([0.8, 0.6, 0.4], contrast),
        "bias_field": ([0.1, 0.2, 0.3], bias), "rotation": ([5.0, 10.0, 15.0], rotate)}


def run(cfg, paths, seed):
    set_seed(seed); dev = get_device()
    reg = ModelRegistry(paths.registry)
    X, y = DataStore(paths.processed, cfg["data"]["max_per_split"], cfg["data"]["split_seed"]).split("test")
    rows = []
    for name, v in targets_for(reg).items():
        if v is None: continue
        model = reg.load_into(build_model(cfg, len(cfg["classes"])), v).to(dev).eval()
        for cond, (sevs, fn) in CORR.items():
            for s in sevs:
                torch.manual_seed(seed); ok = 0
                for xb, yb in iterate(X, y, cfg["eval"]["batch_size"], dev):
                    with torch.no_grad():
                        ok += (model(fn(xb, s)).argmax(1) == yb).sum().item()
                rows.append(dict(model=name, corruption=cond, severity=s, acc=100 * ok / len(y)))
    df = pd.DataFrame(rows); df.to_csv(paths.results / "shift_results.csv", index=False)
    print(df.groupby(["model", "corruption"]).acc.mean().unstack().round(2).to_markdown())


if __name__ == "__main__":
    a = base_parser("Distribution-shift evaluation").parse_args()
    c = load_config(a.config, a.override)
    run(c, Paths(c, a.seed), a.seed)