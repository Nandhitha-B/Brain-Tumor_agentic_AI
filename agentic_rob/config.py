"""Config loading, seeding, device and path helpers (no torch import at module load)."""
from __future__ import annotations
import platform, random, sys, json
from pathlib import Path
import numpy as np
import yaml


def deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str, override: str | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if override:
        with open(override) as f:
            deep_update(cfg, yaml.safe_load(f))
    return cfg


def set_seed(seed: int, deterministic: bool = True) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(pref: str = "auto"):
    import torch
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


class Paths:
    """All file locations for one training seed. Attributes may be overridden (deploy demo)."""

    def __init__(self, cfg: dict, seed: int, root: str = "."):
        root = Path(root)
        self.raw = root / cfg["paths"]["raw_dir"]
        self.processed = root / cfg["paths"]["processed_dir"]
        self.run = root / cfg["paths"]["runs_dir"] / f"seed{seed}"
        self.registry = self.run / "registry"
        self.logs = self.run / "logs"
        self.results = self.run / "results"
        self.agent_log = self.logs / "agent_events.jsonl"
        for p in (self.processed, self.registry, self.logs, self.results):
            p.mkdir(parents=True, exist_ok=True)


def save_env(path: Path) -> None:
    import torch, torchvision
    info = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__}
    Path(path).write_text(json.dumps(info, indent=2))


def base_parser(desc: str):
    import argparse
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", default=None, help="e.g. configs/quick.yaml")
    ap.add_argument("--seed", type=int, default=0)
    return ap
