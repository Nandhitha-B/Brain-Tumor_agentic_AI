"""System A: serves predictions from the registry's CURRENT version. Performs no attacks and no training.
It only reloads when the registry points to a new version (scheduled update window)."""
from __future__ import annotations
import threading
import torch
import torch.nn.functional as F
from agentic_rob.models import build_model, Smoothed
from agentic_rob.registry import ModelRegistry


class InferenceService:
    def __init__(self, cfg, registry_dir, device="cpu"):
        self.cfg, self.reg, self.dev = cfg, ModelRegistry(registry_dir), torch.device(device)
        self.smoothing = cfg["inference"]["smoothing"]
        self._lock = threading.Lock()
        self.version, self.model = None, None
        self.maybe_reload()

    def _build(self, v):
        m = self.reg.load_into(build_model(self.cfg, len(self.cfg["classes"])), v)
        m = Smoothed(m) if self.smoothing else m
        return m.to(self.dev).eval()

    def maybe_reload(self) -> bool:
        v = self.reg.current
        if v is None or v == self.version:
            return False
        new = self._build(v)
        with self._lock:                      # atomic swap, no downtime
            self.model, self.version = new, v
        return True

    @torch.no_grad()
    def predict(self, x):
        """x: (B,3,H,W) in [0,1] -> (class_idx, confidence, model_version)."""
        with self._lock:
            m, v = self.model, self.version
        conf, cls = F.softmax(m(x.to(self.dev)), 1).max(1)
        return cls.cpu(), conf.cpu(), v
