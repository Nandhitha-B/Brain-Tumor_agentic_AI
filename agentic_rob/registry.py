"""Versioned model registry (checkpoints + sha256 + lineage). System A reads `current`; System B writes."""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import torch


class ModelRegistry:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.file = self.root / "registry.json"
        if not self.file.exists():
            self._write({"current": None, "versions": {}})

    def _read(self) -> dict:
        return json.loads(self.file.read_text())

    def _write(self, d: dict) -> None:
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=2))
        os.replace(tmp, self.file)        # atomic: readers never see a half-written registry

    @property
    def current(self):
        return self._read()["current"]

    def versions(self) -> dict:
        return self._read()["versions"]

    def register(self, model, parent=None, kind="candidate", meta=None) -> str:
        d = self._read()
        v = f"v{len(d['versions']) + 1:03d}"
        path = self.root / "checkpoints" / f"resnet18_braintumor_{v}.pth"
        torch.save(model.state_dict(), path)
        d["versions"][v] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                parent=parent, kind=kind, status="registered",
                                created=time.strftime("%Y-%m-%dT%H:%M:%S"), meta=meta or {})
        self._write(d)
        return v

    def set_status(self, v, status):
        d = self._read(); d["versions"][v]["status"] = status; self._write(d)

    def promote(self, v):
        d = self._read(); d["current"] = v; d["versions"][v]["status"] = "promoted"; self._write(d)

    def load_into(self, model, v):
        model.load_state_dict(torch.load(self.versions()[v]["path"], map_location="cpu"))
        return model

    def first_of_kind(self, kind):
        for v, info in self.versions().items():
            if info["kind"] == kind:
                return v
        return None
