"""Synthetic Figshare-like files -> prepare pipeline -> no patient leakage, correct label mapping."""
import numpy as np, pandas as pd
from scipy.io import savemat
from agentic_rob.data import prepare_figshare as pf
from agentic_rob.config import Paths


def test_prepare_on_synthetic(tmp_path):
    cfg = {"paths": {"raw_dir": str(tmp_path / "raw"), "processed_dir": str(tmp_path / "proc"), "runs_dir": str(tmp_path / "runs")},
           "data": {"image_size": 32, "split_seed": 1, "n_folds": 10, "test_folds": 2, "val_folds": 1, "monitor_folds": 1},
           "classes": ["meningioma", "glioma", "pituitary"]}
    raw = tmp_path / "raw"; raw.mkdir()
    rng = np.random.RandomState(0); n = 0
    for pid in range(60):
        label = pid % 3 + 1
        for _ in range(3):
            n += 1
            savemat(raw / f"{n}.mat", {"cjdata": {"label": label, "PID": f"P{pid}", "image": rng.randint(0, 4000, (64, 64)).astype(np.int16)}})
    pf.run(cfg, Paths(cfg, 0, root="."))
    man = pd.read_csv(tmp_path / "proc" / "manifest.csv")
    assert len(man) == n and set(man.label) == {0, 1, 2}
    assert set(man.split) == {"train", "val", "monitor", "test"}
    for a in ("train", "val", "monitor", "test"):
        for b in ("train", "val", "monitor", "test"):
            if a < b:
                assert not set(man[man.split == a].pid) & set(man[man.split == b].pid)
