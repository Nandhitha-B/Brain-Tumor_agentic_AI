"""Figshare brain-tumour dataset (Cheng et al.) -> uint8 array + manifest with PATIENT-LEVEL splits.

Input : the 4 Figshare zips (or the unzipped 1.mat ... 3064.mat) under paths.raw_dir
Output: processed/images.npy  (N,S,S) uint8, processed/manifest.csv, processed/split_summary.json
Labels in the .mat files: 1=meningioma, 2=glioma, 3=pituitary  ->  0,1,2
"""
from __future__ import annotations
import hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold


def read_mat(path: Path):
    """Return (image2d, label_int(1..3), pid_str). Handles v7.3 (HDF5) and older MAT files."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            d = f["cjdata"]
            img = np.array(d["image"]).T            # HDF5 stores MATLAB arrays transposed
            label = int(np.array(d["label"]).squeeze())
            pid = "".join(chr(int(c)) for c in np.array(d["PID"]).squeeze().reshape(-1))
        return img, label, pid
    except (ImportError, OSError):
        from scipy.io import loadmat
        m = loadmat(str(path), struct_as_record=False, squeeze_me=True)["cjdata"]
        return np.asarray(m.image), int(m.label), str(m.PID)


def to_uint8(img: np.ndarray, size: int) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    pil = Image.fromarray((img * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    return np.asarray(pil)


def make_splits(man: pd.DataFrame, n_folds, test_folds, val_folds, monitor_folds, seed) -> pd.Series:
    """Patient-level, label-stratified split into train/val/monitor/test (no patient crosses splits)."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(man), -1)
    for k, (_, te) in enumerate(sgkf.split(man.index, man.label, groups=man.pid)):
        fold[te] = k
    order = list(range(n_folds))
    t, v, m = order[:test_folds], order[test_folds:test_folds + val_folds], \
        order[test_folds + val_folds:test_folds + val_folds + monitor_folds]
    split = np.where(np.isin(fold, t), "test", np.where(np.isin(fold, v), "val",
                     np.where(np.isin(fold, m), "monitor", "train")))
    return pd.Series(split, index=man.index)


def run(cfg: dict, paths) -> None:
    raw, out, size = paths.raw, paths.processed, cfg["data"]["image_size"]
    for z in sorted(raw.rglob("*.zip")):
        dest = raw / "extracted" / z.stem
        if not dest.exists():
            print(f"unzipping {z.name} ...")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
    mats = sorted(raw.rglob("*.mat"), key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9)
    if not mats:
        raise SystemExit(f"No .mat files found under {raw}. Download the Figshare dataset first (see README).")
    imgs, rows = [], []
    for i, p in enumerate(mats):
        img, label, pid = read_mat(p)
        arr = to_uint8(img, size)
        imgs.append(arr)
        rows.append(dict(idx=i, file=p.name, pid=pid, label=label - 1, orig_shape=f"{img.shape[0]}x{img.shape[1]}",
                         md5=hashlib.md5(arr.tobytes()).hexdigest()))
    man = pd.DataFrame(rows)
    assert man.label.isin([0, 1, 2]).all(), "unexpected label values"
    mixed = man.groupby("pid").label.nunique()
    if (mixed > 1).any():
        print(f"WARNING: {(mixed > 1).sum()} patients have more than one label")
    d = cfg["data"]
    man["split"] = make_splits(man, d["n_folds"], d["test_folds"], d["val_folds"], d["monitor_folds"], d["split_seed"])
    # leakage checks
    pid_sets = {s: set(g.pid) for s, g in man.groupby("split")}
    for a in pid_sets:
        for b in pid_sets:
            if a < b:
                assert not (pid_sets[a] & pid_sets[b]), f"patient overlap {a}/{b}"
    dup = man.groupby("md5").split.nunique()
    n_dup = int((dup > 1).sum())
    names = cfg["classes"]
    summary = {"n_images": len(man), "n_patients": int(man.pid.nunique()), "class_names": names,
               "cross_split_exact_duplicate_hashes": n_dup, "splits": {}}
    for s, g in man.groupby("split"):
        summary["splits"][s] = {"images": len(g), "patients": int(g.pid.nunique()),
                                "per_class": {names[k]: int((g.label == k).sum()) for k in range(3)}}
    np.save(out / "images.npy", np.stack(imgs))
    man.to_csv(out / "manifest.csv", index=False)
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    from agentic_rob.config import base_parser, load_config, Paths
    a = base_parser("Prepare Figshare dataset").parse_args()
    c = load_config(a.config, a.override)
    run(c, Paths(c, a.seed))
