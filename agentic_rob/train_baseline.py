"""Stage: ResNet-18 baseline on the train split; best epoch picked on VAL (never on monitor/test)."""
from __future__ import annotations
import copy
import pandas as pd
import torch
import torch.nn.functional as F
from agentic_rob.config import base_parser, load_config, Paths, set_seed, get_device, save_env
from agentic_rob.data.dataset import DataStore, iterate, make_augment
from agentic_rob.models import build_model
from agentic_rob.registry import ModelRegistry


def run(cfg, paths, seed):
    set_seed(seed); dev = get_device()
    save_env(paths.logs / "env.json")
    ds = DataStore(paths.processed, cfg["data"]["max_per_split"], cfg["data"]["split_seed"])
    Xtr, ytr = ds.split("train"); Xva, yva = ds.split("val")
    b = cfg["baseline"]
    model = build_model(cfg, len(cfg["classes"])).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=b["lr"], momentum=b["momentum"], weight_decay=b["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=b["epochs"])
    aug = make_augment() if b["augment"] else None
    best, best_state, hist = -1, None, []
    for ep in range(1, b["epochs"] + 1):
        model.train(); tl, nb = 0.0, 0
        for xb, yb in iterate(Xtr, ytr, b["batch_size"], dev, True, seed * 1000 + ep, aug, drop_last=True):
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); nb += 1
        model.eval(); ok = 0
        with torch.no_grad():
            for xb, yb in iterate(Xva, yva, 64, dev):
                ok += (model(xb).argmax(1) == yb).sum().item()
        acc = 100 * ok / len(yva)
        hist.append(dict(epoch=ep, train_loss=tl / max(nb, 1), val_acc=acc))
        print(f"[baseline] ep{ep} loss={tl / max(nb, 1):.4f} val_acc={acc:.2f}")
        if acc > best:
            best, best_state = acc, copy.deepcopy(model.state_dict())
        sched.step()
    model.load_state_dict(best_state)
    pd.DataFrame(hist).to_csv(paths.logs / "baseline_history.csv", index=False)
    reg = ModelRegistry(paths.registry)
    v = reg.register(model.cpu(), parent=None, kind="baseline", meta=dict(best_val_acc=best, seed=seed))
    reg.promote(v)
    print(f"registered baseline as {v} (val acc {best:.2f})")


if __name__ == "__main__":
    a = base_parser("Train baseline").parse_args()
    c = load_config(a.config, a.override)
    run(c, Paths(c, a.seed), a.seed)
