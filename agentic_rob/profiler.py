"""Robustness profiling module (System B). Evaluates a model on a dataset under bounded probes."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from agentic_rob.attacks import make_attack, LINF_ATTACKS
from agentic_rob.data.dataset import iterate
from agentic_rob.metrics import attack_summary, robustness_gap

PRIMARY = ("fgsm", "pgd", "bim", "cw")


def profile(model, X, y, probes, eps_255, device, batch_size=64, seed=0,
            attacks=PRIMARY, gap_attacks=PRIMARY, keep_per_sample=False):
    model.eval().to(device)
    eps = eps_255 / 255
    confs, preds = [], []
    with torch.no_grad():
        for xb, _ in iterate(X, y, batch_size, device):
            c, p = F.softmax(model(xb), 1).max(1)
            confs.append(c.cpu()); preds.append(p.cpu())
    conf_clean, pred_clean = torch.cat(confs).numpy(), torch.cat(preds).numpy()
    a_clean = 100 * float((pred_clean == y).mean())
    res = {"n": int(len(y)), "A_clean": a_clean, "attacks": {}, "per_sample": {}, "clean_correct": pred_clean ==y, "correct": {}}
    if keep_per_sample:
        res["per_sample"] = {"y": y, "pred_clean": pred_clean, "conf_clean": conf_clean}
    for j, name in enumerate(attacks):
        fn = make_attack(name, probes, eps)
        torch.manual_seed(seed * 100 + j)
        P, C, linf, l2max = [], [], 0.0, 0.0
        for xb, yb in iterate(X, y, batch_size, device):
            xa = fn(model, xb, yb)
            d = (xa - xb).flatten(1)
            linf, l2max = max(linf, d.abs().max().item()), max(l2max, d.norm(dim=1).max().item())
            assert xa.min() >= 0 and xa.max() <= 1, "pixel range violated"
            with torch.no_grad():
                c, p = F.softmax(model(xa), 1).max(1)
            P.append(p.cpu()); C.append(c.cpu())
        pa, ca = torch.cat(P).numpy(), torch.cat(C).numpy()
        if name in LINF_ATTACKS:
            assert linf <= eps + 1e-5, f"{name}: L-inf budget violated ({linf} > {eps})"
        if name == "cw":
            assert l2max <= probes["cw"]["l2_eps"] + 1e-3, f"cw: L2 budget violated ({l2max})"
        s = attack_summary(y, pred_clean, conf_clean, pa, ca)
        s.update(linf_max=linf, l2_max=l2max)
        res["attacks"][name] = s
        res["correct"][name]= (pa ==y)
        if keep_per_sample:
            res["per_sample"][f"pred_{name}"], res["per_sample"][f"conf_{name}"] = pa, ca
    accs = {k: res["attacks"][k]["acc"] for k in gap_attacks if k in res["attacks"]}
    if accs:
        res["delta_R_pp"], res["worst_attack"] = robustness_gap(a_clean, accs)
        res["A_worst_case"] = accs[res["worst_attack"]]
    return res
