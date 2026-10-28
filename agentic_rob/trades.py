"""TRADES / PGD-AT (Madry) fine-tuning with eps+beta warm-up and cosine LR."""
from __future__ import annotations
import time
import torch
import torch.nn.functional as F
from agentic_rob.data.dataset import iterate, make_augment


def trades_loss(model, x, y, eps, alpha, steps, beta):
    kl = torch.nn.KLDivLoss(reduction="sum")
    model.eval()
    with torch.no_grad():
        p_nat = F.softmax(model(x), 1)
    xa = (x + 0.001 * torch.randn_like(x)).clamp(0, 1).detach()
    for _ in range(steps):
        xa.requires_grad_(True)
        g = torch.autograd.grad(kl(F.log_softmax(model(xa), 1), p_nat), xa)[0]
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x - eps), x + eps).clamp(0, 1).detach()
    model.train()
    logits, logits_adv = model(x), model(xa)
    ce = F.cross_entropy(logits, y)
    rob = kl(F.log_softmax(logits_adv, 1), F.softmax(logits, 1)) / x.shape[0]
    return ce + beta * rob, ce.item(), rob.item()


def madry_loss(model, x, y, eps, alpha, steps):
    model.eval()
    xa = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True)
        g = torch.autograd.grad(F.cross_entropy(model(xa), y), xa)[0]
        xa = xa.detach() + alpha * g.sign()
        xa = torch.min(torch.max(xa, x - eps), x + eps).clamp(0, 1).detach()
    model.train()
    loss = F.cross_entropy(model(xa), y)
    return loss, loss.item(), 0.0


def train_trades(model, Xtr, ytr, tcfg, device, seed, epochs=None, Xval=None, yval=None, method="trades"):
    epochs = epochs or tcfg["epochs_per_cycle"]
    eps0, alpha0, bs = tcfg["eps_255"] / 255, tcfg["alpha_255"] / 255, tcfg["batch_size"]
    nb = max(len(Xtr) // bs, 1)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * nb, eta_min=tcfg["lr"] * 0.05)
    aug = make_augment() if tcfg.get("augment", True) else None
    warm, hist = tcfg.get("warmup_epochs", 0), []
    model.to(device)
    for ep in range(1, epochs + 1):
        t0, L, C, K, n = time.time(), 0.0, 0.0, 0.0, 0
        for i, (xb, yb) in enumerate(iterate(Xtr, ytr, bs, device, True, seed * 1000 + ep, aug, True)):
            frac = min(1.0, ((ep - 1) + (i + 1) / nb) / warm) if warm else 1.0   # eps/beta ramp
            if method == "trades":
                loss, ce, kl = trades_loss(model, xb, yb, eps0 * frac, alpha0 * frac, tcfg["steps"], tcfg["beta"] * frac)
            else:
                loss, ce, kl = madry_loss(model, xb, yb, eps0 * frac, alpha0 * frac, tcfg["steps"])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            L += loss.item(); C += ce; K += kl; n += 1
        row = dict(epoch=ep, loss=L / max(n, 1), ce=C / max(n, 1), kl=K / max(n, 1),
                   lr=sched.get_last_lr()[0], ramp=frac, seconds=time.time() - t0)
        if Xval is not None:
            model.eval(); ok = 0
            with torch.no_grad():
                for xb, yb in iterate(Xval, yval, 64, device):
                    ok += (model(xb).argmax(1) == yb).sum().item()
            row["val_clean_acc"] = 100 * ok / len(yval)
        hist.append(row)
        print(f"[{method}] ep{ep} loss={row['loss']:.4f} ce={row['ce']:.4f} kl={row['kl']:.4f} "
              f"val_clean={row.get('val_clean_acc', float('nan')):.1f}")
    model.eval()
    return hist