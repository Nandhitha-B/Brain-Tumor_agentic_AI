"""Bounded perturbation probes. All attacks operate on images in [0,1] and call model(x) -> logits.
FGSM / PGD / BIM : L-inf <= eps.   C&W-L2 : L2 <= l2_eps (final delta projected onto the ball).
Square : gradient-free L-inf attack, used only as a held-out attack.
BIM here = PGD without random start (both clip to the eps-ball every step)."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def _grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    return torch.autograd.grad(loss, x)[0]


def fgsm(model, x, y, eps):
    return (x + eps * _grad(model, x, y).sign()).clamp(0, 1).detach()


def pgd(model, x, y, eps, alpha, steps, random_start=True):
    xa = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1) if random_start else x.clone()
    for _ in range(steps):
        xa = xa.detach() + alpha * _grad(model, xa, y).sign()
        xa = torch.min(torch.max(xa, x - eps), x + eps).clamp(0, 1)
    return xa.detach()


def bim(model, x, y, eps, alpha, steps):
    return pgd(model, x, y, eps, alpha, steps, random_start=False)


def cw_l2(model, x, y, l2_eps, steps=100, lr=0.01, c_init=1.0, binary_steps=1, kappa=0.0):
    b = x.shape[0]
    w0 = torch.atanh((x * 2 - 1).clamp(-1 + 1e-6, 1 - 1e-6))
    const = torch.full((b,), c_init, device=x.device)
    lower, upper = torch.zeros(b, device=x.device), torch.full((b,), 1e10, device=x.device)
    best_l2 = torch.full((b,), float("inf"), device=x.device)
    best_adv = x.clone()
    last_adv = x.clone()
    onehot = F.one_hot(y, num_classes=model(x[:1]).shape[1]).bool()
    for _ in range(binary_steps):
        w = w0.clone().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        succ_bs = torch.zeros(b, dtype=torch.bool, device=x.device)
        for _ in range(steps):
            xa = (torch.tanh(w) + 1) / 2
            logits = model(xa)
            l2 = (xa - x).flatten(1).pow(2).sum(1)
            real = logits[onehot]
            other = logits.masked_fill(onehot, -1e9).max(1)[0]
            f = (real - other + kappa).clamp(min=0)
            loss = (l2 + const * f).sum()
            opt.zero_grad()
            loss.backward(inputs=[w])
            opt.step()
            with torch.no_grad():
                mis = logits.argmax(1) != y
                better = mis & (l2 < best_l2)
                best_l2 = torch.where(better, l2, best_l2)
                best_adv[better] = xa.detach()[better]
                succ_bs |= mis
                last_adv = xa.detach()
        upper = torch.where(succ_bs, torch.min(upper, const), upper)
        lower = torch.where(~succ_bs, torch.max(lower, const), lower)
        const = torch.where(upper < 1e9, (lower + upper) / 2, const * 10)
    adv = torch.where(torch.isfinite(best_l2).view(-1, 1, 1, 1), best_adv, last_adv)
    delta = adv - x                                   # enforce the L2 budget (bounded probe)
    norm = delta.flatten(1).norm(dim=1).clamp(min=1e-12)
    delta = delta * (l2_eps / norm).clamp(max=1.0).view(-1, 1, 1, 1)
    return (x + delta).clamp(0, 1).detach()

def pgd_restarts(model, x, y, eps, alpha, steps, restarts):
    best, fooled, xa = x.clone(), torch.zeros(len(x), dtype=torch.bool, device=x.device), x
    for _ in range(restarts):
        xa = pgd(model, x, y, eps, alpha, steps, True)
        with torch.no_grad():
            wrong = model(xa).argmax(1) != y
        take = wrong & ~fooled
        best[take] = xa[take]; fooled |= wrong
    best[~fooled] = xa[~fooled]
    return best

def _p_selection(p_init, it, n_it):
    it = int(it / n_it * 10000)
    for thr, div in ((10, 1), (50, 2), (200, 4), (500, 8), (1000, 16), (2000, 32), (4000, 64), (6000, 128), (8000, 256)):
        if it <= thr:
            return p_init / div
    return p_init / 512


@torch.no_grad()
def square_linf(model, x, y, eps, n_queries=1000, p_init=0.05):
    b, c, h, w = x.shape
    def margin(logits, yy):
        real = logits.gather(1, yy[:, None]).squeeze(1)
        other = logits.clone(); other.scatter_(1, yy[:, None], -1e9)
        return real - other.max(1)[0]
    stripes = torch.randint(0, 2, (b, c, 1, w), device=x.device).float() * 2 - 1
    xb = (x + eps * stripes).clamp(0, 1)
    mb = margin(model(xb), y)
    for i in range(n_queries):
        active = (mb > 0).nonzero().squeeze(1)
        if active.numel() == 0:
            break
        s = min(max(int(round(math.sqrt(_p_selection(p_init, i, n_queries) * h * w))), 1), h - 1)
        cand, xo = xb[active].clone(), x[active]
        for k in range(active.numel()):
            r = int(torch.randint(0, h - s + 1, (1,))); cc = int(torch.randint(0, w - s + 1, (1,)))
            d = (torch.randint(0, 2, (c, 1, 1), device=x.device).float() * 2 - 1) * eps
            cand[k, :, r:r + s, cc:cc + s] = (xo[k, :, r:r + s, cc:cc + s] + d).clamp(0, 1)
        mn = margin(model(cand), y[active])
        imp = mn < mb[active]
        idx = active[imp]
        xb[idx] = cand[imp]; mb[idx] = mn[imp]
    return xb


def make_attack(name: str, probes: dict, eps: float):
    """Return fn(model, x, y) -> x_adv for a named probe."""
    if name == "fgsm":
        return lambda m, x, y: fgsm(m, x, y, eps)
    if name in ("pgd", "bim"):
        p = probes[name]
        a = p["alpha_255"] / 255
        if name == "pgd":
            return lambda m, x, y: pgd(m, x, y, eps, a, p["steps"], p.get("random_start", True))
        return lambda m, x, y: bim(m, x, y, eps, a, p["steps"])
    if name == "cw":
        p = probes["cw"]
        return lambda m, x, y: cw_l2(m, x, y, p["l2_eps"], p["steps"], p["lr"], p["c_init"], p["binary_steps"], p["kappa"])
    if name == "square":
        p = probes["square"]
        return lambda m, x, y: square_linf(m, x, y, eps, p["n_queries"], p["p_init"])
    if name in ("apgd_ce", "apgd_dlr"):      # needs: pip install torchattacks
        import torchattacks
        steps = probes.get("apgd", {"steps": 50})["steps"]
        cache = {}
        def fn(m, x, y):
            if cache.get("m") is not m:
                cache["atk"] = torchattacks.APGD(m, norm="Linf", eps=eps, steps=steps, n_restarts=1,
                                                 loss=name.split("_")[1], seed=0)
                cache["m"] = m
            return cache["atk"](x, y).detach()
        return fn
    if name == "pgd_rs":
        p = probes["pgd_rs"]; a = p["alpha_255"] / 255
        return lambda m, x, y: pgd_restarts(m, x, y, eps, a, p["steps"], p["restarts"])
    raise ValueError(name)


LINF_ATTACKS = ("fgsm", "pgd", "bim", "square", "apgd_ce", "apgd_dlr","pgd_rs")
