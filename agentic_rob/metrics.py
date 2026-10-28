"""Metric definitions (numpy only). All accuracies in %, gaps in percentage points (pp)."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (centre - half), 100 * (centre + half))


def attack_summary(y, pred_clean, conf_clean, pred_adv, conf_adv) -> dict:
    """ASR (standard)  = P[adv wrong | clean correct]  (needs the TRUE label y).
    flip_rate          = P[pred changes] over all samples.
    conf_drop          = mean(max-softmax clean - max-softmax adv)."""
    y, pc, pa = map(np.asarray, (y, pred_clean, pred_adv))
    cc, ca = pc == y, pa == y
    n = len(y)
    lo, hi = wilson_ci(int(ca.sum()), n)
    return {"n": n, "acc": 100 * ca.mean(), "acc_ci_low": lo, "acc_ci_high": hi,
            "asr": 100 * (cc & ~ca).sum() / max(cc.sum(), 1),
            "flip_rate": 100 * (pa != pc).mean(),
            "conf_drop": float(np.mean(np.asarray(conf_clean) - np.asarray(conf_adv)))}


def robustness_gap(a_clean: float, accs: dict):
    """Empirical gap dR = A_clean - min_k A_k over the evaluated probes (NOT a certified worst case)."""
    worst = min(accs, key=accs.get)
    return a_clean - accs[worst], worst


def per_class_table(y, pred, class_names):
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    rows = [dict(cls=class_names[i], precision=p[i], recall=r[i], f1=f[i], support=int(s[i])) for i in labels]
    mp, mr, mf, _ = precision_recall_fscore_support(y, pred, labels=labels, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y, pred, labels=labels, average="weighted", zero_division=0)
    rows.append(dict(cls="macro", precision=mp, recall=mr, f1=mf, support=int(s.sum())))
    rows.append(dict(cls="weighted", precision=wp, recall=wr, f1=wf, support=int(s.sum())))
    return rows


def confusion(y, pred, n_classes):
    return confusion_matrix(y, pred, labels=list(range(n_classes)))


def paired_bootstrap_diff(correct_a, correct_b, n_boot=2000, seed=0):
    """Mean accuracy difference (b - a) in pp with 95% percentile CI, resampling test images."""
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    rng = np.random.RandomState(seed)
    n = len(a)
    d = np.array([(b[i] - a[i]).mean() for i in (rng.randint(0, n, n) for _ in range(n_boot))]) * 100
    return float((b - a).mean() * 100), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def gap_stats(clean_correct, correct, names):
    """dR = A_clean - min_k A_k, with a paired standard error (same images in every term)."""
    cc = np.asarray(clean_correct, float)
    accs = {k: 100 * float(np.mean(correct[k])) for k in names}
    worst = min(accs, key=accs.get)
    d = (cc - np.asarray(correct[worst], float)) * 100
    return float(d.mean()), worst, float(d.std(ddof=1) / np.sqrt(len(d))), accs