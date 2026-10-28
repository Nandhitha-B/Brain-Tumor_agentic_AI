"""Threshold analysis (MONITOR split only): false-trigger rate on a healthy model vs detection of degradation."""
import copy
import numpy as np, pandas as pd
from agentic_rob.config import base_parser, load_config, Paths, set_seed, get_device
from agentic_rob.data.dataset import DataStore
from agentic_rob.models import build_model
from agentic_rob.profiler import profile
from agentic_rob.registry import ModelRegistry


def run(cfg, paths, seed, n_boot=2000):
    set_seed(seed); dev = get_device(); reg = ModelRegistry(paths.registry)
    X, y = DataStore(paths.processed, cfg["data"]["max_per_split"], cfg["data"]["split_seed"]).split("monitor")
    e0, bs, rng = cfg["attack"]["eps_255"], cfg["agent"]["monitor_batch_size"], np.random.RandomState(seed)
    base, final, full = reg.first_of_kind("baseline"), reg.current, ("fgsm", "pgd", "bim", "cw")
    conds = {"undefended": (base, e0, full), "defended (healthy)": (final, e0, full),
             "defended (2x budget drift)": (final, 2 * e0, full)}
    gaps = {}
    for name, (v, e, atks) in conds.items():
        ps = copy.deepcopy(cfg["probes_monitor"])
        for k in ("pgd", "bim"): 
            ps[k]["alpha_255"] = e / 4
            ps["cw"]["l2_eps"] *= e /e0
        r = profile(reg.load_into(build_model(cfg, len(cfg["classes"])), v), X, y, ps, e, dev, bs, seed, atks, atks)
        cc = r["clean_correct"].astype(float); adv = np.stack([r["correct"][a] for a in atks]).astype(float)
        n = len(y); g = []
        for _ in range(n_boot):
            i = rng.randint(0, n, n); g.append(100 * (cc[i].mean() - adv[:, i].mean(1).min()))
        gaps[name] = np.array(g)
    df = pd.DataFrame([dict(tau_pp=t, **{n: float((g > t).mean()) for n, g in gaps.items()}) for t in range(1, 96)])
    df.to_csv(paths.results / "threshold_analysis.csv", index=False)
    for n, g in gaps.items():
        print(f"{n}: dR mean {g.mean():.1f}  95% [{np.percentile(g, 2.5):.1f}, {np.percentile(g, 97.5):.1f}]")
    ok = df[df["defended (healthy)"] <= 0.01]
    print("smallest tau with <=1% false-trigger rate on the healthy model:", int(ok.tau_pp.min()) if len(ok) else None)
    print(df[df.tau_pp.isin([5, 10, 15, 20, 25, 30])].round(3).to_markdown(index=False))


if __name__ == "__main__":
    a = base_parser("Threshold analysis").parse_args()
    c = load_config(a.config, a.override)
    run(c, Paths(c, a.seed), a.seed)