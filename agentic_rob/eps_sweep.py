import copy
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from agentic_rob.config import base_parser, load_config, Paths, set_seed, get_device
from agentic_rob.data.dataset import DataStore
from agentic_rob.evaluate import targets_for
from agentic_rob.models import build_model
from agentic_rob.profiler import profile
from agentic_rob.registry import ModelRegistry


def run(cfg, paths, seed):
    set_seed(seed); dev = get_device(); reg = ModelRegistry(paths.registry)
    X, y = DataStore(paths.processed, cfg["data"]["max_per_split"], cfg["data"]["split_seed"]).split("test")
    rows = []
    for name, v in targets_for(reg).items():
        if v is None: continue
        for e in cfg["eval"]["sweep_eps_255"]:
            ps = copy.deepcopy(cfg["probes_final"]); ps["pgd"]["alpha_255"] = e / 4
            r = profile(reg.load_into(build_model(cfg, len(cfg["classes"])), v), X, y, ps, e, dev,
                        cfg["eval"]["batch_size"], seed, ("pgd",), ("pgd",))
            rows.append(dict(model=name, eps_255=e, pgd_acc=r["attacks"]["pgd"]["acc"]))
    df = pd.DataFrame(rows); df.to_csv(paths.results / "eps_sweep.csv", index=False)
    piv = df.pivot(index="eps_255", columns="model", values="pgd_acc"); print(piv.round(2).to_markdown())
    piv.plot(marker="o", figsize=(6, 4)); plt.ylabel("PGD accuracy (%)"); plt.xlabel("eps (/255)"); plt.xscale("log", base=2)
    plt.tight_layout(); (paths.results / "figures").mkdir(exist_ok=True)
    plt.savefig(paths.results / "figures" / "fig_eps_sweep.png", dpi=200)


if __name__ == "__main__":
    a = base_parser("PGD accuracy vs epsilon").parse_args()
    c = load_config(a.config, a.override)
    run(c, Paths(c, a.seed), a.seed)