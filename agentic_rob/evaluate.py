"""Final evaluation on the LOCKED test split, identical protocol for every model."""
from __future__ import annotations
import copy, json
import numpy as np
import pandas as pd
from agentic_rob.config import base_parser, load_config, Paths, set_seed, get_device
from agentic_rob.data.dataset import DataStore
from agentic_rob.metrics import per_class_table, confusion, paired_bootstrap_diff, wilson_ci
from agentic_rob.models import build_model, Smoothed, FeatureSqueeze
from agentic_rob.profiler import profile, PRIMARY
from agentic_rob.registry import ModelRegistry


def targets_for(reg):
    return {"baseline": reg.first_of_kind("baseline"), "agent_final": reg.current,
            "direct_trades": reg.first_of_kind("direct_trades"), "direct_selected": reg.first_of_kind("direct_selected"),
            "madry_at": reg.first_of_kind("madry_at")}


def save_result(res, name, out, classes):
    d = out / name; d.mkdir(parents=True, exist_ok=True)
    ps = res["per_sample"]
    lo, hi = wilson_ci(int((ps["pred_clean"] == ps["y"]).sum()), len(ps["y"]))
    rows = [dict(attack="clean", acc=res["A_clean"], acc_ci_low=lo, acc_ci_high=hi, asr=np.nan,
                 flip_rate=np.nan, conf_drop=np.nan, gap_pp=0.0)]
    for a, s in res["attacks"].items():
        rows.append(dict(attack=a, acc=s["acc"], acc_ci_low=s["acc_ci_low"], acc_ci_high=s["acc_ci_high"], asr=s["asr"],
                         flip_rate=s["flip_rate"], conf_drop=s["conf_drop"], gap_pp=res["A_clean"] - s["acc"],
                         linf_max=s["linf_max"], l2_max=s["l2_max"]))
    pd.DataFrame(rows).to_csv(d / "attack_metrics.csv", index=False)
    pc = []
    for cond in ["clean"] + list(res["attacks"]):
        pred = ps["pred_clean"] if cond == "clean" else ps[f"pred_{cond}"]
        pc += [dict(condition=cond, **r) for r in per_class_table(ps["y"], pred, classes)]
        pd.DataFrame(confusion(ps["y"], pred, len(classes)), index=classes, columns=classes).to_csv(d / f"confusion_{cond}.csv")
    pd.DataFrame(pc).to_csv(d / "per_class_metrics.csv", index=False)
    np.savez(d / "per_sample.npz", **ps)
    summ = {k: res[k] for k in ("n", "A_clean", "A_worst_case", "worst_attack", "delta_R_pp") if k in res}
    (d / "summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def compare(a_name, b_name, out, attacks, n_boot):
    A, B = np.load(out / a_name / "per_sample.npz"), np.load(out / b_name / "per_sample.npz")
    rows = []
    for cond in ["clean"] + attacks:
        pa = A["pred_clean"] if cond == "clean" else A[f"pred_{cond}"]
        pb = B["pred_clean"] if cond == "clean" else B[f"pred_{cond}"]
        m, lo, hi = paired_bootstrap_diff(pa == A["y"], pb == B["y"], n_boot)
        rows.append(dict(condition=cond, **{f"acc_{a_name}": 100 * (pa == A["y"]).mean(),
                         f"acc_{b_name}": 100 * (pb == B["y"]).mean()}, diff_pp=m, ci_low=lo, ci_high=hi))
    pd.DataFrame(rows).to_csv(out / f"compare_{b_name}_vs_{a_name}.csv", index=False)


def run(cfg, paths, seed, heldout=False, smoothing_ablation=False, apgd=False, stress=False):
    set_seed(seed); dev = get_device()
    reg = ModelRegistry(paths.registry)
    X, y = DataStore(paths.processed, cfg["data"]["max_per_split"], cfg["data"]["split_seed"]).split("test")
    classes, probes, eps, bs = cfg["classes"], cfg["probes_final"], cfg["attack"]["eps_255"], cfg["eval"]["batch_size"]
    out = paths.results / "final"; out.mkdir(parents=True, exist_ok=True)
    attacks = PRIMARY + (("square",) if heldout else ()) + (("pgd_rs", "apgd_ce", "apgd_dlr") if apgd else ())    
    fresh = lambda v: reg.load_into(build_model(cfg, len(classes)), v)
    done = []
    for name, v in targets_for(reg).items():
        if v is None:
            print(f"skip {name}"); continue
        res = profile(fresh(v), X, y, probes, eps, dev, bs, seed, attacks, PRIMARY, True)
        s = save_result(res, name, out, classes); s["version"] = v
        (out / name / "summary.json").write_text(json.dumps(s, indent=2)); print(name, v, s); done.append(name)
        if smoothing_ablation:
            for tag, wrap in (("smoothing", Smoothed), ("bitdepth", FeatureSqueeze)):
                r2 = profile(wrap(fresh(v)), X, y, probes, eps, dev, bs, seed, PRIMARY, PRIMARY, True)
                save_result(r2, f"{name}+{tag}", out, classes)
        if stress:
            se = cfg["eval"]["stress_eps_255"]; ps = copy.deepcopy(probes); ps["pgd"]["alpha_255"] = se / 4
            r3 = profile(fresh(v), X, y, ps, se, dev, bs, seed, ("pgd",), ("pgd",), True)
            save_result(r3, f"{name}@eps{se}", out, classes)
    for a, b in (("baseline", "agent_final"), ("agent_final", "direct_trades"),
                 ("agent_final", "direct_selected"), ("agent_final", "madry_at")):
        if a in done and b in done:
            compare(a, b, out, list(attacks), cfg["eval"]["bootstrap_samples"])


if __name__ == "__main__":
    ap = base_parser("Final test-set evaluation")
    for f in ("--heldout", "--smoothing-ablation", "--apgd", "--stress"):
        ap.add_argument(f, action="store_true")
    a = ap.parse_args()
    c = load_config(a.config, a.override)
    run(c, Paths(c, a.seed), a.seed, a.heldout, a.smoothing_ablation, a.apgd, a.stress)