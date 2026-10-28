"""Build tables (CSV/Markdown) and figures from MEASURED outputs only. Nothing is projected."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from agentic_rob.config import base_parser, load_config, Paths

ATT = ["fgsm", "pgd", "bim", "cw"]


def _events(path):
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def run(cfg, paths, seed):
    out = paths.results / "final"; fig = paths.results / "figures"; fig.mkdir(parents=True, exist_ok=True)
    md = ["# Results summary (auto-generated, measured values only)\n"]
    models = [m for m in ("baseline", "agent_final", "direct_trades", "direct_selected", "madry_at") if (out / m / "attack_metrics.csv").exists()]
    tabs = {m: pd.read_csv(out / m / "attack_metrics.csv").set_index("attack") for m in models}
    for m in models:
        s = json.loads((out / m / "summary.json").read_text())
        md.append(f"\n## {m} (version {s.get('version')}) | clean {s['A_clean']:.2f}% | worst {s['worst_attack']} "
                  f"{s['A_worst_case']:.2f}% | dR = {s['delta_R_pp']:.2f} pp\n")
        md.append(tabs[m].round(2).to_markdown() + "\n")
    if len(models) > 1:
        rows = []
        for cond in ["clean"] + ATT:
            r = {"condition": cond}
            for m in models:
                r[m] = tabs[m].loc[cond, "acc"]
            rows.append(r)
        cmp = pd.DataFrame(rows); cmp.to_csv(paths.results / "table_before_after.csv", index=False)
        md.append("\n## Accuracy (%) per model, same test images and attack settings\n" + cmp.round(2).to_markdown(index=False) + "\n")
        for f in sorted(out.glob("compare_*.csv")):
            md.append(f"\n### {f.stem} (paired bootstrap, pp)\n" + pd.read_csv(f).round(2).to_markdown(index=False) + "\n")
        x = np.arange(5); w = 0.8 / len(models)
        plt.figure(figsize=(8, 4))
        for i, m in enumerate(models):
            plt.bar(x + i * w, cmp[m], w, label=m)
        plt.xticks(x + w * (len(models) - 1) / 2, cmp.condition); plt.ylabel("Accuracy (%)"); plt.legend()
        plt.title("Test accuracy: clean and under attack"); plt.tight_layout(); plt.savefig(fig / "fig_before_after.png", dpi=200); plt.close()
        plt.figure(figsize=(8, 4))
        for i, m in enumerate(models):
            plt.bar(x[1:] + i * w, tabs[m].loc[ATT, "gap_pp"], w, label=m)
        plt.axhline(cfg["agent"]["tau_pp"], ls=":", c="r", label="tau"); plt.xticks(x[1:] + w * (len(models) - 1) / 2, ATT)
        plt.ylabel("Robustness gap (pp)"); plt.legend(); plt.tight_layout(); plt.savefig(fig / "fig_gap.png", dpi=200); plt.close()
    if "baseline" in tabs:
        fig2, ax = plt.subplots(figsize=(7, 4)); t = tabs["baseline"].loc[ATT]
        ax.bar(ATT, t.asr, label="ASR (%)"); ax2 = ax.twinx(); ax2.plot(ATT, t.conf_drop, "ko-", label="conf. drop")
        ax.set_ylabel("ASR (%)"); ax2.set_ylabel("Max-softmax confidence drop"); plt.tight_layout(); plt.savefig(fig / "fig_asr_conf_baseline.png", dpi=200); plt.close()
    # per-class F1
    for m in models:
        pc = pd.read_csv(out / m / "per_class_metrics.csv"); pc = pc[~pc.cls.isin(["macro", "weighted"])]
        piv = pc.pivot(index="condition", columns="cls", values="f1").reindex(["clean"] + ATT)
        piv.to_csv(paths.results / f"per_class_f1_{m}.csv"); piv.plot(kind="bar", figsize=(7, 4)); plt.ylabel("F1")
        plt.title(f"Per-class F1: {m}"); plt.tight_layout(); plt.savefig(fig / f"fig_per_class_f1_{m}.png", dpi=200); plt.close()
    # agent trajectory + TRADES loss (measured, from the agent's own log, MONITOR set)
    ev = _events(paths.agent_log)
    evals = [e for e in ev if e["event"] == "evaluation"]
    vals = [e for e in ev if e["event"] == "candidate_validation"]
    traj = [(e["model_version"], e["delta_R_pp"], "serving") for e in evals]
    traj += [(e["candidate"], e["delta_R_pp"], "accepted" if e["accepted"] else "rejected") for e in vals if not e["accepted"]]
    if evals:
        pd.DataFrame(traj, columns=["version", "delta_R_pp_monitor", "status"]).to_csv(paths.results / "agent_trajectory.csv", index=False)
        plt.figure(figsize=(6, 4)); plt.plot([e["cycle"] for e in evals], [e["delta_R_pp"] for e in evals], "o-")
        plt.axhline(cfg["agent"]["tau_pp"], ls="--", c="r", label="tau"); plt.xlabel("Evaluation cycle"); plt.ylabel("dR on monitor set (pp)")
        plt.legend(); plt.title("Agent trajectory (measured, monitor set)"); plt.tight_layout(); plt.savefig(fig / "fig_agent_trajectory.png", dpi=200); plt.close()
        md.append("\n## Agent event summary\n" + "\n".join(
            f"- cycle {e['cycle']}: {e['model_version']} dR={e['delta_R_pp']:.2f} pp ({e['worst_attack']}) -> {e['decision']}" for e in evals))
        term = [e for e in ev if e["event"] == "terminal"]
        if term:
            md.append(f"\n- terminal status: **{term[-1]['status']}**, serving {term[-1]['final_version']}")
        md.append("\n\n### Threshold sensitivity (would the trigger have fired?)\n")
        md.append(pd.DataFrame([{"tau_pp": t, **{e["model_version"]: e["delta_R_pp"] > t for e in evals}}
                                for t in (5, 10, 15, 20, 25, 30)]).to_markdown(index=False))
    rts = [e for e in ev if e["event"] == "retraining"]
    if rts:
        plt.figure(figsize=(6, 4))
        for e in rts:
            h = pd.DataFrame(e["trades_history"])
            plt.plot(h.epoch, h.loss, "o-", label=f"c{e['cycle']}-{e['recipe']['id']} total"); plt.plot(h.epoch, h.ce, "--", label=f"cycle {e['cycle']} CE")
        plt.xlabel("Epoch within cycle"); plt.ylabel("TRADES loss (mean over batches)"); plt.legend(fontsize=7)
        plt.title("Training loss only - NOT evidence of robustness"); plt.tight_layout(); plt.savefig(fig / "fig_trades_loss.png", dpi=200); plt.close()
    (paths.results / "results_summary.md").write_text("\n".join(md))
    print(f"wrote {paths.results / 'results_summary.md'}")


if __name__ == "__main__":
    a = base_parser("Make tables and figures").parse_args()
    c = load_config(a.config, a.override)
    run(c, Paths(c, a.seed), a.seed)
