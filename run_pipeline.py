import argparse
from agentic_rob.config import load_config, Paths
from agentic_rob.data import prepare_figshare
from agentic_rob import train_baseline, agent, direct_trades, evaluate, report, shift_eval, eps_sweep, threshold_analysis

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--override", default=None)
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--stages", nargs="+", default=["prepare", "baseline", "agent", "direct", "eval", "shift", "sweep", "threshold", "report"])
for f in ("--heldout", "--smoothing-ablation", "--apgd", "--stress", "--deploy-demo"):
    ap.add_argument(f, action="store_true")
a = ap.parse_args()
cfg = load_config(a.config, a.override)

if "prepare" in a.stages and not (Paths(cfg, 0).processed / "images.npy").exists():
    prepare_figshare.run(cfg, Paths(cfg, 0))
for s in a.seeds:
    P = Paths(cfg, s)
    print(f"\n=========== seed {s} ===========")
    if "baseline" in a.stages: train_baseline.run(cfg, P, s)
    if "agent" in a.stages: agent.run(cfg, P, s)
    if "direct" in a.stages: direct_trades.run(cfg, P, s)
    if "eval" in a.stages: evaluate.run(cfg, P, s, a.heldout, a.smoothing_ablation, a.apgd, a.stress)
    if "shift" in a.stages: shift_eval.run(cfg, P, s)
    if "sweep" in a.stages: eps_sweep.run(cfg, P, s)
    if "threshold" in a.stages: threshold_analysis.run(cfg, P, s)
    if "report" in a.stages: report.run(cfg, P, s)
    if a.deploy_demo:
        from agentic_rob import deploy_demo
        deploy_demo.run(cfg, P, s, a.override, a.config)

if len(a.seeds) > 1:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "agentic_rob.aggregate_seeds", cfg["paths"]["runs_dir"]])