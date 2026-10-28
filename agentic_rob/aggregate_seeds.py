"""Mean +/- SD across training seeds (run after run_pipeline with several seeds)."""
import sys
import pandas as pd
from pathlib import Path

runs = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
rows = []
for d in sorted(runs.glob("seed*")):
    for m in ("baseline", "agent_final", "direct_trades","direct_selected","madry_at"):
        f = d / "results" / "final" / m / "attack_metrics.csv"
        if f.exists():
            t = pd.read_csv(f); t["model"], t["seed"] = m, d.name; rows.append(t)
if not rows:
    sys.exit("no results found")
df = pd.concat(rows)
agg = df.groupby(["model", "attack"])[["acc", "asr", "conf_drop", "gap_pp"]].agg(["mean", "std"]).round(2)
agg.to_csv(runs / "aggregate_over_seeds.csv"); print(agg.to_markdown())
