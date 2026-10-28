"""Controls (no agent): direct_trades (default recipe), direct_selected (the recipe the agent chose),
madry_at (PGD adversarial training). Epoch budget = total epochs the agent promoted."""
from __future__ import annotations
import json
from agentic_rob.config import base_parser, load_config, Paths, set_seed, get_device
from agentic_rob.data.dataset import DataStore
from agentic_rob.models import build_model
from agentic_rob.registry import ModelRegistry
from agentic_rob.trades import train_trades


def run(cfg, paths, seed, epochs=None):
    dev, reg = get_device(), ModelRegistry(paths.registry)
    events = [json.loads(l) for l in paths.agent_log.read_text().splitlines()] if paths.agent_log.exists() else []
    promos = [e for e in events if e["event"] == "promotion"]
    n_epochs = epochs or sum(e["epochs"] for e in promos) or cfg["trades"]["epochs_per_cycle"]
    default = cfg["trades"]["recipes"][0]
    selected = promos[0]["recipe"] if promos else default
    jobs = [("direct_trades", default, "trades"), ("madry_at", default, "madry")]
    if selected["id"] != default["id"]:
        jobs.insert(1, ("direct_selected", selected, "trades"))
    base = reg.first_of_kind("baseline")
    ds = DataStore(paths.processed, cfg["data"]["max_per_split"], cfg["data"]["split_seed"])
    Xtr, ytr = ds.split("train"); Xva, yva = ds.split("val")
    for kind, recipe, method in jobs:
        set_seed(seed)
        t = dict(cfg["trades"]); t.update({k: v for k, v in recipe.items() if k != "id"})
        model = reg.load_into(build_model(cfg, len(cfg["classes"])), base).to(dev)
        hist = train_trades(model, Xtr, ytr, t, dev, seed * 1000 + 1, epochs=n_epochs, Xval=Xva, yval=yva, method=method)
        v = reg.register(model.cpu(), parent=base, kind=kind, meta=dict(epochs=n_epochs, recipe=recipe, method=method))
        reg.set_status(v, "control")
        (paths.logs / f"{kind}_history.json").write_text(json.dumps(hist, indent=2))
        print(f"{kind} registered as {v} ({n_epochs} epochs)")


if __name__ == "__main__":
    ap = base_parser("Controls without the agent")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    c = load_config(a.config, a.override)
    run(c, Paths(c, a.seed), a.seed, a.epochs)