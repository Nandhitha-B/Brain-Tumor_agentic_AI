"""Autonomous robustness agent (System B).

Capabilities (all rule-based, all logged):
  * budgeted probing   : cheap probes first; stop as soon as a violation is statistically established
  * adaptive tau       : trigger only if dR - z*SE(dR) > tau (accounts for monitor-set sampling noise)
  * hyper-parameter search with memory: trains several TRADES recipes, keeps the best validated one
  * validation guardrails: clean-accuracy drop, minimum robust gain, gradient-masking check (Square)
  * rollback + escalation: writes ESCALATION.txt for a human when no candidate passes
The agent never touches the TEST split.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from agentic_rob.config import base_parser, load_config, Paths, set_seed, get_device
from agentic_rob.data.dataset import DataStore
from agentic_rob.metrics import gap_stats
from agentic_rob.models import build_model
from agentic_rob.profiler import profile
from agentic_rob.registry import ModelRegistry
from agentic_rob.trades import train_trades


class RobustnessAgent:
    def __init__(self, cfg, registry, data, device, log_path, seed=0):
        self.cfg, self.reg, self.dev, self.seed = cfg, registry, device, seed
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.Xm, self.ym = data.split("monitor")
        self.Xtr, self.ytr = data.split("train")
        self.Xva, self.yva = data.split("val")
        self.state, self.memory, self.ref_clean = {}, {}, None

    def log(self, **ev):
        ev["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.log_path, "a") as f:
            f.write(json.dumps(ev) + "\n")

    def _model(self, v):
        return self.reg.load_into(build_model(self.cfg, len(self.cfg["classes"])), v).to(self.dev)

    # ---- tool: probe (cached, runs only the probes not yet run for this version) ----------
    def probe(self, v, names):
        st = self.state.setdefault(v, {"correct": {}, "clean_correct": None, "A_clean": None})
        todo = [n for n in names if n not in st["correct"]]
        if todo:
            r = profile(self._model(v), self.Xm, self.ym, self.cfg["probes_monitor"], self.cfg["attack"]["eps_255"],
                        self.dev, self.cfg["agent"]["monitor_batch_size"], self.seed, tuple(todo), tuple(todo))
            st["clean_correct"], st["A_clean"] = r["clean_correct"], r["A_clean"]
            for n in todo:
                st["correct"][n] = r["correct"][n]
        present = [n for n in names if n in st["correct"]]
        gap, worst, se, accs = gap_stats(st["clean_correct"], st["correct"], present)
        return dict(version=v, A_clean=st["A_clean"], delta_R_pp=gap, se=se, worst_attack=worst,
                    A_worst_case=accs[worst], acc=accs, probes=present)

    # ---- policy: adaptive threshold -------------------------------------------------------
    def decide(self, obs):
        a = self.cfg["agent"]
        z = a["tau_z"] if a["tau_mode"] == "adaptive" else 0.0
        if obs["delta_R_pp"] - z * obs["se"] > a["tau_pp"]:
            return "TRIGGER_RETRAINING"
        return "UNCERTAIN_CONTINUE_MONITORING" if obs["delta_R_pp"] > a["tau_pp"] else "CONTINUE_MONITORING"

    def ordered_recipes(self):
        rec = self.cfg["trades"]["recipes"]
        known = sorted([r for r in rec if r["id"] in self.memory], key=lambda r: -self.memory[r["id"]])
        return (known + [r for r in rec if r["id"] not in self.memory])[: self.cfg["trades"]["n_candidates"]]

    def train_candidate(self, parent, recipe, cycle, idx):
        t = dict(self.cfg["trades"]); t.update({k: v for k, v in recipe.items() if k != "id"})
        model, t0 = self._model(parent), time.time()
        hist = train_trades(model, self.Xtr, self.ytr, t, self.dev, self.seed * 1000 + cycle * 10 + idx,
                            Xval=self.Xva, yval=self.yva)
        v = self.reg.register(model.cpu(), parent=parent, kind="agent_candidate",
                              meta=dict(cycle=cycle, recipe=recipe, epochs=t["epochs_per_cycle"], eps_255=t["eps_255"]))
        self.log(event="retraining", cycle=cycle, parent=parent, candidate=v, recipe=recipe,
                 epochs=t["epochs_per_cycle"], trades_history=hist, seconds=time.time() - t0)
        return v

    def validate(self, v, recipe, cur_full, cycle):
        a = self.cfg["agent"]
        c, stage = self.probe(v, a["cheap_probes"]), "cheap"
        drop, ok, reason = self.ref_clean - c["A_clean"], True, "ok"
        if drop > a["max_clean_drop_pp"]:
            ok, reason = False, "clean-accuracy guardrail"
        else:
            c, stage = self.probe(v, a["full_probes"]), "full"
            gain = c["A_worst_case"] - cur_full["A_worst_case"]
            if gain < a["min_gain_pp"]:
                ok, reason = False, f"insufficient robust-accuracy gain ({gain:.1f} pp)"
            elif a.get("masking_probe"):
                m = self.probe(v, [a["masking_probe"]])["acc"][a["masking_probe"]]
                if m - c["A_worst_case"] > a["masking_tol_pp"]:
                    ok, reason = False, "possible gradient masking (gradient-free probe much weaker)"
        self.memory[recipe["id"]] = max(self.memory.get(recipe["id"], -1), c["A_worst_case"] if drop <= a["max_clean_drop_pp"] else -1)
        self.log(event="candidate_validation", cycle=cycle, candidate=v, recipe=recipe, stage=stage,
                 A_clean=c["A_clean"], A_worst_case=c["A_worst_case"], worst_attack=c["worst_attack"],
                 delta_R_pp=c["delta_R_pp"], clean_drop_pp=drop, accepted=ok, reason=reason)
        if not ok:
            self.reg.set_status(v, "rejected")
        return ok, c

    def escalate(self, msg):
        (self.log_path.parent / "ESCALATION.txt").write_text(
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  HUMAN REVIEW REQUIRED\n{msg}\nServing: {self.reg.current}\n")

    def run(self):
        a, status, rejections = self.cfg["agent"], "RUNNING", 0
        for cycle in range(a["max_cycles"] + 1):
            cur = self.reg.current
            obs, stage = self.probe(cur, a["cheap_probes"]), "cheap"
            if self.ref_clean is None:
                self.ref_clean = obs["A_clean"]
            dec = self.decide(obs)
            if dec != "TRIGGER_RETRAINING":                       # healthy so far: confirm with ALL probes
                obs, stage = self.probe(cur, a["full_probes"]), "full"
                dec = self.decide(obs)
            self.log(event="evaluation", cycle=cycle, model_version=cur, stage=stage, probes_run=obs["probes"],
                     A_clean=obs["A_clean"], A_worst_case=obs["A_worst_case"], worst_attack=obs["worst_attack"],
                     per_attack_acc=obs["acc"], delta_R_pp=obs["delta_R_pp"], delta_R_se=obs["se"],
                     tau_pp=a["tau_pp"], tau_mode=a["tau_mode"], decision=dec)
            if dec != "TRIGGER_RETRAINING":
                status = "HEALTHY" if dec == "CONTINUE_MONITORING" else "HEALTHY_WITHIN_UNCERTAINTY"
                break
            if cycle == a["max_cycles"]:
                status = "BUDGET_EXHAUSTED_ABOVE_TAU"; self.escalate("Retraining budget exhausted, dR still above tau."); break
            full, accepted = self.probe(cur, a["full_probes"]), []
            for idx, recipe in enumerate(self.ordered_recipes()):
                v = self.train_candidate(cur, recipe, cycle + 1, idx)
                ok, c = self.validate(v, recipe, full, cycle + 1)
                if ok:
                    accepted.append((v, c, recipe))
            if accepted:
                v, c, recipe = max(accepted, key=lambda t: t[1]["A_worst_case"])
                for u, _, _ in accepted:
                    if u != v: self.reg.set_status(u, "not_selected")
                self.reg.promote(v); rejections = 0
                self.log(event="promotion", model_version=v, recipe=recipe,
                         epochs=self.cfg["trades"]["epochs_per_cycle"], A_worst_case=c["A_worst_case"])
            else:
                rejections += 1
                self.log(event="rollback", kept=cur, reason="no candidate passed validation")
                if rejections >= a["max_consecutive_rejections"]:
                    status = "ESCALATE_TO_HUMAN"; self.escalate("No retraining candidate passed validation."); break
        self.log(event="terminal", status=status, final_version=self.reg.current)
        print(f"[agent] finished: {status}, serving {self.reg.current}")
        return status


def run(cfg, paths, seed, registry_dir=None, log_path=None):
    set_seed(seed)
    reg = ModelRegistry(registry_dir or paths.registry)
    ds = DataStore(paths.processed, cfg["data"]["max_per_split"], cfg["data"]["split_seed"])
    return RobustnessAgent(cfg, reg, ds, get_device(), log_path or paths.agent_log, seed).run()


if __name__ == "__main__":
    ap = base_parser("Run the autonomous robustness agent")
    ap.add_argument("--registry-dir", default=None)
    ap.add_argument("--log-path", default=None)
    a = ap.parse_args()
    c = load_config(a.config, a.override)
    run(c, Paths(c, a.seed), a.seed, a.registry_dir, a.log_path)