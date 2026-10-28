"""Decoupling demo / latency measurement. System A serves requests while System B (the agent) runs as a
SEPARATE PROCESS on a copy of the registry that contains only the baseline. Logs per-request latency and
which model version served each request; System A hot-reloads when the agent promotes a new version."""
from __future__ import annotations
import json, subprocess, sys, time
import numpy as np
import pandas as pd
import torch
from agentic_rob.config import base_parser, load_config, Paths
from agentic_rob.data.dataset import DataStore, iterate
from agentic_rob.inference_service import InferenceService
from agentic_rob.models import build_model
from agentic_rob.registry import ModelRegistry


def run(cfg, paths, seed, override=None, config="configs/default.yaml"):
    demo = paths.run / "deploy_demo"; (demo / "registry").mkdir(parents=True, exist_ok=True)
    src = ModelRegistry(paths.registry)
    reg = ModelRegistry(demo / "registry")
    if reg.current is None:                       # fresh registry holding only the baseline
        m = src.load_into(build_model(cfg, len(cfg["classes"])), src.first_of_kind("baseline"))
        reg.promote(reg.register(m, kind="baseline"))
    ds = DataStore(paths.processed, cfg["data"]["max_per_split"], cfg["data"]["split_seed"])
    X, y = ds.split("test")
    dev = cfg["deploy"]["inference_device"]
    svc = InferenceService(cfg, demo / "registry", dev)
    batches = [xb for xb, _ in iterate(X, y, 1, torch.device(dev))]
    period, rows, i = 1.0 / cfg["deploy"]["request_rate_hz"], [], 0

    def serve(n=None, until=None):
        nonlocal i
        last_reload = time.time()
        while (n is None or n > 0) and not (until and until.poll() is not None):
            t0 = time.perf_counter(); _, _, v = svc.predict(batches[i % len(batches)]); lat = (time.perf_counter() - t0) * 1000
            rows.append(dict(t=time.time(), phase=phase, latency_ms=lat, version=v)); i += 1
            if n is not None: n -= 1
            if time.time() - last_reload > cfg["deploy"]["reload_interval_s"]:
                if svc.maybe_reload(): print(f"[System A] reloaded -> {svc.version}")
                last_reload = time.time()
            time.sleep(max(0, period - (time.perf_counter() - t0)))

    phase = "idle"; serve(n=100)
    cmd = [sys.executable, "-m", "agentic_rob.agent", "--config", config, "--seed", str(seed),
           "--registry-dir", str(demo / "registry"), "--log-path", str(demo / "agent_events.jsonl")]
    if override: cmd += ["--override", override]
    proc = subprocess.Popen(cmd); phase = "during_retraining"; serve(until=proc)
    svc.maybe_reload(); phase = "after"; serve(n=50)
    df = pd.DataFrame(rows); df.to_csv(demo / "latency_log.csv", index=False)
    summ = df.groupby("phase").latency_ms.describe(percentiles=[.5, .95]).round(2)
    summ.to_csv(demo / "latency_summary.csv"); print(summ)
    print("versions served:", df.version.value_counts().to_dict(), "| inference device:", dev)


if __name__ == "__main__":
    a = base_parser("Decoupled deployment demo").parse_args()
    c = load_config(a.config, a.override)
    run(c, Paths(c, a.seed), a.seed, a.override, a.config)
