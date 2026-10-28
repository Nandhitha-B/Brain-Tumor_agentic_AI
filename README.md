# Agentic, attack-agnostic robustness management for brain-tumour MRI (rebuilt from scratch)

Dataset: **Figshare brain tumour dataset (Cheng et al.)**: 3,064 T1-CE slices, 233 patients, 3 classes
(meningioma / glioma / pituitary), stored as `.mat` files. Model: ResNet-18. Response: TRADES.

## 0. Read this first (honesty notes)
* **Run the smoke test first** (`bash scripts/smoke_test.sh`). The code was written without a GPU/PyTorch
  available in the authoring environment: the metric, split and data-prep logic was unit-tested, the PyTorch
  parts (attacks, TRADES, agent, evaluation) were syntax-checked but **not executed**. Fix any trivial runtime error
  the smoke test shows before launching the full run.
* **The numbers will NOT match the current manuscript** (99.5 % clean, 28 % PGD, ...). Those came from the Kaggle
  data, `eps=0.1/0.03`, a four-class head and an overlapping monitoring protocol. Everything now is new:
  Figshare, one eps (8/255), patient-level splits, agent never sees the test set. Write the paper from `results/`,
  never from the old text.
* Nothing here guarantees that robustness recovers or that dR falls below tau = 15 pp. If it doesn't, the agent ends
  with `BUDGET_EXHAUSTED_ABOVE_TAU` or `ESCALATE_TO_HUMAN` and you report that. Do not tune tau on the test set.
* "Agentic" = rule-based controller with tools, memory (event log), guardrails, rollback and escalation. It is
  not a learned policy, and the TRADES response is fixed (PGD-based). Word the paper that way (playbook 8.6/8.7).

## 1. Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # a CUDA GPU is strongly recommended
```
Download the 4 zips from Figshare (`brain tumor dataset`, Cheng 2017) and put them (or the unzipped `.mat` files) in
`data/raw/`. Pretrained ResNet-18 weights download once via torchvision (needs internet; a warning is printed if not).

## 2. Run
```bash
bash scripts/smoke_test.sh                              # ~minutes, CPU ok, tiny subset
python run_pipeline.py --seeds 0                         # one full run
bash scripts/run_all.sh                                  # seeds 0,1,2 + held-out attack + smoothing ablation + latency demo
```
Stages: `prepare -> baseline -> agent -> direct (TRADES control) -> eval (locked test) -> report`.
Outputs in `runs/seed{S}/`: `registry/` (versioned checkpoints + sha256), `logs/agent_events.jsonl`,
`results/final/<model>/{attack_metrics,per_class_metrics}.csv`, confusion matrices, per-sample predictions,
`results/results_summary.md`, `results/figures/*.png`, `deploy_demo/latency_summary.csv`.

## 3. Protocol (what each manuscript claim maps to)
| Manuscript claim | Where it is implemented / evidenced |
|---|---|
| No leakage; agent independent of test | `prepare_figshare.py`: patient-level stratified split train/val/monitor/test, asserts no patient overlap, exact-duplicate check. `agent.py` only touches `monitor`. Test used only in `evaluate.py`. |
| ResNet-18 baseline, 3 classes | `models.py`, `train_baseline.py` (best epoch chosen on `val`) |
| Bounded probes FGSM/PGD/BIM/C&W | `attacks.py`; one eps=8/255 in raw [0,1] space, normalisation inside model; budget asserted at run time in `profiler.py`; C&W-L2 projected to `l2_eps` |
| Profiling module, dR, ASR, conf. drop | `profiler.py`, `metrics.py` (standard ASR = P[adv wrong given clean correct], prediction-flip rate reported separately, dR = A_clean - min_k A_k) |
| Autonomous agent, trigger on dR > tau | `agent.py`: observe, decide (no branching on attack identity), retrain, verify, promote/rollback, escalate; JSONL audit log |
| Adaptive TRADES retraining | `trades.py` (CE and KL logged separately, loss = batch-mean) |
| Decoupled System A / B | `inference_service.py` (no attacks/training, hot-reload from registry) vs `agent.py`; `deploy_demo.py` runs them as separate processes and measures latency |
| Post-defense evidence | `evaluate.py`: baseline vs agent_final on identical test images/attacks, paired-bootstrap CIs, per-class P/R/F1, confusion matrices |
| Is the agent adding anything? | `direct_trades.py` control (same start, data, seed, hyper-parameters, epoch budget, no trigger) |
| Smoothing | OFF by default. `--smoothing-ablation` attacks the full pooled pipeline (adaptive). Otherwise delete it from the paper. |
| Unseen-attack claim | `--heldout` adds the gradient-free Square attack (not used by agent). Claim it only if the numbers support it. |
| Multi-seed | `--seeds 0 1 2` + `python -m agentic_rob.aggregate_seeds` |

## 4. Edits the manuscript needs (dataset / protocol facts)
* Dataset: 3,064 images, 233 patients, 3 classes; split counts from `data/processed/split_summary.json`
  (replace the Kaggle 7,023 / 5,712 / 1,311 text and the "four-class head" sentence). Dataset licence: check Figshare (CC BY 4.0 as listed there at time of writing; verify).
* Attack text: one eps (8/255, raw pixel space); BIM = PGD without random start (both clip to the eps-ball), so remove the "BIM lacks projection" explanation.
* Report `results/final/*` tables, not projected curves. `fig_agent_trajectory.png` is measured on the monitor set; `fig_trades_loss.png` is training loss only.
* Report the agent's terminal status honestly and tau = 15 pp as an operational, uncalibrated choice (`Threshold sensitivity` table in the summary).
* Latency/hardware claims only from `deploy_demo/latency_summary.csv` and `logs/env.json`.

## 5. Knobs (configs/default.yaml)
`attack.eps_255`, `probes_*`, `cw.l2_eps` (3.0 is a placeholder budget: choose and justify it), `trades.*`,
`agent.tau_pp / max_cycles / max_clean_drop_pp`, `baseline.epochs/augment`. Time budget guide on one modern GPU: baseline minutes;
each TRADES cycle (5 epochs x 10 inner steps) roughly 10-20 min; final evaluation (C&W 3x100 steps) several minutes per model.
