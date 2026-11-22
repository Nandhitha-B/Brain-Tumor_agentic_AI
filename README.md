# Agentic Adversarial Robustness Management for Brain-Tumour MRI Classification

An Agentic closed-loop framework that monitors the adversarial robustness of a brain-tumour MRI classifier, decides when it has degraded, retrains it with adversarial fine-tuning, validates the result, and promotes or rolls back the model, all with an auditable event log. Inference and robustness optimisation run as two decoupled systems connected by a versioned model registry.

**Task:** 3-class brain-tumour classification (meningioma, glioma, pituitary) on T1-weighted contrast-enhanced MRI
**Backbone:** ResNet-18 (ImageNet-pretrained) **Response:** TRADES adversarial fine-tuning **Dataset:** Figshare brain-tumour dataset (Cheng et al.)

---

## Contents

1. [Overview](#overview)
2. [Repository layout](#repository-layout)
3. [Installation](#installation)
4. [Dataset preparation](#dataset-preparation)
5. [Running the pipeline](#running-the-pipeline)
6. [Configuration](#configuration)
7. [Outputs](#outputs)
8. [Reference results](#reference-results)
9. [Testing](#testing)
10. [Citation and acknowledgements](#citation-and-acknowledgements)

---

## Overview

```
                 ┌──────────────── System A: inference ────────────────┐
  MRI slice ───► │  (optional pre-filter) ─► ResNet-18 (current version) │ ───► class + confidence
                 └──────────────────────────▲───────────────────────────┘
                                            │ hot-reload of promoted version
                                   ┌────────┴────────┐
                                   │  Model registry │  versioned checkpoints, sha256, lineage
                                   └────────▲────────┘
                                            │ register / promote / rollback
   ┌──────────────────────── System B: robustness management ───────────────────────┐
   │  monitor split ─► probe (cheap ─► full) ─► trigger? ─► train recipe candidates   │
   │        ▲                                        │          │                     │
   │        └──────── audit log ◄── promote / rollback / escalate ◄── validate ──────┘
   └──────────────────────────────────────────────────────────────────────────────────┘
```

**Components**

| Component | Description |
|---|---|
| Baseline classifier | ResNet-18, 3-way head, normalisation inside the model so that all perturbation budgets are in raw pixel space |
| Robustness profiler | Evaluates a model under bounded perturbation probes (FGSM, PGD, BIM, C&W-L2) and reports accuracy, attack success rate, confidence drop and the robustness gap ΔR = A_clean − min_k A_k |
| Autonomous agent | Budgeted probing (cheap probes first, full set to confirm), statistically-aware trigger, recipe search with memory, validation guardrails, promotion / rollback / escalation, JSONL audit log |
| Adversarial response | TRADES fine-tuning (PGD inner maximisation) with ε and β warm-up; a fixed recipe family (β ∈ {6, 3, 1}) searched by the agent |
| Inference service | Serves the current registry version, performs no attacks or training, hot-reloads on promotion |
| Evaluation suite | Locked test-set evaluation with seven additional attack and shift analyses (see [Outputs](#outputs)) |

**Data protocol.** Images are split by patient, stratified by label, into train / validation / monitor / test (≈ 60 / 10 / 10 / 20 %). The agent only ever sees the monitor split; the test split is used once, for final evaluation. Patient overlap between splits is asserted at preparation time.

**Attack probes.** One L∞ budget (ε = 2/255) for FGSM / PGD / BIM / PGD with restarts / APGD (CE and DLR) / Square, and an L2 budget (‖δ‖₂ ≤ 1.0) for C&W. Budgets are checked at run time. The agent uses FGSM, PGD, BIM and C&W; the remaining attacks are used only for final evaluation.

---

## Repository layout

```
agentic_robustness/
├── agentic_rob/
│   ├── config.py               config loading, seeding, device, paths
│   ├── data/
│   │   ├── prepare_figshare.py .mat → uint8 array, manifest, patient-level splits
│   │   └── dataset.py          in-memory dataset and batching
│   ├── models.py               ResNet-18 wrapper, optional pre-filters
│   ├── attacks.py              FGSM, PGD, BIM, C&W-L2, PGD-RS, Square, APGD wrappers
│   ├── metrics.py              ASR, confidence drop, ΔR, CIs, bootstrap
│   ├── profiler.py             robustness profiling module
│   ├── trades.py               TRADES / PGD-AT fine-tuning
│   ├── registry.py             versioned model registry
│   ├── train_baseline.py       baseline training
│   ├── agent.py                autonomous robustness agent (System B)
│   ├── inference_service.py    inference service (System A)
│   ├── direct_trades.py        no-agent control models
│   ├── evaluate.py             final test-set evaluation
│   ├── shift_eval.py           corruption (distribution-shift) evaluation
│   ├── eps_sweep.py            accuracy vs perturbation budget
│   ├── threshold_analysis.py   trigger-threshold calibration on the monitor split
│   ├── report.py               tables and figures
│   ├── deploy_demo.py          decoupled-deployment latency demo
│   └── aggregate_seeds.py      mean ± SD across training seeds
├── configs/
│   ├── default.yaml
│   └── quick.yaml              tiny configuration for smoke tests
├── scripts/                    run_all.sh, smoke_test.sh
├── tests/
├── run_pipeline.py             end-to-end entry point
└── requirements.txt
```

---

## Installation

Requires Python 3.10–3.12. A CUDA GPU is strongly recommended.

**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (Command Prompt / PowerShell)**
```bat
python -m venv .venv
.venv\Scripts\activate
```

Install PyTorch first with the build matching your CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/)), for example:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install torchattacks        # optional: APGD attacks (--apgd)
```

Verify GPU access:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Pretrained ResNet-18 weights are downloaded once by torchvision on first use.

---

## Dataset preparation

1. Download the four archives of the **Figshare brain tumour dataset** (Cheng et al.): 3,064 T1-weighted contrast-enhanced slices from 233 patients — meningioma (708), glioma (1,426), pituitary tumour (930), stored as `.mat` files.
2. Place the `.zip` files (or the extracted `.mat` files) in `data/raw/`.
3. Run:
   ```bash
   python -m agentic_rob.data.prepare_figshare
   ```

This produces, in `data/processed/`:

| File | Content |
|---|---|
| `images.npy` | 224 × 224 uint8 slices (percentile-clipped, min-max scaled, bilinear resize) |
| `manifest.csv` | file, patient ID, label (0 meningioma, 1 glioma, 2 pituitary), split |
| `split_summary.json` | images, patients and class counts per split; cross-split duplicate check |

The pipeline runs this step automatically if `images.npy` is missing.

---

## Running the pipeline

**Smoke test** (tiny subset, a few minutes; numbers are not meaningful):
```bash
python -m pytest tests -q
python run_pipeline.py --override configs/quick.yaml --seeds 0 --heldout --smoothing-ablation --stress
```
Delete the generated `runs/` directory afterwards.

**Full run (one training seed)**
```bash
python run_pipeline.py --seeds 0 --heldout --smoothing-ablation --stress --apgd
```

**Pipeline stages:** `prepare → baseline → agent → direct → eval → shift → report`
Select stages with `--stages`, for example `--stages eval shift report` to re-evaluate existing checkpoints.

**Additional analyses**
```bash
python -m agentic_rob.eps_sweep --seed 0            # PGD accuracy vs ε for all models
python -m agentic_rob.threshold_analysis --seed 0   # trigger threshold calibration (monitor split)
python -m agentic_rob.deploy_demo --seed 0          # decoupled inference latency during retraining
python -m agentic_rob.aggregate_seeds runs          # mean ± SD if several seeds were run
```

**Command-line flags of `run_pipeline.py`**

| Flag | Effect |
|---|---|
| `--seeds 0 1 2` | training seeds to run |
| `--override FILE` | YAML overriding `configs/default.yaml` |
| `--stages …` | subset of stages |
| `--heldout` | add the gradient-free Square attack to the final evaluation |
| `--apgd` | add PGD with restarts, APGD-CE and APGD-DLR (requires `torchattacks`) |
| `--smoothing-ablation` | evaluate 3 × 3 average-pooling and bit-depth pre-filters with adaptive attacks |
| `--stress` | additionally evaluate at a larger budget (`eval.stress_eps_255`) |
| `--deploy-demo` | run the latency demo after each seed |

---

## Configuration

All settings are in `configs/default.yaml`. Principal options:

| Section | Key | Default | Meaning |
|---|---|---|---|
| `attack` | `eps_255` | 2 | L∞ budget (×1/255), raw pixel space; keep equal to `trades.eps_255` |
| `probes_monitor` / `probes_final` | `pgd`, `bim`, `cw`, `square`, `apgd`, `pgd_rs` | | probe strength used by the agent / final evaluation |
| `trades` | `epochs_per_cycle`, `warmup_epochs`, `lr`, `beta`, `steps` | 20, 5, 2e-4, 6, 10 | adversarial fine-tuning |
| `trades` | `recipes`, `n_candidates` | β ∈ {6, 3, 1}, 3 | agent's recipe search space |
| `agent` | `tau_pp` | 15 | robustness-gap threshold (percentage points) |
| `agent` | `tau_mode`, `tau_z` | adaptive, 1.64 | trigger on ΔR − z·SE > τ (`fixed`: ΔR > τ) |
| `agent` | `max_cycles`, `max_clean_drop_pp`, `min_gain_pp` | 3, 5, 2 | retraining budget and validation guardrails |
| `agent` | `cheap_probes`, `full_probes` | [fgsm, pgd], [fgsm, pgd, bim, cw] | budgeted probing |
| `inference` | `smoothing` | false | optional input pre-filter in System A |
| `data` | `split_seed`, `n_folds`, `test_folds`, `val_folds`, `monitor_folds` | 42, 10, 2, 1, 1 | patient-level split |

---

## Outputs

Everything for a training seed is written to `runs/seed{N}/`:

```
runs/seed0/
├── registry/                      versioned checkpoints and registry.json (sha256, lineage, status)
├── logs/
│   ├── agent_events.jsonl         audit log: probes, ΔR (± SE), decisions, recipes, validation, promotion
│   ├── baseline_history.csv
│   ├── *_history.json             training histories of the agent's candidates and control models
│   ├── ESCALATION.txt             written only if the agent escalates to a human
│   └── env.json                   hardware and library versions
└── results/
    ├── final/<model>/             attack_metrics.csv, per_class_metrics.csv, confusion_*.csv,
    │                              summary.json, per_sample.npz   (baseline, agent_final,
    │                              direct_trades, madry_at, and +smoothing / +bitdepth / @eps variants)
    ├── final/compare_*.csv        paired-bootstrap differences between models
    ├── eps_sweep.csv, shift_results.csv, threshold_analysis.csv, agent_trajectory.csv
    ├── results_summary.md         auto-generated tables
    └── figures/                   PNG figures
```

Reported metrics: accuracy with Wilson 95 % CIs, attack success rate (adversarial prediction wrong given clean prediction correct), prediction-flip rate, mean confidence drop, robustness gap ΔR, per-class precision / recall / F1, confusion matrices, and paired-bootstrap CIs for differences between models.

---

## Reference results

Reference run, training seed 0, ε = 2/255, locked test split (n = 613), 95 % CIs in brackets. "Agent final" is the model promoted by the agent.

| | Baseline | Agent final |
|---|---|---|
| Clean accuracy (%) | 96.08 [94.24, 97.36] | 93.47 [91.24, 95.17] |
| PGD accuracy (%) | 0.98 [0.45, 2.12] | 85.97 [83.00, 88.50] |
| C&W-L2 accuracy (%) | 3.43 [2.25, 5.18] | 83.36 [80.21, 86.10] |
| APGD-CE accuracy (%) | 0.65 | 85.97 |
| Robustness gap ΔR (pp) | 95.11 | 10.11 |

Training is deterministic: re-running a seed reproduces the reported numbers exactly.

---

## Testing

```bash
python -m pytest tests -q
```
The tests cover metric definitions, patient-level splitting and attack budget constraints (the latter require PyTorch).

---

## Citation and acknowledgements

If you use this code, please cite the accompanying paper (reference to be added on publication).

**Dataset.** Cheng, J. *Brain tumor dataset*. Figshare. Please cite the dataset according to the citation instructions on its Figshare record and observe its licence.

**Methods implemented:** FGSM (Goodfellow et al., 2015), PGD (Madry et al., 2018), BIM (Kurakin et al., 2017), C&W (Carlini & Wagner, 2017), TRADES (Zhang et al., 2019), Square Attack (Andriushchenko et al., 2020), APGD (Croce & Hein, 2020, via `torchattacks`).

**License.** -
