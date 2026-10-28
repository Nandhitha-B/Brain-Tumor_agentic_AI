#!/usr/bin/env bash
# Full paper run: 3 training seeds, held-out attack, smoothing ablation, deployment demo (seed 0 only).
set -e
python run_pipeline.py --seeds 0 1 2 --heldout --smoothing-ablation
python -m agentic_rob.deploy_demo --seed 0
python -m agentic_rob.aggregate_seeds runs
