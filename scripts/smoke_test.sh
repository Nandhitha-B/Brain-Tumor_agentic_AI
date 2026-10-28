#!/usr/bin/env bash
# Tiny CPU-friendly end-to-end check of the code path (numbers are meaningless).
set -e
python -m pytest tests -q
python run_pipeline.py --override configs/quick.yaml --seeds 0 --heldout --smoothing-ablation
