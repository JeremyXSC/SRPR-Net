#!/usr/bin/env bash
set -euo pipefail
python segment/val_generalized.py \
  --config configs/generalized_blo_pennfudan.yaml \
  --checkpoint runs/generalized_blo_pennfudan/best_generalized.pt \
  --output runs/generalized_blo_pennfudan/final_test
