#!/usr/bin/env bash
set -euo pipefail
python segment/train_generalized.py --config configs/generalized_blo_pennfudan.yaml
